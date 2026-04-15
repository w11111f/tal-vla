import json
import os
import re
import shutil
import subprocess
from copy import deepcopy

from src.envs import approx
from src.utils.misc import convertToDGLGraph


DEFAULT_SYSTEM_PROMPT = (
    'You are a robot task planning assistant. '
    'You must return valid JSON only.'
)

DEFAULT_DASHSCOPE_URL = (
    'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'
)

DEFAULT_USER_PROMPT = (
    "\u4f60\u662f\u4e00\u4e2a\u673a\u5668\u4eba\u4efb\u52a1\u89c4\u5212\u52a9\u624b\u3002"
    "\u5f53\u524d\u73af\u5883\u7684\u573a\u666f\u56feJSON\u5982\u4e0b\uff1a\n"
    "{scene_graph_json}\n\n"
    "\u7528\u6237\u7684\u81ea\u7136\u8bed\u8a00\u6307\u4ee4\u662f\uff1a'{instruction}'\u3002\n\n"
    "\u8bf7\u6839\u636e\u6307\u4ee4\uff0c\u63a8\u6f14\u4efb\u52a1\u5b8c\u6210\u540e\u73af\u5883\u4f1a"
    "\u53d1\u751f\u4ec0\u4e48\u53d8\u5316\uff0c\u5e76\u8f93\u51fa\u4fee\u6539\u540e\u7684\u76ee\u6807"
    "\u573a\u666f\u56feJSON\u3002\n"
    "\u8bf7\u4fdd\u6301\u4e0e\u8f93\u5165\u517c\u5bb9\u7684\u573a\u666f\u56fe\u7ed3\u6784\uff0c"
    "\u81f3\u5c11\u5305\u542b nodes \u548c edges \u4e24\u4e2a\u5b57\u6bb5\u3002\n"
    "\u53ea\u8f93\u51faJSON\uff0c\u4e0d\u8981\u89e3\u91ca\u3002"
)


def _last_index(items, value):
    return len(items) - items[::-1].index(value) - 1


def _resolve_scene_graph_index(datapoint, state_name=None):
    actions = list(getattr(datapoint, 'actions', []))
    metrics = list(getattr(datapoint, 'metrics', []))

    preferred_states = []
    if state_name is not None:
        preferred_states.append(state_name)
    preferred_states.extend(['End', 'Initialize', 'Start'])

    for candidate in preferred_states:
        if candidate in actions:
            return _last_index(actions, candidate)

    for idx in range(len(metrics) - 1, -1, -1):
        if metrics[idx] is not None:
            return idx

    raise ValueError(
        'Could not resolve a valid scene-graph state index from datapoint actions={}.'.format(
            actions
        )
    )


def _make_json_safe(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().tolist()
    elif hasattr(value, 'tolist'):
        value = value.tolist()
    elif isinstance(value, tuple):
        value = [_make_json_safe(item) for item in value]
    elif isinstance(value, list):
        value = [_make_json_safe(item) for item in value]
    elif isinstance(value, dict):
        value = {key: _make_json_safe(item) for key, item in value.items()}
    return value


def _canonical_object_meta(config, object_name):
    for obj in config.objects:
        if obj['name'] == object_name:
            return {
                'id': config.object2idx[object_name],
                'name': object_name,
                'properties': deepcopy(obj['properties']),
                'size': deepcopy(obj['size']),
            }
    raise KeyError('Unknown object in scene graph: {}'.format(object_name))


def _normalize_position(position, fallback_position):
    if not isinstance(position, (list, tuple)) or len(position) != 2:
        return deepcopy(fallback_position)
    xyz, orn = position
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
        xyz = fallback_position[0]
    if not isinstance(orn, (list, tuple)) or len(orn) not in [3, 4]:
        orn = fallback_position[1]
    return [list(xyz), list(orn)]


def _normalize_states(config, states, fallback_states):
    if not isinstance(states, list):
        return deepcopy(fallback_states)
    valid_states = set(config.STATES)
    filtered_states = [state for state in states if state in valid_states]
    return filtered_states if len(filtered_states) != 0 else deepcopy(fallback_states)


def _normalize_edge(config, edge):
    relation = edge.get('relation')
    if relation not in config.EDGES:
        return None

    edge_from = edge.get('from')
    edge_to = edge.get('to')
    if isinstance(edge_from, str):
        if edge_from not in config.object2idx:
            return None
        edge_from = config.object2idx[edge_from]
    if isinstance(edge_to, str):
        if edge_to not in config.object2idx:
            return None
        edge_to = config.object2idx[edge_to]
    if not isinstance(edge_from, int) or not isinstance(edge_to, int):
        return None
    return {'from': edge_from, 'to': edge_to, 'relation': relation}


def simplify_scene_graph_json(graph_data):
    simplified = {'nodes': [], 'edges': []}
    for node in graph_data['nodes']:
        simplified['nodes'].append(
            {
                'id': int(node['id']),
                'name': node['name'],
                'properties': _make_json_safe(node['properties']),
                'states': _make_json_safe(node['states']),
                'position': _make_json_safe(node['position']),
                'size': _make_json_safe(node['size']),
            }
        )
    for edge in graph_data['edges']:
        simplified['edges'].append(
            {
                'from': int(edge['from']),
                'to': int(edge['to']),
                'relation': edge['relation'],
            }
        )
    return simplified


def datapoint_to_scene_graph_json(config, datapoint, state_name=None):
    data_index = _resolve_scene_graph_index(datapoint, state_name=state_name)

    graph_key = 'graph_' + str(data_index)
    graph_data = datapoint.getGraph(index=data_index, embeddings=config.embeddings)[graph_key]
    return simplify_scene_graph_json(graph_data)


def get_current_scene_graph_json(config, state_name=None):
    datapoint = approx.get_datapoint()
    return datapoint_to_scene_graph_json(config, datapoint, state_name=state_name)


def build_scene_graph_translation_messages(current_scene_graph_json, instruction):
    prompt = DEFAULT_USER_PROMPT.format(
        scene_graph_json=json.dumps(current_scene_graph_json, ensure_ascii=False, indent=2),
        instruction=instruction,
    )
    return [
        {'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt},
    ]


def _extract_response_text(response_dict):
    output = response_dict.get('output', response_dict)
    choices = output.get('choices', [])
    if len(choices) == 0:
        raise RuntimeError('DashScope response does not contain choices: {}'.format(response_dict))

    message = choices[0].get('message', {})
    content = message.get('content', '')
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text_parts.append(item.get('text', ''))
            else:
                text_parts.append(str(item))
        return ''.join(text_parts)
    return content


def _parse_json_response(text):
    stripped = text.strip()
    code_block = re.search(r'```(?:json)?\s*(\{.*\})\s*```', stripped, flags=re.DOTALL)
    if code_block is not None:
        stripped = code_block.group(1)
    else:
        start = stripped.find('{')
        end = stripped.rfind('}')
        if start != -1 and end != -1 and end > start:
            stripped = stripped[start:end + 1]
    return json.loads(stripped)


def _post_json_with_curl(url, payload, api_key):
    curl_bin = shutil.which('curl.exe') or shutil.which('curl')
    if curl_bin is None:
        raise RuntimeError('curl was not found on PATH.')

    command = [
        curl_bin,
        '--location',
        url,
        '--header',
        'Authorization: Bearer {}'.format(api_key),
        '--header',
        'Content-Type: application/json',
        '--data',
        json.dumps(payload, ensure_ascii=False)
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            'curl request failed with code {}: {}'.format(
                completed.returncode,
                completed.stderr.strip()
            )
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            'DashScope returned non-JSON output: {}'.format(completed.stdout[:500])
        ) from exc


def translate_scene_graph_with_qwen(
        current_scene_graph_json,
        instruction,
        model_name='qwen3-max',
        api_key=None
):
    api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
    if api_key is None or str(api_key).strip() == '':
        raise ValueError(
            'DashScope API key not found. Set DASHSCOPE_API_KEY or pass --qwen_api_key.'
        )

    messages = build_scene_graph_translation_messages(current_scene_graph_json, instruction)
    payload = {
        'model': model_name,
        'input': {
            'messages': messages
        },
        'parameters': {
            'result_format': 'message'
        }
    }
    response = _post_json_with_curl(DEFAULT_DASHSCOPE_URL, payload, api_key)
    if 'code' in response and response.get('code') not in [None, '']:
        raise RuntimeError('DashScope request failed: {}'.format(response))

    response_text = _extract_response_text(response)
    return _parse_json_response(response_text)


def canonicalize_scene_graph_json(config, current_scene_graph_json, generated_scene_graph_json):
    canonical_graph = deepcopy(current_scene_graph_json)
    canonical_nodes = {node['name']: deepcopy(node) for node in canonical_graph['nodes']}
    generated_nodes = generated_scene_graph_json.get('nodes', [])

    for generated_node in generated_nodes:
        object_name = generated_node.get('name')
        if object_name not in config.object2idx:
            continue

        if object_name in canonical_nodes:
            base_node = canonical_nodes[object_name]
        else:
            base_node = _canonical_object_meta(config, object_name)
            base_node.update({'states': [], 'position': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]})

        merged_node = deepcopy(base_node)
        merged_node['id'] = config.object2idx[object_name]
        merged_node['name'] = object_name
        merged_node['properties'] = _canonical_object_meta(config, object_name)['properties']
        merged_node['size'] = _canonical_object_meta(config, object_name)['size']
        merged_node['states'] = _normalize_states(
            config,
            generated_node.get('states'),
            base_node.get('states', [])
        )
        merged_node['position'] = _normalize_position(
            generated_node.get('position'),
            base_node.get('position', [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        )
        canonical_nodes[object_name] = merged_node

    canonical_graph['nodes'] = sorted(canonical_nodes.values(), key=lambda node: node['id'])

    generated_edges = generated_scene_graph_json.get('edges')
    if isinstance(generated_edges, list):
        normalized_edges = []
        for edge in generated_edges:
            normalized_edge = _normalize_edge(config, edge)
            if normalized_edge is not None:
                normalized_edges.append(normalized_edge)
        canonical_graph['edges'] = normalized_edges

    return canonical_graph


def scene_graph_json_to_dgl(config, scene_graph_json):
    graph_data = {'nodes': [], 'edges': []}
    for node in scene_graph_json['nodes']:
        object_name = node['name']
        meta = _canonical_object_meta(config, object_name)
        graph_data['nodes'].append(
            {
                'id': config.object2idx[object_name],
                'name': object_name,
                'properties': meta['properties'],
                'states': deepcopy(node['states']),
                'position': deepcopy(node['position']),
                'size': meta['size'],
                'vector': _make_json_safe(config.object2vec[object_name]),
            }
        )
    for edge in scene_graph_json['edges']:
        normalized_edge = _normalize_edge(config, edge)
        if normalized_edge is not None:
            graph_data['edges'].append(normalized_edge)
    return convertToDGLGraph(config, graph_data, False, -1)


def translate_instruction_to_goal_state_graph(
        config,
        instruction,
        current_scene_graph_json=None,
        model_name='qwen3-max',
        api_key=None
):
    if current_scene_graph_json is None:
        current_scene_graph_json = get_current_scene_graph_json(config)

    raw_goal_scene_graph_json = translate_scene_graph_with_qwen(
        current_scene_graph_json=current_scene_graph_json,
        instruction=instruction,
        model_name=model_name,
        api_key=api_key
    )
    goal_scene_graph_json = canonicalize_scene_graph_json(
        config,
        current_scene_graph_json=current_scene_graph_json,
        generated_scene_graph_json=raw_goal_scene_graph_json
    )
    goal_state_graph = scene_graph_json_to_dgl(config, goal_scene_graph_json)
    return current_scene_graph_json, goal_scene_graph_json, goal_state_graph
