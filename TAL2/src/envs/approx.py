from copy import deepcopy
import math

import torch
from scipy.spatial import distance

from src.envs.datapoint import Datapoint
from src.envs.utils_env import checkIn, checkOn, convertToDGLGraph, findConstraintTo


datapoint = None
goal_spec = None
config_cache = None
_isaac_backend = None

metrics = {}
constraints = {"husky": []}

sticky = []
fixed = []
on = []
fueled = []
cut = []
cleaner = False
stick = False
clean = []
drilled = []
welded = []
painted = []


def _use_isaac_backend(config):
    backend = str(getattr(config, "policy_backend", "symbolic")).lower()
    return backend == "isaaclab"


def _get_isaac_backend():
    global _isaac_backend
    if _isaac_backend is None:
        from src.envs import isaac_env
        _isaac_backend = isaac_env
    return _isaac_backend


def _default_orientation():
    return [0.0, 0.0, 0.0, 1.0]


def _object_size(config, obj_name):
    obj_entry = config.get_object_entry(obj_name)
    if obj_entry is None:
        return [0.1, 0.1, 0.1]
    return list(obj_entry.get("size", [0.1, 0.1, 0.1]))


def _default_layout(config):
    layout_xy = {
        "husky": (0.0, 0.0),
        "cube_red": (0.8, 0.2),
        "tray": (1.4, -0.1),
        "big-tray": (2.2, 0.5),
        "bottle_red": (0.9, 0.9),
        "stool": (1.8, -0.9),
        "table": (0.2, 1.8),
    }
    default_metrics = {}
    for obj_name in config.all_objects:
        size = _object_size(config, obj_name)
        x, y = layout_xy.get(obj_name, (0.0, 0.0))
        z = max(size[2] / 2.0, 0.01)
        default_metrics[obj_name] = [[x, y, z], _default_orientation()]
    return default_metrics


def _copy_metric_dict(src_metrics, config):
    copied = deepcopy(src_metrics) if src_metrics is not None else {}
    for obj_name in config.all_objects:
        if obj_name not in copied:
            copied[obj_name] = deepcopy(_default_layout(config)[obj_name])
    return copied


def _first_valid_metric_snapshot(input_datapoint, config):
    if input_datapoint is None:
        return None
    for snapshot in input_datapoint.metrics:
        if isinstance(snapshot, dict) and len(snapshot) != 0:
            return _copy_metric_dict(snapshot, config)
    return None


def _last_valid_index(input_datapoint):
    if input_datapoint is None:
        return None
    preferred_actions = ["End", "Initialize", "Start"]
    for action_name in preferred_actions:
        if action_name in input_datapoint.actions:
            return len(input_datapoint.actions) - 1 - input_datapoint.actions[::-1].index(action_name)
    for idx in range(len(input_datapoint.actions) - 1, -1, -1):
        if input_datapoint.metrics[idx] is not None:
            return idx
    return None


def _position_from_metrics(metric_dict):
    husky_metric = metric_dict.get("husky", [[0.0, 0.0, 0.0], _default_orientation()])
    husky_pos = list(husky_metric[0])
    return [husky_pos[0], husky_pos[1], husky_pos[2], 0.0]


def _snapshot_from_datapoint(config, input_datapoint):
    idx = _last_valid_index(input_datapoint)
    if idx is None:
        base_metrics = deepcopy(config.initial_object_metrics) if len(config.initial_object_metrics) != 0 else _default_layout(config)
        return {
            "metrics": _copy_metric_dict(base_metrics, config),
            "constraints": {"husky": []},
            "sticky": [],
            "fixed": [],
            "cleaner": False,
            "on": [],
            "clean": [],
            "stick": False,
            "welded": [],
            "drilled": [],
            "painted": [],
            "fueled": [],
            "cut": [],
            "position": _position_from_metrics(base_metrics),
            "world": getattr(input_datapoint, "world", config.graph_world_name),
            "goal": getattr(input_datapoint, "goal", config.exploration_goal_name),
        }

    snapshot_metrics = _copy_metric_dict(input_datapoint.metrics[idx], config)
    snapshot_constraints = input_datapoint.constraints[idx]
    if snapshot_constraints is None:
        snapshot_constraints = {"husky": []}
    else:
        snapshot_constraints = deepcopy(snapshot_constraints)
    snapshot_constraints.setdefault("husky", [])

    snapshot_position = input_datapoint.position[idx]
    if snapshot_position is None:
        snapshot_position = _position_from_metrics(snapshot_metrics)
    else:
        snapshot_position = deepcopy(snapshot_position)

    return {
        "metrics": snapshot_metrics,
        "constraints": snapshot_constraints,
        "sticky": deepcopy(input_datapoint.sticky[idx]) if input_datapoint.sticky[idx] is not None else [],
        "fixed": deepcopy(input_datapoint.fixed[idx]) if input_datapoint.fixed[idx] is not None else [],
        "cleaner": deepcopy(input_datapoint.cleaner[idx]) if input_datapoint.cleaner[idx] is not None else False,
        "on": deepcopy(input_datapoint.on[idx]) if input_datapoint.on[idx] is not None else [],
        "clean": deepcopy(input_datapoint.clean[idx]) if input_datapoint.clean[idx] is not None else [],
        "stick": deepcopy(input_datapoint.stick[idx]) if input_datapoint.stick[idx] is not None else False,
        "welded": deepcopy(input_datapoint.welded[idx]) if input_datapoint.welded[idx] is not None else [],
        "drilled": deepcopy(input_datapoint.drilled[idx]) if input_datapoint.drilled[idx] is not None else [],
        "painted": deepcopy(input_datapoint.painted[idx]) if input_datapoint.painted[idx] is not None else [],
        "fueled": deepcopy(input_datapoint.fueled[idx]) if input_datapoint.fueled[idx] is not None else [],
        "cut": deepcopy(input_datapoint.cut[idx]) if input_datapoint.cut[idx] is not None else [],
        "position": snapshot_position,
        "world": getattr(input_datapoint, "world", config.graph_world_name),
        "goal": getattr(input_datapoint, "goal", config.exploration_goal_name),
    }


def _sync_runtime_from_datapoint(config, runtime_datapoint):
    global datapoint, metrics, constraints
    global sticky, fixed, on, fueled, cut, cleaner, stick, clean, drilled, welded, painted

    datapoint = deepcopy(runtime_datapoint)
    snapshot = _snapshot_from_datapoint(config, datapoint)
    metrics = snapshot["metrics"]
    constraints = snapshot["constraints"]
    sticky = snapshot["sticky"]
    fixed = snapshot["fixed"]
    cleaner = snapshot["cleaner"]
    on = snapshot["on"]
    clean = snapshot["clean"]
    stick = snapshot["stick"]
    welded = snapshot["welded"]
    drilled = snapshot["drilled"]
    painted = snapshot["painted"]
    fueled = snapshot["fueled"]
    cut = snapshot["cut"]


def _held_object():
    held = constraints.get("husky", [])
    if isinstance(held, (list, tuple)) and len(held) != 0:
        return held[0]
    return None


def _grounded_height(config, obj_name):
    obj_size = _object_size(config, obj_name)
    init_metric = config.initial_object_metrics.get(obj_name)
    if init_metric is not None:
        return max(init_metric[0][2], obj_size[2] / 2.0 + 0.01)
    return max(obj_size[2] / 2.0 + 0.01, 0.01)


def _set_object_pose(obj_name, position, orientation=None):
    if orientation is None:
        orientation = metrics.get(obj_name, [None, _default_orientation()])[1]
        if orientation is None:
            orientation = _default_orientation()
    metrics[obj_name] = [list(position), list(orientation)]


def _sync_held_object(config):
    held_obj = _held_object()
    if held_obj is None:
        return
    husky_pos = list(metrics["husky"][0])
    held_size = _object_size(config, held_obj)
    obj_pos = [husky_pos[0], husky_pos[1], husky_pos[2] + max(held_size[2], 0.2)]
    _set_object_pose(held_obj, obj_pos)


def _move_robot_near(config, target_name, approach_distance=None):
    if target_name not in metrics:
        raise Exception("Logical Rule: Unknown target {}.".format(target_name))

    if approach_distance is None:
        approach_distance = getattr(config, "base_approach_distance", 0.50)
    approach_distance = max(float(approach_distance), 0.0)

    robot_pos = list(metrics["husky"][0])
    target_pos = list(metrics[target_name][0])
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]
    planar_dist = math.sqrt(dx * dx + dy * dy)

    if planar_dist < 1e-6:
        new_x = target_pos[0] - approach_distance
        new_y = target_pos[1]
    else:
        scale = approach_distance / planar_dist
        new_x = target_pos[0] - dx * scale
        new_y = target_pos[1] - dy * scale

    _set_object_pose("husky", [new_x, new_y, robot_pos[2]])
    _sync_held_object(config)


def _place_on_target(config, obj_name, target_name):
    obj_size = _object_size(config, obj_name)
    target_size = _object_size(config, target_name)
    target_pos = list(metrics[target_name][0])
    place_pos = [target_pos[0], target_pos[1], target_pos[2]]

    if target_name in config.property2Objects.get("Container", []):
        place_pos[2] = target_pos[2] + max(target_size[2] * 0.15, obj_size[2] / 2.0 + 0.01)
    else:
        place_pos[2] = target_pos[2] + max(target_size[2] * 0.9, target_size[2] / 2.0 + obj_size[2] / 2.0 + 0.02)

    _set_object_pose(obj_name, place_pos)


def _add_state_point(action_name):
    robot_pos = list(metrics["husky"][0])
    datapoint.addPoint(
        [robot_pos[0], robot_pos[1], robot_pos[2], 0.0],
        deepcopy(sticky),
        deepcopy(fixed),
        deepcopy(cleaner),
        deepcopy(action_name),
        deepcopy(constraints),
        deepcopy(metrics),
        deepcopy(on),
        deepcopy(clean),
        deepcopy(stick),
        deepcopy(welded),
        deepcopy(drilled),
        deepcopy(painted),
        deepcopy(fueled),
        deepcopy(cut),
    )


def _object_has_state(config, obj_name, normalized_state):
    if obj_name not in metrics:
        return False

    if normalized_state == "inside":
        for container_name in config.property2Objects.get("Container", []):
            if container_name == obj_name or container_name not in metrics:
                continue
            obj_meta = config.get_object_entry(obj_name)
            container_meta = config.get_object_entry(container_name)
            if checkIn(obj_name, container_name, obj_meta, container_meta, metrics, constraints):
                return True
        return False

    if normalized_state == "outside":
        return not _object_has_state(config, obj_name, "inside")

    if normalized_state == "grabbed":
        return _held_object() == obj_name

    if normalized_state == "free":
        return _held_object() != obj_name

    if normalized_state == "different_height":
        return abs(metrics[obj_name][0][2] - metrics["husky"][0][2]) > 1.0

    if normalized_state == "same_height":
        return not _object_has_state(config, obj_name, "different_height")

    return False


def _check_target_relation(config, obj_name, target_name):
    if target_name not in metrics:
        return False

    target_meta = config.get_object_entry(target_name)
    obj_meta = config.get_object_entry(obj_name)

    if "Container" in target_meta.get("properties", []):
        return checkIn(obj_name, target_name, obj_meta, target_meta, metrics, constraints)
    if "Surface" in target_meta.get("properties", []):
        return checkOn(obj_name, target_name, obj_meta, target_meta, metrics, constraints)

    return findConstraintTo(obj_name, constraints) == target_name


def cg(goal_file, constraints_arg, states, on_arg, clean_arg, sticky_arg, fixed_arg, drilled_arg, welded_arg, painted_arg, verbose=False):
    del constraints_arg, states, on_arg, clean_arg, sticky_arg, fixed_arg, drilled_arg, welded_arg, painted_arg, verbose

    if not goal_file:
        return False

    if isinstance(goal_file, dict):
        goal_data = goal_file
    else:
        raise ValueError("Only dict goal_json is supported in Isaac-compatible approx.")

    for goal in goal_data.get("goals", []):
        obj_name = goal.get("object", "")
        if obj_name not in metrics:
            return False

        goal_states = goal.get("state", [])
        if isinstance(goal_states, str):
            goal_states = [goal_states]
        for raw_state in goal_states:
            normalized_state = config_cache.normalize_state_name(raw_state)
            if normalized_state is None or not _object_has_state(config_cache, obj_name, normalized_state):
                return False

        target_name = goal.get("target", "")
        if target_name != "" and not _check_target_relation(config_cache, obj_name, target_name):
            return False

        position_name = goal.get("position", "")
        if position_name != "":
            if position_name not in metrics:
                return False
            obj_pos = metrics[obj_name][0]
            goal_pos = metrics[position_name][0]
            tolerance = abs(goal.get("tolerance", 0.0))
            if distance.euclidean(obj_pos, goal_pos) > tolerance:
                return False

    return len(goal_data.get("goals", [])) != 0


def execute(config, actions, goal_file=None):
    global datapoint

    if isinstance(actions, dict):
        action_list = actions.get("actions", [])
    else:
        action_list = list(actions)

    for action in action_list:
        try:
            datapoint.addSymbolicAction([deepcopy(action)])
            act_name = action["name"]
            args = list(action.get("args", []))

            if act_name == "moveTo":
                _move_robot_near(config, args[0], config.base_approach_distance)

            elif act_name == "pick":
                target_obj = args[0]
                if target_obj not in config.property2Objects.get("Movable", []):
                    raise Exception("Logical Rule: Cannot pick {}.".format(target_obj))
                if _held_object() is not None:
                    raise Exception("Logical Rule: Gripper is not free.")
                _move_robot_near(config, target_obj, config.pick_approach_distance)
                constraints["husky"] = [target_obj]
                _sync_held_object(config)

            elif act_name == "drop":
                requested_obj = args[0]
                held_obj = _held_object()
                if held_obj is None:
                    raise Exception("Logical Rule: Gripper is free.")
                if held_obj != requested_obj:
                    raise Exception("Logical Rule: Gripper holds {}, not {}.".format(held_obj, requested_obj))
                husky_pos = list(metrics["husky"][0])
                drop_pos = [husky_pos[0], husky_pos[1], _grounded_height(config, held_obj)]
                _set_object_pose(held_obj, drop_pos)
                constraints["husky"] = []

            elif act_name == "pushTo":
                obj_a, obj_b = args
                if obj_a not in config.property2Objects.get("Movable", []):
                    raise Exception("Logical Rule: Cannot push {}.".format(obj_a))
                if obj_b not in config.place_targets:
                    raise Exception("Logical Rule: Unsupported push target {}.".format(obj_b))
                _move_robot_near(config, obj_a, config.pick_approach_distance)
                _move_robot_near(config, obj_b, config.push_approach_distance)
                _place_on_target(config, obj_a, obj_b)

            elif act_name == "pickNplaceAonB":
                obj_a, obj_b = args
                if obj_a not in config.property2Objects.get("Movable", []):
                    raise Exception("Logical Rule: Cannot pick and place {}.".format(obj_a))
                if obj_b not in config.place_targets:
                    raise Exception("Logical Rule: Cannot place on {}.".format(obj_b))
                if obj_a == obj_b:
                    raise Exception("Logical Rule: Cannot place object on itself.")
                _move_robot_near(config, obj_a, config.pick_approach_distance)
                constraints["husky"] = [obj_a]
                _sync_held_object(config)
                _move_robot_near(config, obj_b, config.place_approach_distance)
                _place_on_target(config, obj_a, obj_b)
                constraints["husky"] = []

            else:
                raise Exception("Error: Unsupported high-level action {}.".format(act_name))

            _add_state_point(action)
            _add_state_point("End")

        except Exception as exc:
            datapoint.addSymbolicAction("Error = " + str(exc))
            datapoint.addPoint(
                None,
                None,
                None,
                None,
                "Error = " + str(exc),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            raise exc

    return cg(goal_file, constraints, None, on, clean, sticky, fixed, drilled, welded, painted)


def start(config, input_args=None, INPUT_DATAPOINT=None):
    del input_args
    global datapoint, config_cache, metrics, constraints
    global sticky, fixed, on, fueled, cut, cleaner, stick, clean, drilled, welded, painted

    config_cache = config
    if _use_isaac_backend(config):
        isaac_backend = _get_isaac_backend()
        isaac_backend.start(config, INPUT_DATAPOINT=INPUT_DATAPOINT)
        _sync_runtime_from_datapoint(config, isaac_backend.getDatapoint(config))
        return

    sticky = []
    fixed = []
    on = []
    fueled = []
    cut = []
    cleaner = False
    stick = False
    clean = []
    drilled = []
    welded = []
    painted = []
    constraints = {"husky": []}

    if INPUT_DATAPOINT is not None:
        initial_snapshot = _first_valid_metric_snapshot(INPUT_DATAPOINT, config)
        if initial_snapshot is not None:
            config.initial_object_metrics = deepcopy(initial_snapshot)
        snapshot = _snapshot_from_datapoint(config, INPUT_DATAPOINT)
    else:
        if len(config.initial_object_metrics) == 0:
            config.initial_object_metrics = deepcopy(_default_layout(config))
        snapshot = {
            "metrics": _copy_metric_dict(config.initial_object_metrics, config),
            "constraints": {"husky": []},
            "sticky": [],
            "fixed": [],
            "cleaner": False,
            "on": [],
            "clean": [],
            "stick": False,
            "welded": [],
            "drilled": [],
            "painted": [],
            "fueled": [],
            "cut": [],
            "position": _position_from_metrics(config.initial_object_metrics),
            "world": config.graph_world_name,
            "goal": config.exploration_goal_name,
        }

    metrics = snapshot["metrics"]
    constraints = snapshot["constraints"]
    sticky = snapshot["sticky"]
    fixed = snapshot["fixed"]
    cleaner = snapshot["cleaner"]
    on = snapshot["on"]
    clean = snapshot["clean"]
    stick = snapshot["stick"]
    welded = snapshot["welded"]
    drilled = snapshot["drilled"]
    painted = snapshot["painted"]
    fueled = snapshot["fueled"]
    cut = snapshot["cut"]

    datapoint = Datapoint(config)
    datapoint.world = snapshot["world"]
    datapoint.goal = snapshot["goal"]
    datapoint.addPoint(
        deepcopy(snapshot["position"]),
        deepcopy(sticky),
        deepcopy(fixed),
        deepcopy(cleaner),
        "Initialize",
        deepcopy(constraints),
        deepcopy(metrics),
        deepcopy(on),
        deepcopy(clean),
        deepcopy(stick),
        deepcopy(welded),
        deepcopy(drilled),
        deepcopy(painted),
        deepcopy(fueled),
        deepcopy(cut),
    )


def initPolicy(config, domain, goal_json, world_num, SET_GAOL_JSON=False, INPUT_DATAPOINT=None):
    del domain, world_num
    global goal_spec
    goal_spec = goal_json if SET_GAOL_JSON else None
    start(config, INPUT_DATAPOINT=INPUT_DATAPOINT)


def last_index(x, data):
    return len(data) - data[::-1].index(x) - 1


def execAction(config, action, e, ACTION_SET=False, ONLY_RES=False):
    del e

    if ACTION_SET:
        if isinstance(action, dict) and "actions" in action:
            plan = action
        else:
            plan = {"actions": list(action)}
    else:
        plan = {"actions": [action]}

    try:
        if _use_isaac_backend(config):
            isaac_backend = _get_isaac_backend()
            isaac_backend.execute_collect_data(config, plan, goal_file=goal_spec, saveImg=False)
            _sync_runtime_from_datapoint(config, isaac_backend.getDatapoint(config))
            res = cg(goal_spec, constraints, None, on, clean, sticky, fixed, drilled, welded, painted)
            if ONLY_RES:
                return res, None, ""
            return res, getDGLGraph(config), ""

        res = execute(config, plan, goal_spec)
        if ONLY_RES:
            return res, None, ""
        data_index = last_index("End", datapoint.actions)
        graph_data = datapoint.getGraph(index=data_index, embeddings=config.embeddings)
        graph_data = graph_data["graph_" + str(data_index)]
        g = convertToDGLGraph(config, graph_data, False, -1)
        return res, g, ""
    except Exception as exc:
        return False, None, str(exc)


def get_datapoint():
    if config_cache is not None and _use_isaac_backend(config_cache):
        _sync_runtime_from_datapoint(config_cache, _get_isaac_backend().getDatapoint(config_cache))
    return deepcopy(datapoint)


def getDGLGraph(config):
    if _use_isaac_backend(config):
        _sync_runtime_from_datapoint(config, _get_isaac_backend().getDatapoint(config))
    data_index = last_index("End", datapoint.actions)
    graph_data = datapoint.getGraph(index=data_index, embeddings=config.embeddings)
    graph_data = graph_data["graph_" + str(data_index)]
    return convertToDGLGraph(config, graph_data, False, -1)


def getInitializeDGLGraph(config):
    if _use_isaac_backend(config):
        _sync_runtime_from_datapoint(config, _get_isaac_backend().getDatapoint(config))
    init_token = "Initialize" if "Initialize" in datapoint.actions else "Start"
    data_index = last_index(init_token, datapoint.actions)
    graph_data = datapoint.getGraph(index=data_index, embeddings=config.embeddings)
    graph_data = graph_data["graph_" + str(data_index)]
    return convertToDGLGraph(config, graph_data, False, -1)


def close_backend():
    if config_cache is not None and _use_isaac_backend(config_cache):
        _get_isaac_backend().destroy()


def cg_state(config, current_state, target_state, goal_attn, num_threshold=0.3, state_threshold=0.5):
    del goal_attn, num_threshold
    current_state = current_state.to(config.device)
    current_state = current_state.ndata["feat"]
    delta = torch.abs(target_state - current_state)
    return torch.all(delta < state_threshold)
