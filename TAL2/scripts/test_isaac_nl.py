# 新建 inference_isaac_nl.py
import torch
from src.config.config import init_args
from src.envs.CONSTANTS import EnvironmentConfig
from src.tal.utils_training import get_model, load_model
import pickle

# 引入翻译与规划模块
from src.tal.scene_graph_translator import translate_instruction_to_goal_state_graph
from src.tal.utils_planning import get_action_pred_with_model_extract_feature, get_action_pred_with_model_actions, process_feature_with_pca

def generate_plan_from_nl(instruction, current_scene_graph_json):
    args = init_args()
    config = EnvironmentConfig(args)
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载训练好的网络模型
    model_afe = get_model(config, 'AFE', config.features_dim, config.num_objects)
    model_afe, _, _, _ = load_model(config, 'AFE_Trained', model_afe)
    model_afe.eval().to(config.device)
    
    model_apN = get_model(config, 'APN', config.features_dim, config.num_objects)
    model_apN, _, _, _ = load_model(config, 'APN_Trained', model_apN)
    model_apN.eval().to(config.device)
    
    # 2. 加载之前提取好的动作特征库
    with open('checkpoints/home/action_effect_features_avg.pkl', 'rb') as f:
        action_effect_features = pickle.load(f)
        
    # 3. 调用大语言模型，把你的自然语言和当前 Isaac 场景图 变成目标图 DGL 张量
    # 注意：需要你配置 DASHSCOPE_API_KEY (通义千问) 环境变量
    print(f"Translating instruction: '{instruction}'")
    _, goal_json, goal_state_graph = translate_instruction_to_goal_state_graph(
        config, 
        instruction=instruction,
        current_scene_graph_json=current_scene_graph_json
    )
    goal_state_graph = goal_state_graph.to(config.device)
    
    # 这里为了演示，我们假设将 current_scene_graph_json 转为 DGL 张量
    # 在真实流程中，你需要提供一个 DGL current_state_graph
    from src.tal.scene_graph_translator import scene_graph_json_to_dgl
    current_state_graph = scene_graph_json_to_dgl(config, current_scene_graph_json).to(config.device)
    
    # 4. 使用 CAG (PCA特征分解) 生成候选动作集
    action_features_tensor = torch.stack(action_effect_features['features']).squeeze(1)
    action_features_tensor, principal_directions = process_feature_with_pca(action_features_tensor, q_value=500)
    
    with torch.no_grad():
        _, current_task_feature = model_afe(current_state_graph, goal_state_graph)
        current_task_feature, _ = process_feature_with_pca(current_task_feature, principal_directions)
        
    act_generalized_inverse_mat = torch.linalg.pinv(action_features_tensor)
    output_cag = torch.matmul(current_task_feature, act_generalized_inverse_mat).squeeze(0)
    
    candidate_action_num = 15
    topk_actions = torch.topk(output_cag, candidate_action_num)
    
    selected_actions =[]
    for idx in topk_actions.indices:
        selected_actions.append(action_effect_features['names'][idx])
        
    # 5. 使用 APN 进行动作打分，挑选当前最该执行的一步
    actions_prob = get_action_pred_with_model_actions(
        config, model_apN, action_effect_features, current_state_graph, goal_state_graph
    )
    
    selected_actions_prob = []
    for act in selected_actions:
        idx = action_effect_features['names'].index(act)
        selected_actions_prob.append(actions_prob[0][idx].item())
        
    # 选出最高分动作
    max_prob = max(selected_actions_prob)
    best_action_dict = selected_actions[selected_actions_prob.index(max_prob)]
    
    print("\n>>> 模型预测输出的 Python 结构化字典指令 <<<")
    print(best_action_dict)
    # 示例输出: {'name': 'pickNplaceAonB', 'args': ['apple', 'table']}
    
    return best_action_dict

if __name__ == '__main__':
    # 假设这是你从 Isaac Lab 中提取出来的当前场景图格式
    mock_isaac_scene_graph = {
        "nodes":[
            {"id": 23, "name": "apple", "properties": ["Movable"], "states": ["Clean", "Free"], "position": [[1.0, 2.0, 0.5], [0,0,0,1]], "size":[0.1, 0.1, 0.1]},
            {"id": 6, "name": "table", "properties": ["Surface"], "states": [], "position": [[1.0, 2.0, 0.0], [0,0,0,1]], "size":[1.0, 1.0, 0.5]},
            {"id": 5, "name": "husky", "properties":[], "states": [], "position": [[0.0, 0.0, 0.0],[0,0,0,1]], "size": [1.0, 1.0, 1.0]}
        ],
        "edges":[]
    }
    
    nl_instruction = "把苹果放在桌子上"
    generate_plan_from_nl(nl_instruction, mock_isaac_scene_graph)