import argparse


def init_args():
    """Input arguments for the simulation."""
    parser = argparse.ArgumentParser('This will simulate a world describe in a json file.')

    parser.add_argument('--model_name', type=str, default='APN',
                        help='APN | AFE | AFE_MLP')
    parser.add_argument('--world', type=str, default='src/envs/jsons/home_worlds/world_home0.json',
                        help='The json file to visualize')
    parser.add_argument('--input', type=str, default='jsons/input_home.json',
                        help='The json file of input high level actions')
    parser.add_argument('--logging', action='store_true', default=False,
                        help='Video recording of simulation')
    parser.add_argument('--pretrained_ckpt', default=None, help='pretrained checkpoint')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--speed', type=float, required=False, default=1.0,
                        help='How quickly to step through the visualization')
    parser.add_argument('--goal', type=str, required=False,
                        default='src/envs/jsons/home_goals/goal1-milk-fridge.json',
                        help='Path of goal file')
    parser.add_argument('--randomize', action='store_true', required=False,
                        help='Turning this on randomizes the goal file and the scene file')
    parser.add_argument('--domain', type=str, default='home', help='home or factory')
    parser.add_argument('--embedding', type=str, default='conceptnet',
                        help='conceptnet or fasttext')
    parser.add_argument('--display', action='store_true', default=False,
                        help='Display states on matplotlib animation')
    parser.add_argument('--data_dir', type=str, default='data')

    # * Used in train_src.py
    parser.add_argument('--training', type=str, default='action', help='gcn | gcn_seq | action')
    parser.add_argument('--exec_type', type=str, default='train',
                        help='train | accuracy | ablation | generalization | policy')
    parser.add_argument('--split', type=str, default='world', help='random | world | tool')
    parser.add_argument('--global_node', action='store_true', default=False)
    parser.add_argument('--ignoreNoTool', action='store_true', default=False)
    parser.add_argument('--graph_seq_length', type=int, default=4)
    parser.add_argument('--goal_type', type=int, default=2, help='0 | 1 | 2')
    parser.add_argument('--state_noise', default=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--instruction', type=str, default='',
                        help='Natural language instruction for LLM-conditioned inference')
    parser.add_argument('--qwen_model', type=str, default='qwen3-max',
                        help='DashScope model name for scene graph translation')
    parser.add_argument('--qwen_api_key', default=None,
                        help='Optional DashScope API key override')
    parser.add_argument('--sample_index', type=int, default=-1,
                        help='Dataset sample index used to initialize inference state')
    parser.add_argument('--dataset_path', type=str, default='',
                        help='Dataset path used with sample_index for inference')
    parser.add_argument('--policy_backend', type=str, default='isaaclab',
                        help='Policy execution backend: isaaclab | symbolic')
    parser.add_argument('--candidate_action_num', type=int, default=20,
                        help='Candidate action pool size during planning')
    parser.add_argument('--select_from_candidate', type=int, default=10,
                        help='Pool refill threshold during planning')
    parser.add_argument('--max_planning_steps', type=int, default=60,
                        help='Maximum planning horizon during inference')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Maximum number of dataset samples to evaluate; -1 means all')
    parser.add_argument('--print_sample_info', action='store_true', default=False,
                        help='Print current sample index and world name during evaluation')

    # return parser.parse_args()
    opt, unknown = parser.parse_known_args()
    return opt
