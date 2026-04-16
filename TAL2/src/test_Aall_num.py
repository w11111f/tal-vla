import pickle
from src.config.config import init_args
from src.envs.CONSTANTS import EnvironmentConfig

if __name__ == '__main__':
    args = init_args()
    config = EnvironmentConfig(args)
    # config.display = True

    with open(config.Aall_path, 'rb') as f:
        A_all = pickle.load(f)

    print('--' * 20)
    for data in A_all:
        print(data)
    print('--' * 20)
