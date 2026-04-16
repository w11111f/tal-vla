"""
Project refactoring.
Used to handle pickle.load() errors after file path changes.
"""
import sys

from src import tal
from src.envs import CONSTANTS, datapoint

sys.modules['src.GNN'] = tal
sys.modules['src.GNN.CONSTANTS'] = CONSTANTS
sys.modules['src.datapoint'] = datapoint
