"""Legacy PyBullet world initialisation removed during the Isaac Lab migration."""

import json


def _legacy_error(function_name):
    raise RuntimeError(
        f"{function_name} depends on the removed PyBullet backend. "
        "Load scenes through src.envs.isaac_env instead."
    )


def loadObject(*args, **kwargs):
    _legacy_error("loadObject")


def loadWorld(*args, **kwargs):
    _legacy_error("loadWorld")


def initWingPos(wing_file='src/envs/jsons/wings.json'):
    wings = {}
    control_joints = [
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint',
        'robotiq_85_left_knuckle_joint',
    ]
    with open(wing_file, 'r') as handle:
        poses = json.load(handle)['poses']
    for pose in poses:
        wings[pose['name']] = dict(zip(control_joints, pose['pose']))
    return wings


def initHuskyUR5(*args, **kwargs):
    _legacy_error("initHuskyUR5")
