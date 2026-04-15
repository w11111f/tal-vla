"""Legacy motion helpers removed during the Isaac Lab migration."""

import math


def euler_to_quaternion(rpy):
    roll, pitch, yaw = rpy
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def _legacy_error(function_name):
    raise RuntimeError(
        f"{function_name} is part of the removed PyBullet backend. "
        "Use src.envs.isaac_env for Isaac Lab execution."
    )


def move(*args, **kwargs):
    _legacy_error("move")


def moveTo(*args, **kwargs):
    _legacy_error("moveTo")


def constrain(*args, **kwargs):
    _legacy_error("constrain")


def removeConstraint(*args, **kwargs):
    _legacy_error("removeConstraint")


def changeState(*args, **kwargs):
    _legacy_error("changeState")
