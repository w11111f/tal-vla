"""Legacy UR5 PyBullet controller removed during the Isaac Lab migration."""


def _legacy_error(function_name):
    raise RuntimeError(
        f"{function_name} is part of the removed PyBullet UR5 backend. "
        "Use the Isaac Lab robot controller instead."
    )


def initGripper(*args, **kwargs):
    _legacy_error("initGripper")


def getUR5Controller(*args, **kwargs):
    _legacy_error("getUR5Controller")
