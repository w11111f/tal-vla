from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = REPO_ROOT / "robot_ws" / "src" / "robot_navigation" / "launch" / "nav2_launch.py"


def _install_launch_stubs() -> None:
    launch_module = ModuleType("launch")
    launch_actions_module = ModuleType("launch.actions")
    launch_conditions_module = ModuleType("launch.conditions")
    launch_substitutions_module = ModuleType("launch.substitutions")
    launch_sources_module = ModuleType("launch.launch_description_sources")
    launch_ros_actions_module = ModuleType("launch_ros.actions")
    ament_pkg_module = ModuleType("ament_index_python.packages")

    class FakeLaunchDescription(list):
        pass

    class FakeDeclareLaunchArgument:
        def __init__(self, name, default_value=None, description=None):
            self.name = name
            self.default_value = default_value
            self.description = description

    class FakeGroupAction:
        def __init__(self, actions=None, condition=None):
            self.actions = list(actions or [])
            self.condition = condition

    class FakeTimerAction:
        def __init__(self, period=None, actions=None, condition=None):
            self.period = period
            self.actions = list(actions or [])
            self.condition = condition

    class FakeIncludeLaunchDescription:
        def __init__(self, launch_description_source, launch_arguments=None):
            self.launch_description_source = launch_description_source
            self.launch_arguments = dict(launch_arguments or [])

    class FakeLaunchConfiguration:
        def __init__(self, name, default=None):
            self.name = name
            self.default = default

        def __str__(self):
            return self.name

    class FakePythonExpression:
        def __init__(self, expression):
            self.expression = list(expression)

    class FakeIfCondition:
        def __init__(self, predicate):
            self.predicate = predicate

    class FakeUnlessCondition:
        def __init__(self, predicate):
            self.predicate = predicate

    class FakePythonLaunchDescriptionSource:
        def __init__(self, location):
            self.location = location

    class FakeNode:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_get_package_share_directory(_package_name):
        return str(REPO_ROOT / "robot_ws" / "src" / "robot_navigation")

    launch_module.LaunchDescription = FakeLaunchDescription
    launch_actions_module.DeclareLaunchArgument = FakeDeclareLaunchArgument
    launch_actions_module.GroupAction = FakeGroupAction
    launch_actions_module.TimerAction = FakeTimerAction
    launch_actions_module.IncludeLaunchDescription = FakeIncludeLaunchDescription
    launch_conditions_module.IfCondition = FakeIfCondition
    launch_conditions_module.UnlessCondition = FakeUnlessCondition
    launch_substitutions_module.LaunchConfiguration = FakeLaunchConfiguration
    launch_substitutions_module.PythonExpression = FakePythonExpression
    launch_sources_module.PythonLaunchDescriptionSource = FakePythonLaunchDescriptionSource
    launch_ros_actions_module.Node = FakeNode
    ament_pkg_module.get_package_share_directory = fake_get_package_share_directory

    sys.modules["launch"] = launch_module
    sys.modules["launch.actions"] = launch_actions_module
    sys.modules["launch.conditions"] = launch_conditions_module
    sys.modules["launch.substitutions"] = launch_substitutions_module
    sys.modules["launch.launch_description_sources"] = launch_sources_module
    sys.modules["launch_ros.actions"] = launch_ros_actions_module
    sys.modules["ament_index_python"] = ModuleType("ament_index_python")
    sys.modules["ament_index_python.packages"] = ament_pkg_module


def _load_module():
    _install_launch_stubs()
    module_name = "robot_navigation_nav2_launch_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_nav2_launch_exposes_isaac_mode_default():
    module = _load_module()

    description = module.generate_launch_description()
    mode_args = [item for item in description if getattr(item, "name", None) == "mode"]

    assert len(mode_args) == 1
    assert mode_args[0].default_value == "isaac"


def test_nav2_launch_defaults_to_expff_map():
    module = _load_module()

    description = module.generate_launch_description()
    map_args = [item for item in description if getattr(item, "name", None) == "map"]

    assert len(map_args) == 1
    assert str(map_args[0].default_value).endswith("/maps/expff_map.yaml")


def test_mode_condition_helper_targets_expected_mode():
    module = _load_module()
    mode = SimpleNamespace(name="mode")

    expression = module._mode_equals_expression(mode, "isaac")

    assert expression.expression == ["'", mode, "' == '", "isaac", "'"]


def test_nav2_launch_staggers_lifecycle_managers():
    module = _load_module()

    description = module.generate_launch_description()
    delayed_actions = [item for item in description if hasattr(item, "period")]

    periods = sorted(item.period for item in delayed_actions)

    assert module.LOCALIZATION_MANAGER_DELAY_SEC in periods
    assert module.NAVIGATION_STACK_DELAY_SEC in periods
    assert module.ISAAC_NAVIGATION_STACK_DELAY_SEC in periods
