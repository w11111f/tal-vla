import json


SUPPORTED_ACTIONS = {
    "drop",
    "pick",
    "moveTo",
    "pushTo",
    "changeState",
    "pickNplaceAonB",
}


def convertActions(inp, world):
    action_list = []
    for high_level_action in inp["actions"]:
        if isinstance(high_level_action, list) and len(high_level_action) == 1:
            high_level_action = high_level_action[0]
        action_name = high_level_action["name"]
        args = high_level_action["args"]

        if action_name not in SUPPORTED_ACTIONS:
            raise ValueError(f"Unsupported action in reduced TAL action space: {action_name}")

        if action_name == "pickNplaceAonB":
            action_list.extend(
                [
                    ["moveTo", args[0]],
                    ["changeWing", "up"],
                    ["constrain", args[0], "ur5"],
                    ["moveTo", args[1]],
                    ["removeConstraint", args[0], "ur5"],
                    ["constrain", args[0], args[1]],
                ]
            )

        elif action_name == "pushTo":
            action_list.extend(
                [
                    ["moveTo", args[0]],
                    ["changeWing", "up"],
                    ["constrain", args[0], "ur5"],
                    ["moveToXY", args[1]],
                    ["removeConstraint", args[0], "ur5"],
                ]
            )

        elif action_name == "moveTo":
            action_list.append(["moveTo", args[0]])

        elif action_name == "pick":
            action_list.extend(
                [
                    ["moveTo", args[0]],
                    ["changeWing", "up"],
                    ["constrain", args[0], "ur5"],
                ]
            )

        elif action_name == "drop":
            action_list.append(["removeConstraint", args[0], "ur5"])

        elif action_name == "changeState":
            action_list.extend(
                [
                    ["moveTo", args[0]],
                    ["changeWing", "up"],
                    ["changeState", args[0], args[1]],
                ]
            )

        action_list.append(["saveBulletState"])

    return action_list


def convertActionsFromFile(action_file):
    with open(action_file, "r") as handle:
        inp = json.load(handle)
    return inp
