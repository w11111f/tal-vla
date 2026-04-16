import pickle

if __name__ == '__main__':
    dataset_file_path = './data/home/world_home0/0.graph'
    with open(dataset_file_path, 'rb') as f:
        dataset = pickle.load(f)

    edges = dataset.edges.data()

    print('edges num: {}'.format(len(edges)))  # * 2994
    # * (0, 1, {'weight': 1.0, 'action': {'actions': [{'name': 'pushTo', 'args': ['orange', 'cube_green']}]}})
    print(list(edges)[0])

    # * 11
    action_dict = {
        'drop': 0, 'climbDown': 0, 'pick': 0, 'moveTo': 0, 'climbUp': 0,
        'pushTo': 0, 'changeState': 0, 'pickNplaceAonB': 0, 'clean': 0, 'apply': 0,
        'stick': 0
    }

    # * 36
    object_dict = {
        'floor': 0, 'walls': 0, 'door': 0, 'fridge': 0, 'cupboard': 0, 'husky': 0, 'table': 0,
        'table2': 0, 'couch': 0, 'big-tray': 0,
        'book': 0, 'paper': 0, 'cube_gray': 0, 'cube_green': 0, 'cube_red': 0, 'tray': 0,
        'tray2': 0, 'bottle_blue': 0, 'chair': 0, 'stick': 0,
        'bottle_gray': 0, 'bottle_red': 0, 'box': 0, 'apple': 0, 'orange': 0, 'dumpster': 0,
        'light': 0, 'milk': 0, 'shelf': 0, 'glue': 0,
        'tape': 0, 'stool': 0, 'mop': 0, 'sponge': 0, 'vacuum': 0, 'dirt': 0
    }

    for edge in list(edges):
        action_item = edge[2]['action']['actions'][0]
        action_name = action_item['name']
        action_dict[action_name] += 1
        object_1 = action_item['args'][0]
        object_dict[object_1] += 1
        if action_name != 'changeState' and len(action_item['args']) > 1:
            object_2 = action_item['args'][1]
            object_dict[object_2] += 1

    action_dict = sorted(action_dict.items(), key=lambda action_dict: action_dict[1], reverse=True)
    object_dict = sorted(object_dict.items(), key=lambda object_dict: object_dict[1], reverse=True)

    print('--' * 20)
    print(action_dict)
    print('--' * 20)
    print(object_dict)

    # ----------------------------------------
    # [('pushTo', 1055), ('pickNplaceAonB', 543), ('moveTo', 542), ('drop', 374), ('pick', 361),
    # ('climbUp', 50), ('climbDown', 39), ('clean', 18), ('changeState', 12), ('apply', 0),
    # ('stick', 0)]
    # ----------------------------------------
    # [('stool', 283), ('tray', 231), ('paper', 211), ('cube_green', 208), ('chair', 195),
    # ('bottle_red', 194), ('box', 193), ('orange', 190), ('tray2', 189), ('cube_red', 188),
    # ('bottle_blue', 186), ('cube_gray', 182), ('bottle_gray', 180), ('tape', 168),
    # ('stick', 167), ('book', 164), ('sponge', 164), ('apple', 157), ('vacuum', 143), ('glue', 141),
    # ('mop', 125), ('shelf', 106), ('table', 105), ('big-tray', 100), ('couch', 93), ('table2', 91),
    # ('fridge', 83), ('dirt', 72), ('cupboard', 60), ('walls', 14), ('door', 4), ('milk', 4),
    # ('light', 1), ('floor', 0), ('husky', 0), ('dumpster', 0)]
