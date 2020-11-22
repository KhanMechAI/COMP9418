'''
COMP9418 Assignment 2
This file is the example code to show how the assignment will be tested.

Name:     zID: 5020362
'''

# Make division default to floating-point, saving confusion
from __future__ import division
from __future__ import print_function

import copy
import datetime
from collections import OrderedDict as odict
from itertools import product
from pathlib import Path

# Allowed libraries - Removed redundant
import numpy as np
import pandas as pd



# Assignment 1 Code re-used

def learn_outcome_space(data: pd.DataFrame) -> dict:
    return {var: tuple(data[var].unique()) for var in data.columns.values}


# From Tutorial 2
def transposeGraph(G):
    GT = dict((v, []) for v in G)
    for v in G:
        for w in G[v]:
            GT[w].append(v)

    return GT


# From tutorial 3

def allEqualThisIndex(dict_of_arrays, **fixed_vars):
    """
    Helper function to create a boolean index vector into a tabular data structure,
    such that we return True only for rows of the table where, e.g.
    column_a=fixed_vars['column_a'] and column_b=fixed_vars['column_b'].

    This is a simple task, but it's not *quite* obvious
    for various obscure technical reasons.

    It is perhaps best explained by an example.

    >>> all_equal_this_index(
    ...    {'X': [1, 1, 0], Y: [1, 0, 1]},
    ...    X=1,
    ...    Y=1
    ... )
    [True, False, False]
    """
    # base index is a boolean vector, everywhere true
    first_array = dict_of_arrays[list(dict_of_arrays.keys())[0]]
    index = np.ones_like(first_array, dtype=np.bool_)
    for var_name, var_val in fixed_vars.items():
        index = index & (np.asarray(dict_of_arrays[var_name]) == var_val)
    return index


def estProbTable(data, var_name, parent_names, outcomeSpace, add_smooth=False, alpha=1):
    """
    Calculate a dictionary probability table by ML given
    `data`, a dictionary or dataframe of observations
    `var_name`, the column of the data to be used for the conditioned variable and
    `parent_names`, a tuple of columns to be used for the parents and
    `outcomeSpace`, a dict that maps variable names to a tuple of possible outcomes
    Return a dictionary containing an estimated conditional probability table.
    """
    var_outcomes = outcomeSpace[var_name]
    parent_outcomes = [outcomeSpace[var] for var in (parent_names)]
    # cartesian product to generate a table of all possible outcomes
    all_parent_combinations = product(*parent_outcomes)

    prob_table = odict()

    num_combs = 0
    if not add_smooth:
        alpha = 0
    else:
        num_combs = np.prod([len(x) for x in parent_outcomes]) * len(var_outcomes)
        all_parent_combinations = product(*parent_outcomes)

    for i, parent_combination in enumerate(all_parent_combinations):
        parent_vars = dict(zip(parent_names, parent_combination))
        parent_index = allEqualThisIndex(data, **parent_vars)
        counts = (parent_index.sum() + alpha * num_combs)

        for var_outcome in var_outcomes:
            if not counts:
                prob_table[tuple(list(parent_combination) + [var_outcome])] = 0
                continue
            var_index = (np.asarray(data[var_name]) == var_outcome)
            prob_table[tuple(list(parent_combination) + [var_outcome])] = \
                ((var_index & parent_index).sum() + alpha) / counts

    return {'dom': tuple(list(parent_names) + [var_name]), 'table': prob_table}


## Develop your code for learn_bayes_net(G, data, outcomeSpace) in one or more cells here

def learn_bayes_net(G, data, outcomeSpace, add_smooth=False, alpha=1) -> dict:
    cond_tables_ml = odict()
    G_T = transposeGraph(G)
    for node, parents in G_T.items():
        cond_tables_ml[node] = estProbTable(data, node, parents, outcomeSpace, add_smooth=add_smooth, alpha=alpha)
    return cond_tables_ml


def get_robot_data(robot: str, train_df):
    return pd.DataFrame([tuple(str(k.strip("\(\) ''")) for k in x.split(',')) for x in train_df[robot].tolist()],
                        index=train_df.index)


def prob_to_df(prob_table, single=False) -> dict:
    p_t = copy.deepcopy(prob_table)
    col_name = "P"
    if single:
        return pd.DataFrame.from_dict(p_t['table'], orient='index', columns=[col_name])
    else:
        for node in p_t.keys():
            p_t[node]['table'] = pd.DataFrame.from_dict(p_t[node]['table'], orient='index', columns=[col_name])
    return p_t

def markov_blanket(G, node):
    children = G[node]

    GT = transposeGraph(G)
    parents = GT[node]

    spouse = []
    for child in children:
        spouse.extend(GT[child])

    blanket_nodes = [*children, *parents, *spouse]
    return list(set(blanket_nodes))


def assess_bayes_net(G, prob_tables, data, outcomeSpace, class_var, smoothing=True) -> float:
    G_T = transposeGraph(G)
    children = G[class_var]

    pred = np.ones([data.shape[0], len(outcomeSpace[class_var])])

    index_map = {i: outcome for i, outcome in enumerate(outcomeSpace[class_var])}

    prob_t = prob_to_df(prob_tables)
    for i, outcome in index_map.items():
        query_params = [*G_T[class_var], class_var]
        test_data = data[query_params].copy(deep=True)
        test_data[class_var] = outcome
        probs = prob_t[class_var]['table'].loc[test_data.to_records(index=False)].values.flatten()
        if smoothing:
            pred[:, i] = np.log(probs)
        else:
            pred[:, i] = probs
        for child in children:
            query_params = [*G_T[child], child]
            test_data = data[query_params].copy(deep=True)
            test_data[class_var] = outcome
            probs = prob_t[child]['table'].loc[test_data.to_records(index=False)].values.flatten()
            if smoothing:
                pred[:, i] += np.log(probs)
            else:
                pred[:, i] = np.multiply(pred[:, i], probs)

    predits = np.argmax(pred, axis=1)

    df_map = pd.DataFrame.from_dict(index_map, orient='index', columns=[class_var])
    df_map.reset_index(inplace=True)
    df_map.set_index([class_var], inplace=True)
    acc = np.sum(
        pd.merge(df_map, data, how='right', right_on=class_var, left_index=True, sort=False)['index'] == predits) / \
          data.shape[0]

    return acc


def k_fold_split(num_samples, k=10):
    # Yeilds the test index to be excluded
    indices = np.arange(num_samples, dtype=int)
    np.random.shuffle(indices)
    indices = np.array_split(indices, k)
    while indices:
        yield indices.pop()


def cv_bayes_net(G, data, class_var, k=10, add_smooth=True, alpha=1):
    k_fold_idx = k_fold_split(data.shape[0], k)
    accuracy = np.zeros(k)
    for i in range(k):
        test_idx = next(k_fold_idx)
        train_idx = data.index.difference(test_idx)

        test_set = data.loc[test_idx]
        train_set = data.loc[train_idx]

        outcomeSpace = learn_outcome_space(train_set)
        prob_tables = learn_bayes_net(G, train_set, outcomeSpace, add_smooth=add_smooth, alpha=alpha)

        accuracy[i] = assess_bayes_net(G, prob_tables, test_set, outcomeSpace, class_var, smoothing=add_smooth)

    return accuracy.mean(), accuracy.std()


def load_data(path) -> pd.DataFrame:
    return pd.read_csv(training_data_path, index_col=0)


def binarise(df: pd.DataFrame, drop: list=[]) -> pd.DataFrame:
    '''

    :param df: Input training data
    :type df: dataframe
    :param drop: List of columns to exclude from training data
    :type drop: list
    :return: binarised dataframe
    :rtype: Pandas Dataframe
    '''
    binary_train = df.copy()
    binary_train.drop(columns=drop, inplace=True)
    for col in binary_train.columns:
        if binary_train[col].dtype == "int64":
            binary_train[col] = binary_train[col] > 0

    return binary_train


def binarise_dict(sensor_data, ):
    '''

    :param sensor_data: incoming sensor data
    :type sensor_data: dict
    :return: binarised dict
    :rtype: dict
    '''
    binarised_data = sensor_data
    for sensor, reading in binarised_data.items():
        if str(reading).isdigit():
            binarised_data[sensor] = reading > 0

    return binarised_data

# Setup parameters
actions_dict = {'lights1': 'off', 'lights2': 'on', 'lights3': 'off', 'lights4': 'off', 'lights5': 'off',
                'lights6': 'off', 'lights7': 'off', 'lights8': 'off', 'lights9': 'off', 'lights10': 'off',
                'lights11': 'off', 'lights12': 'off', 'lights13': 'off', 'lights14': 'off', 'lights15': 'off',
                'lights16': 'off', 'lights17': 'off', 'lights18': 'off', 'lights19': 'off', 'lights20': 'off',
                'lights21': 'off', 'lights22': 'off', 'lights23': 'off', 'lights24': 'off', 'lights25': 'off',
                'lights26': 'off', 'lights27': 'off', 'lights28': 'off', 'lights29': 'off', 'lights30': 'off',
                'lights31': 'off', 'lights32': 'off', 'lights33': 'off', 'lights34': 'off', 'lights35': 'on'}

rooms_lights_mapping = {x.replace("lights", "r"): x for x in actions_dict.keys()}  # Used to map rooms to lights
lights_rooms_mapping = {x: x.replace("lights", "r") for x in actions_dict.keys()}  # Used to map rooms to lights

rooms = list(rooms_lights_mapping.keys())  # Used to iterate
lights = list(rooms_lights_mapping.values())  # Used to set lights on initially
robots = ["robot1", "robot2"]  # Used to iterate

# Network structure
networks = {
    "unreliable_sensor3": ["r1"],
    "unreliable_sensor2": ["r6"],
    "unreliable_sensor1": ["r29"],
    "r1": [],
    "r29": [],
    "unreliable_sensor4": ["r24", "r23", "r13", ],
    "r24": [],
    "r23": [],
    "r13": [],
    "reliable_sensor1": ["r16"],
    "r16": [],
    "reliable_sensor2": ["r5", "r6"],
    "r5": [],
    "r6": [],
    "reliable_sensor3": ["r25", ],
    "reliable_sensor4": ["r31", "r32"],
    "r25": [],
    "r31": [],
    "r32": [],
    "door_sensor1": ["r8", "r9", "r13"],
    "r8": [],
    "r9": [],
    "door_sensor3": ["r26", "r27", "r32"],
    "r26": [],
    "r27": [],
    "door_sensor4": ["r35", "r29"],
    "door_sensor2": ["r29"],
    "r35": []
}

room_to_network_map = transposeGraph(networks)

all_rooms = [f"r{x + 1}" for x in range(35)]

# Load training data
training_data_path = Path(__file__).parent / "data.csv"
train_df = load_data(training_data_path)

# Binarise the training data
cols_to_drop = ["robot1", "robot2", "time", "electricity_price"]
binary_train = binarise(train_df, drop=cols_to_drop)

# Learn outcome space
outcome_space = learn_outcome_space(binary_train)

# Learn the bayes net
smoothing = True
prob_tables = learn_bayes_net(networks, binary_train, outcome_space, add_smooth=smoothing, alpha=1)







# Initially set all lights on just to be safe
actions = {k: "on" for k in lights}

rs3_switch = True

toggle_light = {
    0: "off",
    1: "on",
}

def get_action(sensor_data):
    # declare state as a global variable so it can be read and modified within this function
    global state
    global rs3_switch

    # First, assumed everything is off. Then toggle lights off.
    actions = {k: "on" for k in lights}

    # At the end of the day, we can turn off all the lights
    if sensor_data["time"] == datetime.time(18, 0, 0):
        rs3_switch = True

    # If no one has entered the building, and there remains no motion, then lets keep the lights off
    if (sensor_data["reliable_sensor3"] == "no motion") and rs3_switch:
        rs3_switch = False
        actions = {k: "off" for k in lights}
        actions["lights12"] = "on"  # leave the entrance on
        actions["lights22"] = "on"  # leave the entrance on

    # The BN needs data in the same outcome space as what it was trained on.
    binarised_sensor_data = binarise_dict(sensor_data, )

    # Need to do checks for bung sensors
    dud_sensors = [s for s in sensor_data if sensor_data[s] is None and s in networks]

    # Disabling inference on rooms that have dud sensors
    disabled_rooms = [x for dud in dud_sensors for x in networks[dud]]

    for room in all_rooms:
        # If the room isnt in the network, or is disabled, go to next
        if (room not in room_to_network_map) or (room in disabled_rooms):
            continue

        #Check if any duds in the room's sub network and adjust
        query_duds = [x for x in room_to_network_map[room] if x in dud_sensors]
        if ~any(query_duds):
            query = [binarised_sensor_data[x] for x in room_to_network_map[room]]
        else:
            query = [binarised_sensor_data[x] for x in room_to_network_map[room] if x not in dud_sensors]

        # Perform MPE
        combs = [(*query, y) for y in [True, False]]
        candidate = np.argmax([prob_tables[room]["table"][x] for x in combs])

        actions[rooms_lights_mapping[room]] = toggle_light[combs[candidate][-1]]

    # Robots are assumed to be 100% accurate when not failed. Hence, any room without people counts needs to be
    # toggled off.
    for robot in robots:
        room_state = sensor_data[robot]
        if room_state:
            room, count = tuple(str(k.strip("\(\) '")) for k in room_state.split(','))
            if room in rooms and count == 0:
                actions[rooms_lights_mapping[room]] = "off"

    return actions
