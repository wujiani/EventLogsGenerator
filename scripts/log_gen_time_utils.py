import pandas as pd
import numpy as np
import random

def fillna_parallel_data(bb, state_column):
    i = bb.index[0]
    j = 0
    while j < len(bb):
        if bb[state_column].loc[i] == 'complete':
            paired = int(bb['paired_event'].loc[i])
            bb['process_time'].loc[paired] = bb['process_time'].loc[i]
            bb['paired_event'].loc[paired] = i
            bb['next'].loc[paired] = bb['next'].loc[i]
            bb['waiting_time'].loc[i] = bb['waiting_time'].loc[paired]
            bb['preceding_evts'].loc[i] = bb['preceding_evts'].loc[paired]
            bb['next'].loc[paired] = bb['next'].loc[i]
        i+=1
        j+=1

def string_to_list(c):
    if isinstance(c, str):
        c = c.strip('[')
        c = c.strip(']')
        c = c.split(',')
        c = [int(each) for each in c]
    else:
        pass
    return c

def get_name(df, c, act_column):
    if isinstance(c, list):
        result =  tuple(sorted([df[act_column].loc[each] for each in c]))
    else:
        result = None
    return result

# function used to generate the next event for one prefix according to probabilities

def number_of_certain_probability(sequence, probability):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    i = 0
    for i, item_probability in enumerate(probability):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return sequence[i]

def get_next_event(last_cur, time_name, dict_time):
    next_event_prob = dict_time[time_name][last_cur]
    next_event = number_of_certain_probability(list(next_event_prob.keys()), list(next_event_prob.values()))
    return next_event

def get_time(last_cur, time_name, dict_time, dict_kde):
    cur_time = get_next_event(last_cur, time_name, dict_time)
    if cur_time == 'kde':
        cur_time = -1
        while cur_time < 0:
            cur_time = np.round(dict_kde[time_name][last_cur].sample(1))
            cur_time = cur_time[0][0]
    return cur_time

def get_emb(act_name, act_index, act_weights):
    temp = []
    if type(act_name) == tuple:
        index = [act_index[each] for each in act_name]
        for each in index:
            temp.append(act_weights[each])
        return pd.DataFrame(temp).sum(axis=0).to_list()
    else:
        return act_weights[act_index[act_name]]

def eucliDist(A,B):
    return np.sqrt(sum(np.power((A - B), 2)))


def get_k_nearest(last_cur, k, dict_cur_last, act_index, act_weights):
    temp = {}
    act_last = last_cur[0]
    act_cur = last_cur[1]
    cur_emb = get_emb(act_cur, act_index, act_weights)
    all_last = dict_cur_last[act_cur]
    for each_last in all_last:
        last_emb = get_emb(each_last, act_index, act_weights)
        temp[each_last] = eucliDist(np.array(cur_emb), np.array(last_emb))
    temp = sorted(temp.items(), key=lambda x: x[1], reverse = False)
    return dict(temp[:k])

def get_time_no_key_exist(last_cur, k, time_name, dict_cur_last, dict_time, dict_kde, act_index, act_weights):
    k_last = get_k_nearest(last_cur, k, dict_cur_last, act_index, act_weights).keys()
    k_last_value = [get_time((each_last, last_cur[1]), time_name, dict_time, dict_kde) for each_last in k_last]
    return np.mean(k_last_value)

# add start event in the beginning and end in the last
def add_start_end(group):
    start_series = group.iloc[0]
    start_series['act'] = 'Start'
    start_series['index'] = start_series['index']-1
#     start_series[resource_column] = 'addstart'
#     start_series[state_column] = 'start'
#     start_series[end_time_column] = start_series[start_time_column]
#     start_series_ = copy.deepcopy(start_series)
#     start_series_[state_column] = 'complete'
    start_serieses = pd.DataFrame([start_series], columns = group.columns)
    group = pd.concat([start_serieses, group], axis=0 ,ignore_index=True)

#     end_series = group.iloc[len(group)-1]
#     end_series['act'] = 'End'
#     end_series['index'] = end_series['index']+1
# #     end_series[resource_column] = 'addend'
# #     end_series[state_column] = 'start'
# #     end_series[start_time_column] = end_series[end_time_column]
# #     end_series_ = copy.deepcopy(end_series)
# #     end_series_[state_column] = 'complete'
#     end_serieses = pd.DataFrame([end_series], columns = group.columns)
#     group = pd.concat([group, end_serieses], axis=0 ,ignore_index=True)
    return group
