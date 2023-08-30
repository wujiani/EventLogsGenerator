import pandas as pd
import copy
import random

# add start event in the beginning and end in the last
def add_start_end(group, act_column, resource_column, state_column):
    start_series = group.iloc[0]
    start_series[act_column] = 'addstart'
    start_series[resource_column] = 'addstart'
    start_series[state_column] = 'start'
    start_series_ = copy.deepcopy(start_series)
    start_series_[state_column] = 'complete'
    start_serieses = pd.DataFrame([start_series, start_series_])
    group = pd.concat([start_serieses, group], axis=0 ,ignore_index=True)

    end_series = group.iloc[len(group)-1]
    end_series[act_column] = 'addend'
    end_series[resource_column] = 'addend'
    end_series[state_column] = 'start'
    end_series_ = copy.deepcopy(end_series)
    end_series_[state_column] = 'complete'
    end_serieses = pd.DataFrame([end_series, end_series_])
    group = pd.concat([group, end_serieses], axis=0 ,ignore_index=True)
    return group

# to record the order and the info of the parallel
# input: group: trace with parallel
#        i: first index of event of the parallel
#        sub_groups: dictionary that consists the order and info of all parallel situations
# output: the name of the parallel and the index of event that first after the parallel
def deal_sub_group(group, i, sub_groups, act_column, resource_column, state_column):
    start_from_act = group.iloc[i - 1]
    not_complete = 0  # a flag used to check whether the parallel ends
    sub_group = []  # list to contain the order and the info of the parallel
    cur_act = group.iloc[i]
    sub_group.append(cur_act)
    not_complete += 1
    j = i + 1
    dict_name = ()
    while j < len(group):
        cur_act = group.iloc[j]
        next_act = group.iloc[j + 1]
        if (not_complete == 0) \
                and (cur_act[act_column] == next_act[act_column]) \
                and (cur_act[resource_column] == next_act[resource_column]) \
                and (cur_act[state_column] == 'start') \
                and (next_act[state_column] == 'complete'):  # if the parallel ends
            dict_name = (start_from_act[act_column], cur_act[act_column])  # tuple, the name of the parallel
            #             if dict_name == ("Validar solicitud", "Validacion final"):
            # #                 print(group)
            temp = sub_groups.get(dict_name, [])
            temp.append(sub_group)
            sub_groups[
                dict_name] = temp  # store the (list to contain the order and the info of the parallel) into the dictionary that has (tuple, the name of the parallel) as keys
            break
        else:
            sub_group.append(cur_act)  # add the order and the info of the parallel
            j += 1  # the next event
            if cur_act[state_column] == 'complete':
                not_complete -= 1  # there is one event ends in the parallel
            else:
                not_complete += 1  # there is one events happens in the parallel
    return dict_name, j

# split the sequence into the sequence delete the last one(seq1) and the last one(evt2), used to compute the probabilities of the next event
# also for the same seq1, collect all the possible evt2's in the dictionary 'transition'
# input: list of sequences(prefix/ n-gram)
# output: dictionary(keys:seq1, values:list of all the possible next events(multi-sets) in all traces
def get_transition(input_trans):
    transition = {}
    for each in input_trans:
        prefix = each[:-1]
        prefix = tuple(prefix)
        tmp = transition.get(prefix, [])
        tmp.append(each[-1])
        transition[prefix] = tmp
    return transition

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

# n-gram-list
def create_ngram_list(input_list, ngram_num=2):
#     input_list = ['addstart']+input_list+['addend']
    ngram_list = []
    if len(input_list) <= ngram_num:
        ngram_list.append(input_list)
    else:
        for tmp in zip(*[input_list[i:] for i in range(ngram_num)]):
            ngram_list.append(list(tmp))
        if ngram_num > 2:
            temp = input_list[:ngram_num-1]
            for i in range(2, ngram_num):
                ngram_list.append(temp[:i])
            temp = input_list[-ngram_num+1:]
            for i in range(2, ngram_num):
                ngram_list.append(temp[-i:])
    return ngram_list

# # get all prefixes, used in the parallel
# def all_prefix(input_list):
#     input_list = ['addstart']+list(input_list)+['addend']
#     all_pref = []
#     for i in range(2, len(input_list)+1):
#         all_pref.append(input_list[:i])
#     return all_pref

# get all prefixes
def all_prefix_(input_list):
    all_pref = []
    for i in range(2, len(input_list)+1):
        all_pref.append(input_list[:i])
    return all_pref

def all_prefix(input_list):
    input_list = ['addstart']+list(input_list)+['addend']
    all_pref = []
    for i in range(2, len(input_list)+1):
        all_pref.append(input_list[:i])
    return all_pref
# get all prefixes or n-gram
# input: all_traces: list that contains all the sequences of traces
#        method: 'n_gram'/'prefix'
#        num: if used 'n_gram' method, set n
#        if_sub_group: check if parallel
# output: sub_traces: list that consists of all prefixes or n-grams
def get_sub_traces(all_traces, method='n_gram', num=2, if_sub_group=False):
    sub_traces = []
    if method == 'prefix':
        for each_trace in all_traces:
            if if_sub_group:
                sub_traces.append(all_prefix(each_trace))
            else:
                sub_traces.append(all_prefix_(each_trace))
        sub_traces = [each_1 for each in sub_traces for each_1 in each]
    elif method == 'n_gram':
        for each_trace in all_traces:
            sub_traces.append(create_ngram_list(each_trace, ngram_num=num))
        sub_traces = [each_1 for each in sub_traces for each_1 in each]
    return sub_traces

# # function used to generate next event in the parallel
# # input: seq1, the name of the parallel
# # output: the next event after seq1
# def get_next_event_tuple(cur_event, input_tuple):
#     next_event_prob = sta_sub[input_tuple][tuple(cur_event)]
#     next_event = number_of_certain_probability(list(next_event_prob.keys()), list(next_event_prob.values()))
#     return next_event

# function used to generate next event in the trace sequence
# input: seq1
# output: the next event after seq1
def get_next_event(cur_event, sequence_prob):
    next_event_prob = sequence_prob[tuple(cur_event)]
    next_event = number_of_certain_probability(list(next_event_prob.keys()), list(next_event_prob.values()))
    return next_event


# generate the sequences of traces
# input: number_traces: the number of generated sequences
#        method: 'n_gram'/'prefix', the method used to generate sequence
#        if_start_end: if data contains 'start' and 'end' in the beginning and end of the trace
# output: dataframe, with caseid and event name of the traces(contains names of parallel)
def gen_new_data(sequence_prob, number_traces = 10, method = 'n_gram', if_start_end=False, num=2):
    gen_data = pd.DataFrame()
    for i in range(number_traces):
        cur_event='addstart'
        gen_sequence = [cur_event]
        cur_key = gen_sequence
        while cur_event != 'addend':
            next_event = get_next_event(cur_key, sequence_prob)
            gen_sequence.append(next_event)
            cur_event = next_event
            if method == 'n_gram':
                if len(gen_sequence) >= num:
                    cur_key = gen_sequence[-num+1:]
        if if_start_end:
            gen_sequence = gen_sequence[1:-1]
        dd = pd.DataFrame(gen_sequence, columns=['task'])
        dd['caseid'] = i
#         dd = dd.iloc[1:-1]
        gen_data = pd.concat([gen_data, dd], ignore_index = True)

    gen_data = gen_data[['caseid', 'task']]
    return gen_data


# # generate the sequences of parallel
# # input: number_traces: the number of generated parallel
# #        input_tuple: the name of parallel
# #        case_id: the id of case that contains parallel
# # output: dataframe, with caseid and event name in the parallel
# def gen_new_data_tuple(number_traces, input_tuple, case_id):
#     gen_data = pd.DataFrame()
#     for i in range(number_traces):
#         cur_event='addstart'
#         gen_sequence = [cur_event]
#         cur_key = gen_sequence
#         while cur_event != 'addend':
#             next_event = get_next_event_tuple(cur_key, input_tuple)
#             gen_sequence.append(next_event)
#             cur_event = next_event
#         dd = pd.DataFrame(gen_sequence, columns=['task'])
#         dd['caseid'] = case_id
#         dd = dd.iloc[1:-1]
#         gen_data = pd.concat([gen_data, dd], ignore_index = True)

#     gen_data = gen_data[['caseid', 'task']]
#     return gen_data

# fill the None in the dataframe
def fillna_parallel_data(df, state_column):
    i = df.index[0]
    j = 0
    while j < len(df):
        if df[state_column].loc[i] == 'complete':
            paired = int(df['paired_event'].loc[i])
            df['process_time'].loc[paired] = df['process_time'].loc[i]
            df['paired_event'].loc[paired] = i
            df['next'].loc[paired] = df['next'].loc[i]
            df['waiting_time'].loc[i] = df['waiting_time'].loc[paired]
            df['preceding_evts'].loc[i] = df['preceding_evts'].loc[paired]
            df['next'].loc[paired] = df['next'].loc[i]
        i+=1
        j+=1

# record the sequence in the parallel
def get_seq_dict(if_first, tuple_name, df, end, temp_dict, act_column):
    # record the sequence in the parallel
    def find_sequence(df, end, i, sub_tuple):
        next_acts = df['next'].loc[i]

        if len(next_acts) == 1:  # if there is no sub_parallel
            for each in next_acts:
                if each == end:
                    return [df[act_column].loc[i]]
                else:
                    return [df[act_column].loc[i]] + find_sequence(df, end, each,
                                                                   sub_tuple)  # record the sequence until the end of parallel
        else:  # if there is sub_parallel
            sub_parallel_name = tuple(
                sorted([df[act_column].loc[each] for each in next_acts], reverse=False))  # get the name of sub_parallel
            temp = sub_tuple.get(sub_parallel_name, [])
            temp.append(next_acts)
            sub_tuple[
                sub_parallel_name] = temp  # record the positions of sub_parallel events under the name of sub_parallel
            return [df[act_column].loc[i]] + [
                sub_parallel_name]  # record the sequence also with the name of sub_parallel

    if if_first == False:   # if the event isn't the first events in the parallel (happens simultaneously)
        if len(tuple_name) == 0:   # if there is no sub-parallel in the parallism
            return
        for each in tuple_name:   # tuple_name is dictionary: key: sub_parallel name, value: list contains lists of positions that events happens together in the sub_parallel
            sub_tuple = {}
            temp = temp_dict.get(each, {})   # each is the name of sub_parallel
            sub_temp = temp.get('seq', [])
            for each_sub_event in tuple_name[each]:  # each_sub_event is each sub_parallel list contains the positions of sub_parallel events
                for i in each_sub_event:   # i is each event which happens simultaneously in the sub parallel
                    paral_seq = find_sequence(df, end, i, sub_tuple)   # find the sequence start with this event
                    sub_temp.append(tuple(paral_seq))
            temp['seq'] = sub_temp
            temp_dict[each] = temp    # record the sequence under this sub_parallel name, stored in the dictionary with the key 'seq'
            get_seq_dict(False, sub_tuple, df, end, temp_dict, act_column)   # recurrent if there is (sub_)sub_parallel in this sub_parallel
    else:   # if the event is the first events in the parallel (happens simultaneously)
        if len(tuple_name) == 0:   # if there is no sub-parallel in the parallism
            sub_tuple = {}
            temp = temp_dict.get('seq',[])
            paral_seq = find_sequence(df, end, if_first, sub_tuple)   # find the sequence start with this event
            temp.append(tuple(paral_seq))
            temp_dict['seq'] = temp   # record the sequence of the parallel, stored under the key 'seq'
            get_seq_dict(False, sub_tuple, df, end, temp_dict, act_column)   # recurrent if there is (sub_)sub_parallel in this sub_parallel
            return paral_seq

# generate the key of dictionary according to its value(probability)
def get_first_parallel_event(tuple_name, my_prob):
    next_event_prob = my_prob[tuple_name]['first']
    next_event = number_of_certain_probability(list(next_event_prob.keys()), list(next_event_prob.values()))
    return next_event
def get_parallel_seqs(tuple_name, first_evt, my_prob):
    parallel_seq = my_prob[tuple_name]['seqs'][first_evt]
    possible_seq = number_of_certain_probability(list(parallel_seq.keys()), list(parallel_seq.values()))
    return possible_seq
def get_sub_parallel_seqs(tuple_name, sub_tuple_name, sub_evt, my_prob):
    sub_parallel_seq = my_prob[tuple_name][sub_tuple_name][sub_evt]
    sub_parallel_seq = number_of_certain_probability(list(sub_parallel_seq.keys()), list(sub_parallel_seq.values()))
    return sub_parallel_seq

# generate the sequences according to the name of parallel
def gen_para(tuple_name, my_prob):
    seq = []
    first_parallels = get_first_parallel_event(tuple_name, my_prob)
    for each in first_parallels:
        new_s = get_parallel_seqs(tuple_name, each, my_prob)
        seq.append(new_s)
    return seq

# gen the sub_parallel sequences according to the name of parallel and sub_parallel
def gen_sub_para(my_prob, if_in_parallel, index_id,
                 tuple_name, sub_tuple_name, i, act_name, act_position, last_events, next_events, index, caseid, if_in_para=1):
    preced = i
    pos = i
    next_list = []
    for each in sub_tuple_name:
        preceding = preced
        next_list.append(i+1)
        gen_sub_seq = get_sub_parallel_seqs(tuple_name, sub_tuple_name, each, my_prob)  # generate sub_parallel according to the name of parallel and name of sub_parallel
        for each_ in gen_sub_seq:
            if type(each_) != tuple:   # if generated sequence is sequential
                last_events.append(preceding)
                i+=1
#                 j+=1
                index[i] = each_
                act_position.append(i)
                preceding = i
                act_name.append(each_)
                if_in_parallel.append(if_in_para)
                next_events.append(i+1)
                index_id.append(caseid)
                return i
            else:   # if generated sequence has sub_parallel
                i = gen_sub_para(my_prob, if_in_parallel, index_id,
                                 tuple_name, each_, i,  act_name, act_position, last_events, next_events, index, caseid, if_in_para=1)
        next_events[-1] = -1
    next_events[pos] = tuple(next_list)

# traverse the generated sequence and add infos, if there exits sub_parallel name, add sub_parallel
def add_sub(my_prob, if_in_parallel, index_id, tuple_name, seq, startt, act_name, act_position, last_events, next_events, caseid, if_in_para=1):
    index = {}
    i = startt
#     j = 0
    firsts = []
    new_seq = []

    for each_seq in seq:
#         ll = []
        preceding = startt
        firsts.append(i+1)
        for each in each_seq:
            if type(each) != tuple:   # if there is no sub_paralism
                last_events.append(preceding)
                i+=1
#                 j+=1
                act_position.append(i)
                preceding = i
                act_name.append(each)
                if_in_parallel.append(if_in_para)
                index[i] = each
                next_events.append(i+1)
                index_id.append(caseid)
#                 ll.append(i)
            else:   # if there is sub_parallel
                i = gen_sub_para(my_prob, if_in_parallel, index_id,tuple_name, each, i, act_name, act_position, last_events, next_events, index, caseid, if_in_para=1)   # generate the sub_parallel according to the names of parallel and sub_parallel
        next_events[-1] = -1
        new_seq.append(i)
    return firsts  # the first events of parallelism that happened simultaneously


