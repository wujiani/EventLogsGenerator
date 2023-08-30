import pm4py
import os
import numpy as np
from collections import Counter
import click
import warnings
import sys
import pickle
import datetime

from log_gen_seq_utils import *
warnings.filterwarnings("ignore")

@click.command()
@click.option('--experiment-name', required=True, type=str)
@click.option('--import-file', required=True, type=str)
@click.option('--import-test-file', required=True, type=str)
@click.option('--id-column', default='caseid', type=str)
@click.option('--act-column', default='concept:name', type=str)
@click.option('--time-column', default='time:timestamp', type=str)
@click.option('--resource-column', default='user', type=str)
@click.option('--state-column', default='lifecycle:transition', type=str)
@click.option('--method', default='prefix', type=str)
@click.option('--num', default=2, type=int)
@click.option('--suffix', default=0, type=int)
@click.option('--train_size_generation', default=False, required=False, type=bool)
def main(experiment_name, import_file, import_test_file, id_column, act_column, time_column, resource_column, state_column, method, num, suffix, train_size_generation):


    dirStr, ext = os.path.splitext(import_file)
    file_name = dirStr.split("\\")[-1]
    print(file_name)

    output_folder = f'example_outputs\{experiment_name}'
    os.makedirs(output_folder, exist_ok=True)

    # info of test data, here we just need the amount of generated traces(len_case), so len_case can be anything else that we defined
    data_test = pm4py.convert_to_dataframe(pm4py.read.read_xes(import_test_file))
    # data_test.drop_duplicates(keep='first',inplace=True)
    caseid_list = list(data_test[id_column].unique())
    len_case = len(caseid_list)
    print(f"number of traces in test dataset: {len_case}")

    # if if_csv:
    #     df = pd.read_csv('C:\\Users\\19wuj\\Desktop\\test_0514\\bac_cut_time.csv')
    #     df_start = df[[id_column, act_column, start_time_column, resource_column]].rename(
    #         columns={start_time_column: time_column})
    #     df_start[state_column] = 'start'
    #
    #     df_end = df[[id_column, act_column, end_time_column, resource_column]].rename(
    #         columns={end_time_column: time_column})
    #     df_end[state_column] = 'complete'
    #
    #     df_start_end = pd.concat([df_start, df_end])
    #     df_start_end['index'] = df_start_end.index
    #
    #     new = pd.DataFrame()
    #     for key, group in df_start_end.groupby(id_column):
    #         temp = group.sort_values(by=[time_column, 'index'], ascending=[True, True]).reset_index(drop=True)
    #         new = pd.concat([new, temp], ignore_index=True)
    #     del new['index']
    #     df = new
    #     df.to_csv(f'{file_name}_unfold.csv', index=False)
    # else:

    data = pm4py.convert_to_dataframe(pm4py.read.read_xes(import_file))
    caseid_list_train = list(data[id_column].unique())
    len_case_train = len(caseid_list_train)
    print(f"number of traces in train dataset: {len_case_train}")
    # reorder the columns of dataset
    df = copy.deepcopy(pd.DataFrame(data, columns=[id_column, act_column, time_column, resource_column, state_column]))

    # pre-processing
    mydata = pd.DataFrame()
    for key, group in df.groupby(id_column):
        group = add_start_end(group, act_column, resource_column, state_column)   # add start event in the beginning and end in the last
        mydata = pd.concat([mydata, group], ignore_index=True)
    df = mydata
    df[time_column] = pd.to_datetime(df[time_column])

    # compute the waiting time and process time of one event, also record event(s) before it and event(s) after it
    df['waiting_time'] = None
    df['process_time'] = None
    df['last_complete_event'] = None  # record the name of events before current event
    df['preceding_evts'] = None  # record the name of events after current event
    df['paired_event'] = None  # record the position of paired event of current event
    df['next'] = None  # record the position of events after current event
    df['index'] = df.index  # record the position of current event
    for key, group in df.groupby(id_column):
        start_flag = 0
        i = list(group.index)[0]  # iterate all the events in one trace from the trace beginning
        preceding_evt = []
        not_complete_evt_idx = []  # record the uncompleted event position until current
        j = 0
        df['waiting_time'].loc[i] = 0  # the waiting time of the first event is 0
        not_complete_evt_idx.append(i)
        i += 1
        last_complete_evt_idx = i
        while j < len(group) - 1:
            j += 1
            cur_act = df.loc[i]
            if cur_act[state_column] == 'complete':
                start_flag = 0
                preceding_evt.append(cur_act.name)
                last_complete_evt_idx = i

                for each_idx in not_complete_evt_idx:  # try to find its paired event (i.e.: find its 'start' version)
                    to_pair = df.loc[each_idx]
                    if (cur_act[act_column] == to_pair[act_column]) \
                            and (cur_act[resource_column] == to_pair[resource_column]):
                        df['paired_event'].loc[i] = each_idx
                        not_complete_evt_idx.remove(each_idx)
                        df['process_time'].loc[i] = (
                                    df[time_column].loc[i] - df[time_column].loc[each_idx]).seconds  # compute process time
                        df['last_complete_event'].loc[i] = df['last_complete_event'].loc[
                            each_idx]  # record the same 'last_complete_event' as its paired(start version) event
                        break
            else:
                if start_flag == 1:  # meaning last event is in the 'start' state in the dataframe
                    df['preceding_evts'].loc[i] = df['preceding_evts'].loc[
                        i - 1]  # current event has the same preceding events as the event before it in the dataframe
                else:  # meaning last event is in the 'complete' state in the dataframe
                    df['preceding_evts'].loc[
                        i] = preceding_evt  # tuple(sorted, reverse=False))   # current event has brand new preceding events
                # record the current event position as in the 'next' column for these preceding events
                for each in df['preceding_evts'].loc[i]:
                    if df['next'].loc[each] == None:
                        df['next'].loc[each] = [i]
                    else:
                        temp = df['next'].loc[each]
                        temp.append(i)
                        df['next'].loc[each] = temp
                start_flag = 1
                preceding_evt = []
                not_complete_evt_idx.append(i)

                df['last_complete_event'].loc[i] = last_complete_evt_idx
                df['waiting_time'].loc[i] = (df[time_column].loc[i] - df[time_column].loc[
                    last_complete_evt_idx]).seconds  # compute waiting time
            i += 1

    all_seq = [] # list that contains all the sequences of traces
    sub_groups={} # dictionary consists the order and info of all parallel situations :
                                #keys(tuple)=the name of the parallel situation(2 entities, event before parallel and event after parallel),
                                #values(list)=consists all the orders and the infos of the parallelisms named as key

    # this for loop is used to collect the sequences of all traces,
    # when there is a parallel in the trace, it is expressed as a tuple(event before parallel, event after parallel)
    for key, group in df.groupby(id_column):
    #     group = add_start_end(group)
        i=0
        seq = []
        while i<len(group):
            cur_act = group.iloc[i]
            next_act = group.iloc[i+1]
            if (cur_act[act_column] == next_act[act_column]) \
            and (cur_act[resource_column] == next_act[resource_column]) \
            and (cur_act[state_column] == 'start') \
            and (next_act[state_column] == 'complete'): # means that there is no parallel
                seq.append(cur_act[act_column])
                i+=2
            else:   # if there is parallel
                a, i = deal_sub_group(group, i, sub_groups, act_column, resource_column, state_column) # to record the order and the info of the parallel
                seq.append(a)
        all_seq.append(seq)

    # compute the probabilities of next events after one seq1 in the trace sequences(may contain the name of parallel)
    if_sub_group=False
    sub_traces = get_sub_traces(all_seq, method, num, if_sub_group)
    final_data = get_transition(sub_traces)
    if os.path.exists(os.path.join(output_folder, f'sequence_prob_{file_name}.pkl')):
        print('found sequence probability')
        with open(os.path.join(output_folder, f'sequence_prob_{file_name}.pkl'), "rb") as pkl_file:
            sequence_prob = pickle.load(pkl_file)
    else:
        sequence_prob = {}   # dictionary: keys: all the seq1,
        #                                  values(dictionary): the next events(keys) and the probabilities(values)
        for each in final_data.items():
            counter = Counter(each[1])
            tol_events = len(each[1])
            number = np.array(list(counter.values()))
            prob = np.round(np.divide(number, tol_events), 10)
            sequence_prob[each[0]] = dict(zip(counter.keys(), prob))

        with open(os.path.join(output_folder, f'sequence_prob_{file_name}.pkl'), "wb") as f:
            pickle.dump(sequence_prob, f)


    first_parallels = {}  # record the first events happens simultaneously in the parallel
    # dict_sub_parallel = {}
    parallel_seqs = {}  # record the sequences
    dict_para = {}  # dict stores the info of parallelism
    for each_tuple_name in sub_groups:
        #     if dict_sub_parallel.get(each_tuple_name) == None:
        #         dict_sub_parallel[each_tuple_name] = {}
        if parallel_seqs.get(each_tuple_name) == None:
            parallel_seqs[each_tuple_name] = {}
        for each in sub_groups[each_tuple_name]:
            parallel_data = each
            parallel_data = pd.DataFrame(parallel_data)
            parallel_data.set_index(["index"], inplace=True)
            fillna_parallel_data(parallel_data, state_column)
            parallel_data['paired_event'] = parallel_data['paired_event'].astype(int)
            i = parallel_data.index[0]  # first event in the parallel
            ed = parallel_data.index[-1]  # last event in the parallel
            start_index = parallel_data['last_complete_event'].loc[i]  # position of events before the parallelism
            end_index = parallel_data['next'].loc[ed]  # position of events after the parallelism
            end = None
            if end_index == None:
                #             print(parallel_data)
                break
            for each_ in end_index:  # end_index is a list, we want the element in the list
                end = each_
            j = 0
            parallel_seq = []
            temp_dict = dict_para.get(each_tuple_name, {})

            while j < len(parallel_data):
                cur = parallel_data.loc[i]
                if cur[state_column] == 'start' and cur[
                    'last_complete_event'] == start_index:  # if event is the first events happens simultaneously in the parallel
                    seq_first = get_seq_dict(i, [], parallel_data, end, temp_dict, act_column)
                    parallel_seq.append(seq_first)

                i += 1
                j += 1
            dict_para[each_tuple_name] = temp_dict
            first_parallel = tuple(sorted([each[0] for each in parallel_seq], reverse=False))
            temp = first_parallels.get(each_tuple_name, [])
            temp.append(first_parallel)
            first_parallels[each_tuple_name] = temp

    # first_dict = {}
    all_para_seqs = {}
    for each_tuple in dict_para:
        seq_p = dict_para[each_tuple]['seq']
    #     seq_first = [each[0] for each in seq_p]   # get the first events happens simultaneously in the parallel
    #     first_dict[each_tuple] = seq_first
        all_para_seqs[each_tuple] = {}
        for each_seq in seq_p:
            temp = all_para_seqs[each_tuple].get(each_seq[0], [])
            temp.append(each_seq)
            all_para_seqs[each_tuple][each_seq[0]] = temp   # record the sequences start with the first event, the key is the parallel name and sub_key is the start event

    all_sub_para = {}
    for each_tuple in dict_para:
        all_sub_para[each_tuple] = {}
        for each_sub_tuple in dict_para[each_tuple].items():
            if each_sub_tuple[0] != 'seq':
                all_sub_para[each_tuple][each_sub_tuple[0]] = {}
                seq_p = each_sub_tuple[1]['seq']
    #             seq_first = [each[0] for each in seq_p]
    #             first_dict[each_tuple] = seq_first
                for each_seq in seq_p:
                    temp = all_sub_para[each_tuple][each_sub_tuple[0]].get(each_seq[0], [])
                    temp.append(each_seq)
                    all_sub_para[each_tuple][each_sub_tuple[0]][each_seq[0]] = temp   # record the seuqences start with the first event in the sub_parallel,
                                                                                    #the key is the parallel name and sub_key is the name of sub_parallel and (sub_)sub_key is the start event in the sub_parallel

    if os.path.exists(os.path.join(output_folder, f'my_prob_{file_name}.pkl')):
        print('found my probability')
        with open(os.path.join(output_folder, f'my_prob_{file_name}.pkl'), "rb") as pkl_file:
            my_prob = pickle.load(pkl_file)
    else:

        my_prob = {}  # dictionary stores probabilities

        for each in first_parallels.items():
            my_prob[each[0]] = {}
            a = each[1]
            counter = Counter(a)
            tol_events = len(a)
            number = np.array(list(counter.values()))
            prob = np.round(np.divide(number, tol_events), 10)
            my_prob[each[0]]['first'] = dict(zip(counter.keys(), prob))
            # the key is the combination of the first events in the parallel under the name of parallism and value is its probability

        for each in all_para_seqs.items():
            my_prob[each[0]]['seqs'] = {}
            for each_first_seq in each[1].items():
                a = each_first_seq[1]
                counter = Counter(a)
                tol_events = len(a)
                number = np.array(list(counter.values()))
                prob = np.round(np.divide(number, tol_events), 10)
                my_prob[each[0]]['seqs'][each_first_seq[0]] = dict(zip(counter.keys(), prob))
                # the key is sequence start with first event in the parallel under the name of parallism, then under the first event and value is its probability

        for each in all_sub_para.items():
            for each_sub_para in each[1].items():
                my_prob[each[0]][each_sub_para[0]] = {}
                for each_ in each_sub_para[1].items():
                    a = each_[1]
                    counter = Counter(a)
                    tol_events = len(a)
                    number = np.array(list(counter.values()))
                    prob = np.round(np.divide(number, tol_events), 10)
                    my_prob[each[0]][each_sub_para[0]][each_[0]] = dict(zip(counter.keys(), prob))
                    # the key is sequence start with first event in the sub parallel under the name of parallism, then under the name of sub_parallel, then under the first event
                    # and value is its probability
        with open(os.path.join(output_folder, f'my_prob_{file_name}.pkl'), "wb") as f:
            pickle.dump(my_prob, f)

    gen_number = len_case if not train_size_generation else len_case_train
    print(f"generating {gen_number}")
    # gen_number = 160
    new_data = gen_new_data(sequence_prob, gen_number, method) # generate new data

    index_id = []
    caseid = 0
    act_name = [] # a
    act_position = [] # b
    last_events = [] # c
    next_events = [] # d
    if_in_parallel = []
    if_in_para = 0
    i = -1
    preceding = 0
    last_tuple = False
    for each in new_data['task']:
        if each == 'addstart':
            caseid+=1
        if type(each) == tuple:
            startt = i   # start position of parallel(the event before the parallel)
            seq = gen_para(each, my_prob)  # generate parallelism
            firsts = add_sub(my_prob, if_in_parallel, index_id, each, seq, startt, act_name, act_position, last_events, next_events, caseid)   # first events position of parallelism
            next_events[startt] = tuple(firsts)
            i = act_position[-1]
            preced = []
            for index, each in enumerate(next_events):
                if each == -1:
                    preced.append(index)
            next_events =[i+1 if each ==-1 else each for each in next_events]
            last_events.append(tuple(preced))
            last_tuple = True
        else:
            if not last_tuple:
                last_events.append(preceding)
            last_tuple = False
            i+=1
            act_name.append(each)
            if_in_parallel.append(if_in_para)
            act_position.append(i)
            preceding = i
            next_events.append(i+1)
            index_id.append(caseid)

    dict_sub_data = { 'caseid':index_id, 'act': act_name, 'index':act_position, 'last': last_events, 'next':next_events, 'if_para': if_in_parallel}
    new_ = pd.DataFrame(dict_sub_data)
    # dddf = pd.DataFrame()
    # new_['start_timestamp'] = None
    # new_['end_timestamp'] = None
    # for key, group in new_.groupby('caseid'):
    #     group = group[1:-1]
    #     group['last'].iloc[0] = 'None'
    #     group['next'].iloc[-1] = 'None'
    #     for i in range(len(group)):
    #         cur = group.iloc[i]
    #         if cur['last'] == 'None':
    #             group['start_timestamp'].iloc[i] = pd.to_datetime('2012/3/13  7:00:00', format = '%Y/%m/%d  %H:%M:%S')
    #         elif type(cur['last']) == tuple:
    #             last_time = max(group['end_timestamp'].loc[list(cur['last'])])
    #             group['start_timestamp'].iloc[i] = last_time + pd.Timedelta(days=5, hours=12, minutes=50, seconds=20)
    #         else:
    #             last_time = group['end_timestamp'].loc[cur['last']]
    #             group['start_timestamp'].iloc[i] = last_time + pd.Timedelta(days=5, hours=12, minutes=50, seconds=20)
    #         group['end_timestamp'].iloc[i] = group['start_timestamp'].iloc[i] + pd.Timedelta(days=5, hours=12, minutes=50, seconds=20)

    #     dddf = pd.concat([dddf, group])


    dddf = pd.DataFrame()
    new_['start_timestamp'] = None
    new_['end_timestamp'] = None
    for key, group in new_.groupby('caseid'):
        a = []
        group = group[1:-1]
        group['last'].iloc[0] = 'None'
        group['next'].iloc[-1] = 'None'
        #     for i in range(len(group)):
        #         cur = group.iloc[i]

        #         if cur['last'] == 'None':
        #             group['start_timestamp'].iloc[i] = pd.to_datetime('2012/3/13  7:00:00', format = '%Y/%m/%d  %H:%M:%S')
        #         elif type(cur['last']) == tuple:
        #             last_time = max(group['end_timestamp'].loc[list(cur['last'])])
        #             group['start_timestamp'].iloc[i] = last_time + pd.Timedelta(days=5, hours=12, minutes=50, seconds=20)
        #         else:
        #             last_time = group['end_timestamp'].loc[cur['last']]
        #             group['start_timestamp'].iloc[i] = last_time + pd.Timedelta(days=5, hours=12, minutes=50, seconds=20)

        #         group['end_timestamp'].iloc[i] = group['start_timestamp'].iloc[i] + pd.Timedelta(days=5, hours=12, minutes=50, seconds=20)

        dddf = pd.concat([dddf, group])

    dddf['resource'] = dddf['act']


    # fileName = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    dddf.to_csv(os.path.join(output_folder, f'gen_seq_{file_name}_{suffix}.csv'))
    # g.to_csv(os.path.join(f'example_outputs\{experiment_name}', f'gen_seq_time_{file_name}.csv'), index=False)

if __name__ == "__main__":
    main(sys.argv[1:])

