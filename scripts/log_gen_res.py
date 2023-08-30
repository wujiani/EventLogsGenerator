import pm4py
import os
import numpy as np
from collections import Counter
import click
import warnings
import sys
import copy
import pandas as pd
import pickle

from log_gen_res_utils import *
warnings.filterwarnings("ignore")

@click.command()
@click.option('--experiment-name', required=True, type=str)
@click.option('--import-file', required=True, type=str)
@click.option('--id-column', default='caseid', type=str)
@click.option('--act-column', default='concept:name', type=str)
@click.option('--time-column', default='time:timestamp', type=str)
@click.option('--resource-column', default='user', type=str)
@click.option('--state-column', default='lifecycle:transition', type=str)
@click.option('--suffix', default=0, type=int)

def main(experiment_name, import_file, id_column, act_column, time_column, resource_column, state_column, suffix):
    dirStr, ext = os.path.splitext(import_file)
    file_name = dirStr.split("\\")[-1]
    output_folder = f'example_outputs\{experiment_name}'

    data = pm4py.convert_to_dataframe(pm4py.read.read_xes(import_file))
    # reorder the columns of dataset
    df = copy.deepcopy(pd.DataFrame(data, columns=[id_column, act_column, time_column, resource_column, state_column]))

    df = df.rename(columns={act_column: 'task', resource_column: 'resource', state_column: 'state'})

    df = df[df['state'] == 'start']
    df = df.reset_index()

    add_occurence(df)

    df_gen = pd.read_csv(os.path.join(output_folder, f'gen_seq_time_{file_name}_{suffix}.csv'))
    add_occurence(df_gen)
    df_gen['res'] = None

    dict_re_by_task = resources_by_task(df)

    if os.path.exists(os.path.join(output_folder, f'dict_re_tasks_{file_name}.pkl')):
        if os.path.exists(os.path.join(output_folder, f'dict_re_caseid_{file_name}.pkl')):
            if os.path.exists(os.path.join(output_folder, f'dict_re_caseids_tasks_{file_name}.pkl')):
                if os.path.exists(os.path.join(output_folder, f'dict_caseid_tasks_{file_name}.pkl')):
                    print('found required dicts')
                    with open(os.path.join(output_folder, f'dict_re_tasks_{file_name}.pkl'), "rb") as pkl_file:
                        dict_re_tasks = pickle.load(pkl_file)
                    with open(os.path.join(output_folder, f'dict_re_caseid_{file_name}.pkl'), "rb") as pkl_file:
                        dict_re_caseid = pickle.load(pkl_file)
                    with open(os.path.join(output_folder, f'dict_re_caseids_tasks_{file_name}.pkl'), "rb") as pkl_file:
                        dict_re_caseids_tasks = pickle.load(pkl_file)
                    with open(os.path.join(output_folder, f'dict_caseid_tasks_{file_name}.pkl'), "rb") as pkl_file:
                        dict_caseid_tasks = pickle.load(pkl_file)
    else:
        dict_re_tasks, dict_re_caseid, dict_re_caseids_tasks, dict_caseid_tasks = tasks_by_resource(df)
        with open(os.path.join(output_folder, f'dict_re_tasks_{file_name}.pkl'), "wb") as f:
            pickle.dump(dict_re_tasks, f)
        with open(os.path.join(output_folder, f'dict_re_caseid_{file_name}.pkl'), "wb") as f:
            pickle.dump(dict_re_caseid, f)
        with open(os.path.join(output_folder, f'dict_re_caseids_tasks_{file_name}.pkl'), "wb") as f:
            pickle.dump(dict_re_caseids_tasks, f)
        with open(os.path.join(output_folder, f'dict_caseid_tasks_{file_name}.pkl'), "wb") as f:
            pickle.dump(dict_caseid_tasks, f)

    dict_tasks_re = {}
    for each in dict_re_tasks.items():
        for a in each[1]:
            tmp = dict_tasks_re.get(a, [])
            tmp.append(each[0])
            dict_tasks_re[a] = tmp
    # dict_tasks_re


    if os.path.exists(os.path.join(output_folder, f'tasks_re_prob_{file_name}.pkl')):
        print('found tasks_re_prob')
        with open(os.path.join(output_folder, f'tasks_re_prob_{file_name}.pkl'), "rb") as pkl_file:
            tasks_re_prob = pickle.load(pkl_file)
    else:

        tasks_re_prob = {}  # dictionary: keys: all the seq1,
        #                                  values(dictionary): the next events(keys) and the probabilities(values)
        for each in dict_tasks_re.items():
            counter = Counter(each[1])
            tol_events = len(each[1])
            number = np.array(list(counter.values()))
            prob = np.round(np.divide(number, tol_events), 10)
            tasks_re_prob[each[0]] = dict(zip(counter.keys(), prob))
        with open(os.path.join(output_folder, f'tasks_re_prob_{file_name}.pkl'), "wb") as f:
            pickle.dump(tasks_re_prob, f)

    all_acts_under_re = [each_[1] for each in dict_re_caseids_tasks.items() for each_ in each[1]]

    all_acts_under_re = set(sorted(all_acts_under_re, key=lambda x: len(x), reverse=True))

    def id_acts(df):
        dict_ = {}
        for key, group in df.groupby(id_column):
            dict_[key] = set(group['task_index'].to_list())
        return dict_

    dict_id_acts = id_acts(df_gen)
    dict_id_acts_train = id_acts(df)

    actsets_id = {}
    for each in all_acts_under_re:
        for each_ in dict_id_acts.items():
            #         print(each_[0])
            act_set = each_[1]
            if set(each).issubset(act_set):
                tmp = actsets_id.get(each_[0], [])
                tmp.append(each)
                actsets_id[each_[0]] = tmp
    #             dict_id_acts[each_[0]] = act_set-set(each)
    # actsets_id

    list_not_exist = []
    for each_ in dict_id_acts.items():
        exist = False
        for each in dict_id_acts_train.items():
            if each_[1] == each[1]:
                exist = True
        if exist == False:
            list_not_exist.append(each_[0])

    dict_gen_possible_comb = {}
    for each_item in actsets_id.items():

        caseid = each_item[0]
        #     if caseid == 'Case143':
        sets = each_item[1]
        if caseid not in list_not_exist:
            dict_gen_possible_comb[caseid] = {}
            #         print(caseid)
            #         print(dict_id_acts[caseid])#
            for each_train in dict_id_acts_train.items():
                if dict_id_acts[caseid].issubset(each_train[1]):
                    comb = tuple(set(dict_caseid_tasks[each_train[0]]))
                    left_acts = each_train[1]-dict_id_acts[caseid]
                    if len(left_acts) > 0:
                        left_union = find_containing_subsets(set(dict_caseid_tasks[each_train[0]]),left_acts)
                        if left_union == left_acts:
                            tmp = dict_gen_possible_comb[caseid].get(comb, [])
                            tmp.append(each_train[0])
                            dict_gen_possible_comb[caseid][comb] = tmp
                    else:
                        tmp = dict_gen_possible_comb[caseid].get(comb, [])
                        tmp.append(each_train[0])
                        dict_gen_possible_comb[caseid][comb] = tmp

    tasks_comb_prob = {}  # dictionary: keys: all the seq1,
    #                                  values(dictionary): the next events(keys) and the probabilities(values)
    for each in dict_gen_possible_comb.items():
        counter = dict(zip(each[1].keys(), [len(ee[1]) for ee in each[1].items()]))
        tol_events = sum(counter.values())
        number = np.array(list(counter.values()))
        prob = np.round(np.divide(number, tol_events), 10)
        tasks_comb_prob[each[0]] = dict(zip(counter.keys(), prob))

    dict_no_exit_tasks = {}
    for each_no_exit in list_not_exist:
        dict_no_exit_tasks[each_no_exit] = {}
        a = sorted(list(dict_id_acts[each_no_exit]))
        for each in dict_id_acts_train.items():
            b = sorted(each[1])
            diff = damerau_levenshtein_distance(a, b)
            tmp = dict_no_exit_tasks[each_no_exit].get(diff, {})
            tmpp = tmp.get(tuple(b), [])
            tmpp.append(dict_caseid_tasks[each[0]])
            tmp[tuple(b)] = tmpp
            dict_no_exit_tasks[each_no_exit][diff] = tmp

    for each in dict_no_exit_tasks.items():
        p = each[1]
        p = sorted(p.items(), key=lambda x: x[0], reverse=False)
        tmp = p[0][1]
        #     print(tmp.values())
        #     print('\n')
        dict_no_exit_tasks[each[0]] = [tuple(sorted(y)) for each in tmp.values() for y in each]

    no_exist_tasks_comb_prob = {}  # dictionary: keys: all the seq1,
    #                                  values(dictionary): the next events(keys) and the probabilities(values)
    for each in dict_no_exit_tasks.items():
        counter = Counter(each[1])
        tol_events = sum(counter.values())
        number = np.array(list(counter.values()))
        prob = np.round(np.divide(number, tol_events), 10)
        no_exist_tasks_comb_prob[each[0]] = dict(zip(counter.keys(), prob))
    tasks_comb_prob.update(no_exist_tasks_comb_prob)

    for each_id in tasks_comb_prob.keys():
        if len(tasks_comb_prob[each_id]) != 0:
            tasks_comb = possible_tasks_comb(each_id, tasks_comb_prob)
            for each_tasks in tasks_comb:
                resource = possible_resource(each_tasks, tasks_re_prob)
                for each_task in each_tasks:
                    #             print(each_task, resource)
                    #                 task_index = df_gen[df_gen[id_column] == each_id][df_gen['task_index'] == each_task]['index'].iloc[0]
                    task_index = df_gen[df_gen[id_column] == each_id][df_gen['task_index'] == each_task]['index']
                    if len(task_index) != 0:
                        df_gen['res'].loc[task_index.iloc[0]] = resource

    df_gen['res'] = df_gen['index'].map(lambda x: fillna(x, df_gen, dict_re_by_task))

    df_gen_output = df_gen[[id_column, 'task', 'start_timestamp', 'end_timestamp', 'res']]
    df_gen_output = df_gen_output.rename(columns={'res': 'resource'})

    df_gen_output.to_csv(os.path.join(output_folder, f'gen_seq_time_res_{file_name}_{suffix}.csv'), index=False)


if __name__ == "__main__":
    main(sys.argv[1:])