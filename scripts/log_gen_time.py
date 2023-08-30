import pm4py
import os
from sklearn.neighbors import KernelDensity
from collections import Counter
import copy
import warnings
import click
import sys

from log_gen_time_utils import *
import pickle
from core_modules.instances_generator.prophet_generator import ProphetGenerator

warnings.filterwarnings("ignore")

@click.command()
@click.option('--experiment-name', required=True, type=str)
@click.option('--import-file', required=True, type=str)
@click.option('--import-test-file', required=True, type=str)
@click.option('--file', required=True, type=str)
@click.option('--simulator-model-folder', required=True, type=str)
@click.option('--embedding-matrix', required=True, type=str)
@click.option('--id-column', default='caseid', type=str)
@click.option('--act-column', default='concept:name', type=str)
@click.option('--time-column', default='time:timestamp', type=str)
@click.option('--resource-column', default='user', type=str)
@click.option('--state-column', default='lifecycle:transition', type=str)
@click.option('--suffix', default=0, type=int)
def main(experiment_name, import_file, import_test_file, file, simulator_model_folder, embedding_matrix, id_column, act_column, time_column, resource_column, state_column, suffix):

    output_folder = f'example_outputs\{experiment_name}'
    dirStr, ext = os.path.splitext(import_file)
    file_name = dirStr.split("\\")[-1]

    data = pm4py.convert_to_dataframe(pm4py.read.read_xes(import_file))
    # reorder the columns of dataset
    df = copy.deepcopy(pd.DataFrame(data, columns=[id_column, act_column, time_column, resource_column, state_column]))

    df['waiting_time'] = None
    df['process_time'] = None
    df['last_complete_event'] = None
    df['preceding_evts'] = None
    df['paired_event'] = None
    df['next'] = None
    df['index'] = df.index
    for key, group in df.groupby(id_column):
        flag = 0
        i = list(group.index)[0]
        preceding_evt = []
        not_complete_evt_idx = []
        j = 0
        df['waiting_time'].loc[i] = 0
        not_complete_evt_idx.append(i)
        i += 1
        last_complete_evt_idx = i
        while j < len(group) - 1:
            j += 1
            cur_act = df.loc[i]
            if cur_act[state_column] == 'complete':
                flag = 0
                preceding_evt.append(cur_act.name)
                last_complete_evt_idx = i

                for each_idx in not_complete_evt_idx:
                    to_pair = df.loc[each_idx]
                    if (cur_act[act_column] == to_pair[act_column]) \
                            and (cur_act[resource_column] == to_pair[resource_column]):
                        df['paired_event'].loc[i] = each_idx
                        not_complete_evt_idx.remove(each_idx)
                        df['process_time'].loc[i] = (
                                    df[time_column].loc[i] - df[time_column].loc[each_idx]).total_seconds()
                        df['last_complete_event'].loc[i] = df['last_complete_event'].loc[each_idx]
                        break

            else:
                if flag == 1:
                    df['preceding_evts'].loc[i] = df['preceding_evts'].loc[i - 1]
                else:
                    df['preceding_evts'].loc[i] = preceding_evt  # tuple(sorted, reverse=False))
                for each in df['preceding_evts'].loc[i]:
                    if df['next'].loc[each] == None:
                        df['next'].loc[each] = [i]
                    else:
                        temp = df['next'].loc[each]
                        temp.append(i)
                        df['next'].loc[each] = temp
                flag = 1
                preceding_evt = []
                not_complete_evt_idx.append(i)

                df['last_complete_event'].loc[i] = last_complete_evt_idx
                df['waiting_time'].loc[i] = (
                            df[time_column].loc[i] - df[time_column].loc[last_complete_evt_idx]).total_seconds()
            i += 1
    fillna_parallel_data(df, state_column)

    wait_column = 'waiting_time'
    proc_column = 'process_time'
    last_column = 'preceding_evts'
    next_column = 'next'

    def add_act_name(df):
        #     df[next_column] = df[next_column].map(lambda x: string_to_list(x))
        df['next_act'] = df[next_column].map(lambda x: get_name(df, string_to_list(x), act_column))
        df['last_act'] = df[last_column].map(lambda x: get_name(df, string_to_list(x), act_column))
        df = df[
            [id_column, act_column, resource_column, wait_column, proc_column, state_column, 'last_act', 'next_act']]
        return df

    df = add_act_name(df)
    df = df[df[state_column] == 'start']

    last_cur_next = {}
    for key, group in df.groupby(['last_act', act_column]):
        last_cur_next[key] = group

    process_data = {}
    waiting_data = {}
    for each in last_cur_next.items():
        process_data[each[0]] = each[1]['process_time'].to_list()
        waiting_data[each[0]] = each[1]['waiting_time'].to_list()

    dict_cur_last = {}
    for key, group in df.groupby(act_column):
        dict_cur_last[key] = list(group['last_act'].unique())

    process_data_use = copy.deepcopy(process_data)
    waiting_data_use = copy.deepcopy(waiting_data)

    if os.path.exists(os.path.join(output_folder, f'dict_time_{file_name}.pkl')):
        if os.path.exists(os.path.join(output_folder, f'dict_kde_{file_name}.pkl')):
            print('found dict time')
            print('found dict kde')
            with open(os.path.join(output_folder, f'dict_time_{file_name}.pkl'), "rb") as pkl_file:
                dict_time = pickle.load(pkl_file)
            with open(os.path.join(output_folder, f'dict_kde_{file_name}.pkl'), "rb") as pkl_file:
                dict_kde = pickle.load(pkl_file)

    else:

        dict_time = {}
        dict_kde = {}
        dict_time['pro'] = {}
        dict_kde['pro'] = {}
        for each in process_data_use.items():
            dict_time['pro'][each[0]] = {}
            data_list = each[1]
            count = Counter(data_list)
            tol_events = len(data_list)

            if 0 in count:
                dict_time['pro'][each[0]][0] = count[0] / tol_events
                del count[0]
                while 0 in data_list:
                    data_list.remove(0)
            if len(count) == 1:
                for each_ in count:
                    dict_time['pro'][each[0]][each_] = count[each_] / tol_events
            elif len(count) > 1:
                dict_time['pro'][each[0]]['kde'] = len(data_list) / tol_events
                data = np.array(data_list).reshape(-1, 1)
                kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data)
                dict_kde['pro'][each[0]] = kde


        dict_time['wait'] = {}
        dict_kde['wait'] = {}
        for each in waiting_data_use.items():
            dict_time['wait'][each[0]] = {}
            data_list = each[1]
            count = Counter(data_list)
            tol_events = len(data_list)

            if 0 in count:
                dict_time['wait'][each[0]][0] = count[0] / tol_events
                del count[0]
                while 0 in data_list:
                    data_list.remove(0)
            if len(count) == 1:
                for each_ in count:
                    dict_time['wait'][each[0]][each_] = count[each_] / tol_events
            elif len(count) > 1:
                dict_time['wait'][each[0]]['kde'] = len(data_list) / tol_events
                data = np.array(data_list).reshape(-1, 1)
                kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data)
                dict_kde['wait'][each[0]] = kde

        with open(os.path.join(output_folder, f'dict_time_{file_name}.pkl'), "wb") as f:
            pickle.dump(dict_time, f)
        with open(os.path.join(output_folder, f'dict_kde_{file_name}.pkl'), "wb") as f:
            pickle.dump(dict_kde, f)

    # read generated sequences
    gen_df = pd.read_csv(os.path.join(output_folder, f'gen_seq_{file_name}_{suffix}.csv'))
    gen_df = gen_df.set_index('Unnamed: 0')
    gen_df['next'] = gen_df['next'].astype('str')
    gen_df['last'] = gen_df['last'].astype('str')

    # get minimum date
    data_test = pm4py.convert_to_dataframe(pm4py.read.read_xes(import_test_file))
    start_time = data_test[time_column].min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

    num_instances = gen_df[id_column].nunique()
    print(f"num_instances: {num_instances}")
    ia_list = ProphetGenerator._generate(num_instances, start_time, simulator_model_folder, file)
    ia_list = [ia_list.iloc[i][0] for i in range(num_instances)]

    id_column = 'caseid'
    act_column = 'act'
    resource_column = 'resource'

    # wait_column = 'waiting_time'
    # proc_column = 'process_time'
    # last_column = 'preceding_evts'
    # next_column = 'next'

    def string_to_list_gen(c):
        if c == 'None':
            c = c
        else:
            c = c.strip('(')
            c = c.strip(')')
            c = c.split(',')
            c = [int(each) for each in c]
        return c

    gen_df['next'] = gen_df['next'].map(lambda x: string_to_list_gen(x))
    gen_df['next_act'] = gen_df['next'].map(lambda x: get_name(gen_df, x, act_column))
    gen_df['last'] = gen_df['last'].map(lambda x: string_to_list_gen(x))
    gen_df['last_act'] = gen_df['last'].map(lambda x: get_name(gen_df, x, act_column))

    with open(embedding_matrix) as f:
        emb = [each.split(',') for each in f.read().split('\n')]
    emb = pd.DataFrame(emb)
    emb = emb[:-1]

    act_index = dict(zip(emb[1].map(lambda x: x.strip()), emb[0].astype('int')))
    act_weights = emb.iloc[:, 2:].astype('float')
    act_weights = act_weights.T.to_dict('list')

    gen_add_time = pd.DataFrame()
    for key, group in gen_df.groupby(['last_act', 'act']):
        group['pro'] = None
        group['wait'] = None
        if key in process_data:
            group['pro'] = group['caseid'].map(lambda x: get_time(key, 'pro', dict_time, dict_kde))
        else:
            group['pro'] = group['caseid'].map(lambda x: get_time_no_key_exist(key, 3, 'pro', dict_cur_last, dict_time, dict_kde, act_index, act_weights))
        if key in waiting_data:
            group['wait'] = group['caseid'].map(lambda x: get_time(key, 'wait', dict_time, dict_kde))
        else:
            group['wait'] = group['caseid'].map(lambda x: get_time_no_key_exist(key, 3, 'wait', dict_cur_last, dict_time, dict_kde, act_index, act_weights))

        gen_add_time = pd.concat([gen_add_time, group])

    gen_add_time = gen_add_time.sort_values(by='Unnamed: 0', ascending=True)

    g = pd.DataFrame()
    for key, group in gen_add_time.groupby('caseid'):
        group = add_start_end(group)
        group = group.set_index('index')
        for i in range(len(group)):
            cur = group.iloc[i]
            if i == 0:
                group['start_timestamp'].iloc[i] = ia_list[int(key) - 1]
                group['end_timestamp'].iloc[i] = group['start_timestamp'].iloc[i] + pd.Timedelta(seconds=0)
            else:
                if len(cur['last']) > 1:
                    last_time = max(pd.to_datetime(group['end_timestamp'].loc[list(cur['last'])]))
                    group['start_timestamp'].iloc[i] = last_time + pd.Timedelta(seconds=group['wait'].iloc[i])
                else:
                    last_time = pd.to_datetime(group['end_timestamp'].loc[cur['last'][0]])
                    group['start_timestamp'].iloc[i] = last_time + pd.Timedelta(seconds=group['wait'].iloc[i])
                group['end_timestamp'].iloc[i] = group['start_timestamp'].iloc[i] + pd.Timedelta(
                    seconds=group['pro'].iloc[i])
        g = pd.concat([g, group])

    g = g.rename(columns={'act': 'task'})
    g['caseid'] = g['caseid'].map(lambda x: 'Case' + str(x))
    g = g[['caseid', 'task', 'start_timestamp', 'end_timestamp', 'resource']]

    output_folder = f'example_outputs\{experiment_name}'
    g.to_csv(os.path.join(output_folder, f'gen_seq_time_{file_name}_{suffix}.csv'), index=False)

if __name__ == "__main__":
    main(sys.argv[1:])