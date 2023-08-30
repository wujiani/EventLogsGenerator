# -*- coding: utf-8 -*-
from sys import stdout
import numpy as np
import datetime
import os
import csv
import uuid
import json
import platform as pl
from networkx.readwrite import json_graph
import time
import functools
import traceback
import pandas as pd


def folder_id():
    return (datetime.datetime.today()
            .strftime('%Y%m%d_') + str(uuid.uuid4())
            .upper()
            .replace('-', '_'))


def folder_id_with_prefix(prefix):
    return prefix + (datetime.datetime.today()
                     .strftime('%Y%m%d'))


def file_id(prefix='',extension='.csv'):
    return (prefix+datetime.datetime.today()
            .strftime('%Y%m%d_%H%M%S%f')+extension)

#generate unique bimp element ids
def gen_id():
    return "qbp_" + str(uuid.uuid4())

#printing process functions
def print_progress(percentage, text):
    stdout.write("\r%s" % text + str(percentage)[0:5] + chr(37) + "...      ")
    stdout.flush()

def print_performed_task(text):
    stdout.write("\r%s" % text + "...      ")
    stdout.flush()

def print_done_task():
    stdout.write("[DONE]")
    stdout.flush()
    stdout.write("\n")

def file_size(path_file):
    size = 0
    file_exist = os.path.exists(path_file)
    if file_exist:
        size = len(open(path_file).readlines())
    return size

#printing formated float
def ffloat(num, dec):
    return float("{0:.2f}".format(np.round(num,decimals=dec)))

#transform a string into date object
def get_time_obj(date, timeformat):
    date_modified = datetime.datetime.strptime(date,timeformat)
    return date_modified

#reduce list of lists with no repetitions
def reduce_list(input, dtype='int'):
    text = str(input).replace('[', '').replace(']', '')
    text = [x for x in text.split(',') if x != ' ']
    if text and not text == ['']:
        if dtype=='int':
            return list(set([int(x) for x in text]))
        elif dtype=='float':
            return list(set([float(x) for x in text]))
        elif dtype=='str':
            return list(set([x.strip() for x in text]))
        else:
            raise ValueError(dtype)
    else:
        return list()

#print a csv file from list of lists
def create_file_from_list(index, output_file):
    with open(output_file, 'w') as f:
        for element in index:
            f.write(', '.join(list(map(lambda x: str(x), element))))
            f.write('\n')
        f.close()

#print a csv file from list of lists
def create_text_file(index, output_file):
    with open(output_file, 'w') as f:
        for element in index:
            f.write(element+'\n')
        f.close()

#print debuging csv file
def create_csv_file(index, output_file, mode='w'):
    with open(output_file, mode) as f:
        for element in index:
            w = csv.DictWriter(f, element.keys())
            w.writerow(element)
        f.close()

def create_csv_file_header(index, output_file, mode='w'):
    with open(output_file, mode, newline='') as f:
        fieldnames = index[0].keys()
        w = csv.DictWriter(f, fieldnames)
        w.writeheader()
        for element in index:
            w.writerow(element)
        f.close()

def create_json(dictionary, output_file):
    with open(output_file, 'w') as f:
         f.write(json.dumps(dictionary, indent=4, sort_keys=True))
         f.close()


def round_preserve(l, expected_sum):
    '''
    rounding lists values preserving the sum values
    '''
    actual_sum = sum(l)
    difference = round(expected_sum - actual_sum, 2)
    if difference > 0.00:
        idx= l.index(min(l))
    else:
        idx= l.index(max(l))
    l[idx] +=difference
    return l


def avoid_zero_prob(l):
    if len(l) == 2:
        if l[0] == 0.00:
            l = [0.01, 0.99]
        elif l[1]==0:
            l = [0.99, 0.01]
    return l


def create_symetric_list(width, length):
    positions = list()
    numbers = list()
    [positions.append(width * (i + 1)) for i in range(0, length)]
    a = np.median(positions)
    [numbers.append(x - a) for x in positions]
    return numbers

def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x == 0 else x for x in values]

def copy(source, destiny):
    if pl.system().lower() == 'windows':
        os.system('copy "' + source + '" "' + destiny + '"')
    else:
        os.system('cp "' + source + '" "' + destiny + '"')


def save_graph(graph, output_file):
    data = json_graph.node_link_data(graph)
    with open(output_file, 'w') as f:
        f.write(json.dumps(data))
        f.close()

def timeit(func=None, rec_name=None) -> dict:
    """
    Decorator to measure execution times of methods

    Parameters
    ----------
    method : Any method.

    Returns
    -------
    dict : execution time record

    """
    if not func:
        return functools.partial(timeit, rec_name=rec_name)
    @functools.wraps(func)
    def wrapper(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = rec_name if rec_name else kw.get('log_name', func.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('executed %r  %2.2f ms' % \
                  (func.__name__, (te - ts) * 1000))
        return result
    return wrapper

def safe_exec(method):
    """
    Decorator to safe execute methods and return the state
    ----------
    method : Any method.
    Returns
    -------
    dict : execution status
    """
    def safety_check(*args, **kw):
        is_safe = kw.get('is_safe', method.__name__.upper())
        if is_safe:
            try:
                method(*args)
            except Exception as e:
                print(e)
                traceback.print_exc()
                is_safe = False
        return is_safe
    return safety_check


def df_export_xes(df, output_filename, id_column, act_column,  resource_column, end_time_column, start_time_column):

    state_column = 'lifecycle:transition'
    time_column = 'time:timestamp'
    df[resource_column].fillna('no_resource', inplace=True)

    df_start = df[[id_column, act_column, start_time_column, resource_column]].rename(columns={start_time_column:time_column})
    df_start[state_column] = 'start'
    df_end = df[[id_column, act_column, end_time_column, resource_column]].rename(columns={end_time_column:time_column})
    df_end[state_column] = 'complete'
    df_start_end = pd.concat([df_start, df_end])
    df_start_end['index'] = df_start_end.index
    df_start_end[time_column] = pd.to_datetime(df_start_end[time_column], format='%Y/%m/%d  %H:%M:%S')
    new = pd.DataFrame()
    for key, group in df_start_end.groupby(id_column):
        temp = group.sort_values(by=[time_column, 'index'], ascending=[True, True]).reset_index(drop=True)
        new = pd.concat([new,temp],ignore_index = True)
    del new['index']
    df = new
    #     print(new.columns)
    df['concept:name'] = df[act_column].astype('str')
    df[id_column] = df[id_column].astype('str')
    df = df[[id_column, 'concept:name', state_column, resource_column, time_column, act_column]]
    #     df = df[[id_column, act_column,  state_column, resource_column, time_column]]
    df = df.rename(columns = {id_column: 'caseid', resource_column: 'user', act_column: 'task'})

    from pm4py.objects.log.exporter.xes import exporter as xes_exporter
    from pm4py.util import constants

    parameters = {}
    parameters[constants.PARAMETER_CONSTANT_CASEID_KEY] = "caseid"
    parameters["extensions"] = None

    xes_exporter.apply(df, output_filename, parameters=parameters)


