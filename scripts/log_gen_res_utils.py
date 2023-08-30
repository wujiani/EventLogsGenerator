import random
import itertools

def add_occurence(df):
    df['task_index'] = None
    df['index'] = df.index
    for key, group in df.groupby('caseid'):
        for key_, group_ in group.groupby('task'):
            group_ = group_.reset_index()
            group_['occurence'] = group_.index
            for _, a in group_.iterrows():
                df['task_index'].iloc[a['index']] = a['task'] + str(a['occurence'])

def resources_by_task(df):
    dict_ = {}
    for key, group in df.groupby('task'):
        dict_[key] = group['resource'].to_list()
    return dict_


def tasks_by_resource(df):
    dict_re_tasks = {}
    dict_re_caseid = {}
    dict_re_caseids_tasks = {}
    dict_caseid_tasks = {}

    aa = {}
    for key, group in df.groupby('caseid'):
        for key_, group_ in group.groupby('resource'):
            temp = dict_re_caseids_tasks.get(key_, [])
            temp.append((key, tuple(group_['task_index'].to_list())))
            dict_re_caseids_tasks[key_] = temp

            tmp = dict_caseid_tasks.get(key, [])
            tmp.append(tuple(group_['task_index'].to_list()))
            dict_caseid_tasks[key] = tmp

    for each in dict_re_caseids_tasks.items():
        dict_re_tasks[each[0]] = [a[1] for a in each[1]]
        aa[each[0]] = {}
        for a in each[1]:
            tmp = aa[each[0]].get(a[1], [])
            tmp.append(a[0])
            aa[each[0]][a[1]] = tmp
    for each in aa.items():
        dict_re_caseid[each[0]] = [a[1] for a in each[1].items()]
    return dict_re_tasks, dict_re_caseid, dict_re_caseids_tasks, dict_caseid_tasks

def get_subset(mylist):
    all_sub_sets = []
    n = len(mylist)
    for num in range(n):
        for i in itertools.combinations(mylist, num + 1):
            all_sub_sets.append(i)
    return all_sub_sets

def get_comb(sets, acts):
    all_sub_sets = get_subset(sets)
    possible_combination = []
    for each_sub in all_sub_sets:
        union = set(each_sub[0]).union(*each_sub[1:])
        if union == acts:
            if len([ee for e in each_sub for ee in e]) == len(acts):
                #                 print(each_sub)
                possible_combination.append(each_sub)
    return possible_combination

def damerau_levenshtein_distance(string1, string2):
    m = len(string1)
    n = len(string2)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if string1[i - 1] == string2[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1
    return d[m][n]

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



def possible_resource(tasks, tasks_re_prob):
    resource_prob = tasks_re_prob[tasks]
    resource = number_of_certain_probability(list(resource_prob.keys()), list(resource_prob.values()))
    return resource


def possible_tasks_comb(caseid, tasks_comb_prob):
    tasks_com_prob = tasks_comb_prob[caseid]
    comb = number_of_certain_probability(list(tasks_com_prob.keys()), list(tasks_com_prob.values()))
    return comb

def possible_tasks_comb_no_exist(caseid, tasks_comb_prob):
    tasks_com_prob = tasks_comb_prob[caseid]
    comb = number_of_certain_probability(list(tasks_com_prob.keys()), list(tasks_com_prob.values()))
    return comb

def fillna(index, df_gen, dict_re_by_task):
    if df_gen.loc[index]['res'] is None:
        task =  df_gen.loc[index]['task']
        return random.sample(dict_re_by_task[task], 1)[0]
    else:
        return df_gen.loc[index]['res']

def find_containing_subsets(A, b):
    A = convert_tuples_to_sets(A)
    result = []  # 用于存储含有每个元素的小集合
    for element in b:
        containing_subsets = [subset for subset in A if element in subset]
        if containing_subsets:
            result.extend(containing_subsets)
    return set.union(*map(set, result))


def convert_tuples_to_sets(A):
    result = [set(subset) for subset in A]
    return result