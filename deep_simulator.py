# -*- coding: utf-8 -*-
"""
@author: Manuel Camargo
"""
import os
import copy
import shutil
import datetime
import time


import pandas as pd
import numpy as np
from operator import itemgetter

import utils.support as sup
from utils.support import timeit, safe_exec
import readers.log_reader as lr
import readers.bpmn_reader as br
import readers.process_structure as gph
import readers.log_splitter as ls
import analyzers.sim_evaluator as sim

from core_modules.instances_generator import instances_generator as gen
from core_modules.sequences_generator import seq_generator as sg
from core_modules.times_allocator import times_generator as ta

from sklearn.decomposition import PCA, TruncatedSVD, DictionaryLearning
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score

import warnings
warnings.filterwarnings("ignore")

class DeepSimulator():
    """
    Main class of the Simulator
    """

    def __init__(self, parms):
        """constructor"""
        self.parms = parms
        self.is_safe = True
        self.sim_values = list()

    def execute_pipeline(self) -> None:
        exec_times = dict()
        self.is_safe = self._read_inputs(
            log_time=exec_times, is_safe=self.is_safe)
        # modify number of instances in the model
        num_inst = len(self._log_train.caseid.unique()) if self.parms['train_size_generation'] else len(self.log_test.caseid.unique())
        # get minimum date
        start_time = (self.log_test.start_timestamp.min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"))
        print('############ Structure optimization ############')
        # Structure optimization
        seq_gen = sg.SeqGenerator({**self.parms['gl'],
                                   **self.parms['s_gen']},
                                  self.log_train)
        print('############ Generate interarrivals ############')
        self.is_safe = self._read_bpmn(
            log_time=exec_times, is_safe=self.is_safe)
        generator = gen.InstancesGenerator(self.process_graph,
                                           self.log_train,
                                           self.parms['i_gen']['gen_method'],
                                           {**self.parms['gl'],
                                            **self.parms['i_gen']})
        print('########### Generate instances times ###########')
        times_allocator = ta.TimesGenerator(self.process_graph,
                                            self.log_train,
                                            {**self.parms['gl'],
                                             **self.parms['t_gen']})
        _prefix = self.parms['gl']['file'].split('.')[0].replace('-', '_') + "_"
        _prefix = _prefix + '_train_size' if self.parms['train_size_generation'] else _prefix
        output_path = os.path.join('output_files',
                                   sup.folder_id_with_prefix(_prefix))
        for rep_num in range(0, self.parms['gl']['exp_reps']):
            gen_loop_start_time = time.time()
            seq_gen.generate(num_inst, start_time)
            # TODO: remover esto, es simplemente para test
            if self.parms['i_gen']['gen_method'] == 'test':
                iarr = generator.generate(self.log_test, start_time)
            else:
                iarr = generator.generate(num_inst, start_time)
            event_log = times_allocator.generate(seq_gen.gen_seqs, iarr)
            event_log = pd.DataFrame(event_log)
            # Export log
            self._export_log(event_log, output_path, rep_num)
            # Evaluate log
            if self.parms['gl']['evaluate']:
                self.sim_values.extend(
                    self._evaluate_logs(self.parms, self.log_test,
                                        event_log, rep_num))
            gen_loop_end_time = time.time()
            print(f"gen loop n.{rep_num + 1} ended in: {(gen_loop_end_time - gen_loop_start_time):.6f} seconds")
        self._export_results(output_path)
        print("-- End of trial --")

    @timeit
    @safe_exec
    def _read_inputs(self, **kwargs) -> None:
        # Event log reading
        self.log = lr.LogReader(os.path.join(self.parms['gl']['event_logs_path'],
                                             self.parms['gl']['file']),
                                self.parms['gl']['read_options'])
        # Time splitting 80-20
        self._split_timeline(0.8,
                             self.parms['gl']['read_options']['one_timestamp'])

    @timeit
    @safe_exec
    def _read_bpmn(self, **kwargs) -> None:
        bpmn_path = os.path.join(self.parms['gl']['bpmn_models'],
                                 self.parms['gl']['file'].split('.')[0] + '.bpmn')
        self.bpmn = br.BpmnReader(bpmn_path)
        self.process_graph = gph.create_process_structure(self.bpmn)

    @staticmethod
    def _evaluate_logs(parms, log, sim_log, rep_num):
        """Reads the simulation results stats
        Args:
            settings (dict): Path to jar and file names
            rep (int): repetition number
        """
        # print('Reading repetition:', (rep+1), sep=' ')
        sim_values = list()
        log = copy.deepcopy(log)
        log = log[~log.task.isin(['Start', 'End'])]
        sim_log = sim_log[~sim_log.task.isin(['Start', 'End'])]
        log['source'] = 'log'
        log.rename(columns={'user': 'resource'}, inplace=True)
        log['caseid'] = log['caseid'].astype(str)
        log['caseid'] = 'Case' + log['caseid']
        evaluator = sim.SimilarityEvaluator(
            log,
            sim_log,
            parms['gl'],
            max_cases=1000)
        metrics = [parms['gl']['sim_metric']]
        if 'add_metrics' in parms['gl'].keys():
            metrics = list(set(list(parms['gl']['add_metrics']) + metrics))
        for metric in metrics:
            evaluator.measure_distance(metric)
            sim_values.append({**{'run_num': rep_num}, **evaluator.similarity})
        return sim_values

    def _export_log(self, event_log, output_path, r_num) -> None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        event_log.to_csv(
            os.path.join(
                output_path, 'gen_' +
                             datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                             + '_' + str(r_num + 1) + '.csv'),
            index=False)

    def clustering_method(self, dataframe, method, K=3):
    
        cols = [x for x in dataframe.columns if 'id_' in x]
        X = dataframe[cols]

        if method == 'kmeans':
            kmeans = KMeans(n_clusters = K, random_state=30).fit(X)
            dataframe['cluster'] = kmeans.labels_
        elif method == 'mean_shift':
            ms = MeanShift(bandwidth=K, bin_seeding=True, random_state=30).fit(X)
            dataframe['cluster'] = ms.labels_
        elif method == 'gaussian_mixture':
            dataframe['cluster'] = GaussianMixture(n_components=K, covariance_type='spherical', random_state=30).fit_predict(X)
            
        return dataframe

    def decomposition_method(self, dataframe, method):
    
        cols = [x for x in dataframe.columns if 'id_' in x]
        X = dataframe[cols]
        
        if method == 'pca':
            dataframe[['x', 'y', 'z']] = PCA(n_components=3).fit_transform(X)
        elif method == 'truncated_svd':
            dataframe[['x', 'y', 'z']] = TruncatedSVD(n_components=3).fit_transform(X)
        elif method == 'dictionary_learning':
            dataframe[['x', 'y', 'z']] = DictionaryLearning(n_components=3, transform_algorithm='lasso_lars').fit_transform(X)
            
        return dataframe

    def _clustering_metrics(self, params):

        file_name = params['gl']['file']
        embedded_path = params['gl']['embedded_path']
        concat_method = params['t_gen']['concat_method']
        include_times = params['t_gen']['include_times']
        
        if params['t_gen']['emb_method'] == 'emb_dot_product':
            emb_path = os.path.join(embedded_path, 'ac_DP_' + file_name.split('.')[0]+'.emb')
        elif params['t_gen']['emb_method'] == 'emb_w2vec':
            emb_path = os.path.join( embedded_path, 'ac_W2V_' + '{}_'.format(concat_method) + file_name.split('.')[0] + '.emb')
        elif params['t_gen']['emb_method'] == 'emb_dot_product_times':
            emb_path = os.path.join( embedded_path, 'ac_DP_times_' + file_name.split('.')[0] + '.emb')
        elif params['t_gen']['emb_method'] == 'emb_dot_product_act_weighting' and include_times:
            emb_path = os.path.join( embedded_path, 'ac_DP_act_weighting_times_' + file_name.split('.')[0] + '.emb')
        elif params['t_gen']['emb_method'] == 'emb_dot_product_act_weighting' and not include_times:
            emb_path = os.path.join( embedded_path, 'ac_DP_act_weighting_no_times_' + file_name.split('.')[0] + '.emb')

        print(emb_path)
        df_embeddings = pd.read_csv(emb_path, header=None)
        n_cols = len(df_embeddings.columns)
        df_embeddings.columns = ['id', 'task_name'] + ['id_{}'.format(idx) for idx in range(1, n_cols-1)]
        df_embeddings['task_name'] = df_embeddings['task_name'].str.lstrip()
        
        """
        clustering_ms = ['kmeans', 'gaussian_mixture']
        decomposition_ms = ['pca', 'truncated_svd']
        KS = [3, 5, 7]
        """

        clustering_ms = ['kmeans']
        decomposition_ms = ['pca']
        KS = [3]

        metrics = []
        for clustering_m in clustering_ms:
            for decomposition_m in decomposition_ms:
                for K in KS:
                    df_embeddings_tmp = self.clustering_method(df_embeddings, clustering_m, K)
                    df_embeddings_tmp = self.decomposition_method(df_embeddings_tmp, decomposition_m)
                    s_score = silhouette_score(df_embeddings_tmp[['x', 'y', 'z']], df_embeddings_tmp['cluster'], metric='euclidean')
                    ch_score = calinski_harabasz_score(df_embeddings_tmp[['x', 'y', 'z']], df_embeddings_tmp['cluster'])
                    metrics.append([clustering_m, decomposition_m, K, s_score, ch_score])

        metrics_df = pd.DataFrame(data= metrics, columns = ['clustering_method', 'decomposition_method', 'number_clusters', 'silhouette_score', 'calinski_harabasz_score'])
        best = metrics_df.sort_values(by=['silhouette_score', 'calinski_harabasz_score'], ascending=True).head(1)

        return best.T.reset_index()

    def _export_results(self, output_path, evaluation_only=False) -> None:

        clust_mets = self._clustering_metrics(self.parms)
        clust_mets.columns = ['metric', 'sim_val']
        clust_mets['run_num'] = 0.0
        sim_values_df = pd.DataFrame(self.sim_values).sort_values(by='metric')
        results_df = pd.concat([sim_values_df, clust_mets])

        # Save results
        results_df.to_csv(
            os.path.join(output_path, sup.file_id(prefix='SE_')),
            index=False)

        if not evaluation_only:

            file_name = self.parms['gl']['file']
            embedded_path = self.parms['gl']['embedded_path']
            concat_method = self.parms['t_gen']['concat_method']

            if self.parms['t_gen']['emb_method'] == 'emb_dot_product':
                emb_path = 'ac_DP_' + file_name.split('.')[0]+'.emb'
                embedd_method = 'Dot product'
                input_method = 'No aplica'
                include_times = 'No aplica'
            elif self.parms['t_gen']['emb_method'] == 'emb_w2vec':
                if self.parms['t_gen']['include_times']:
                    emb_path = 'ac_W2V_' + '{}_times_'.format(concat_method) + file_name.split('.')[0] +'.emb'
                    embedd_method = 'Word2vec'
                    input_method = self.parms['t_gen']['concat_method']
                    include_times = self.parms['t_gen']['include_times']
                else:
                    emb_path = 'ac_W2V_' + '{}_no_times_'.format(concat_method) + file_name.split('.')[0] +'.emb'
                    embedd_method = 'Word2vec'
                    input_method = self.parms['t_gen']['concat_method']
                    include_times = self.parms['t_gen']['include_times']
            elif self.parms['t_gen']['emb_method'] == 'emb_dot_product_times':
                emb_path = 'ac_DP_times_' + file_name.split('.')[0] + '.emb'
                embedd_method = 'Dot product'
                input_method = 'Times'
                include_times = True
            elif self.parms['t_gen']['emb_method'] == 'emb_dot_product_act_weighting':
                if self.parms['t_gen']['include_times']:
                    emb_path = 'ac_DP_act_weighting_times_' + file_name.split('.')[0] + '.emb'
                    embedd_method = 'Dot product'
                    input_method = 'Activity weighting'
                    include_times = self.parms['t_gen']['include_times']
                else:
                    emb_path = 'ac_DP_act_weighting_no_times_' + file_name.split('.')[0] + '.emb'
                    embedd_method = 'Dot product'
                    input_method = 'Activity weighting'
                    include_times = self.parms['t_gen']['include_times']

            results_df_T = results_df.set_index('metric').T.reset_index(drop=True)
            results_df_T['input_method'] = input_method
            results_df_T['embedding_method'] = embedd_method
            results_df_T['log_name'] = self.parms['gl']['file']
            results_df_T['times_included'] = include_times

            results_df_T.to_csv(
                os.path.join('output_files', emb_path.replace('.emb', '.csv')),
                index=False)

            # Save logs
            # log_test = self.log_test[~self.log_test.task.isin(['Start', 'End'])]
            log_test = self.log_test
            log_test.to_csv(
                os.path.join(output_path, 'test_' +
                             self.parms['gl']['file'].split('.')[0] + '.csv'),
                index=False)
            sup.df_export_xes(log_test.copy(),
                              os.path.join(output_path, 'test_' +
                                           self.parms['gl']['file'].split('.')[0] + '.xes'),
                              *log_test.columns)

            log_train = pd.DataFrame(self.log_train.data)
            log_train.to_csv(
                os.path.join(output_path, 'train_' +
                             self.parms['gl']['file'].split('.')[0] + '.csv'),
                index=False)
            sup.df_export_xes(log_train.copy(),
                              os.path.join(output_path, 'train_' +
                                           self.parms['gl']['file'].split('.')[0] + '.xes'),
                              *log_train.columns)

            if self.parms['gl']['save_models']:
                paths = ['bpmn_models', 'embedded_path', 'ia_gen_path',
                         'seq_flow_gen_path', 'times_gen_path']
                sources = list()
                for path in paths:
                    for root, dirs, files in os.walk(self.parms['gl'][path]):
                        for file in files:
                            if self.parms['gl']['file'].split('.')[0] in file:
                                sources.append(os.path.join(root, file))
                for source in sources:
                    base_folder = os.path.join(
                        output_path, os.path.basename(os.path.dirname(source)))
                    if not os.path.exists(base_folder):
                        os.makedirs(base_folder)
                    destination = os.path.join(base_folder,
                                               os.path.basename(source))
                    # Copy dl models
                    allowed_ext = self._define_model_path({**self.parms['gl'],
                                                           **self.parms['t_gen']})
                    is_dual = self.parms['t_gen']['model_type'] == 'dual_inter'
                    if is_dual and ('times_gen_models' in source) and any(
                            [x in source for x in allowed_ext]):
                        shutil.copyfile(source, destination)
                    elif not is_dual and ('times_gen_models' in source) and any(
                            [self.parms['gl']['file'].split('.')[0] + x in source
                             for x in allowed_ext]):
                        shutil.copyfile(source, destination)
                    # copy other models
                    folders = ['bpmn_models', 'embedded_matix', 'ia_gen_models']
                    allowed_ext = ['.emb', '.bpmn', '_mpdf.json', '_prf.json',
                                   '_prf_meta.json', '_mpdf_meta.json', '_meta.json']
                    if any([x in source for x in folders]) and any(
                            [self.parms['gl']['file'].split('.')[0] + x in source
                             for x in allowed_ext]):
                        shutil.copyfile(source, destination)

    @staticmethod
    def _define_model_path(parms):
        inter = parms['model_type'] in ['inter', 'dual_inter', 'inter_nt']
        is_dual = parms['model_type'] == 'dual_inter'
        next_ac = parms['model_type'] == 'inter_nt'
        arpool = parms['all_r_pool']
        if inter:
            if is_dual:
                if arpool:
                    return ['_dpiapr', '_dwiapr', '_diapr']
                else:
                    return ['_dpispr', '_dwispr', '_dispr']
            else:
                if next_ac:
                    if arpool:
                        return ['_inapr']
                    else:
                        return ['_inspr']
                else:
                    if arpool:
                        return ['_iapr']
                    else:
                        return ['_ispr']
        else:
            return ['.h5', '_scaler.pkl', '_meta.json']

    @staticmethod
    def _save_times(times, parms):
        times = [{**{'output': parms['output']}, **times}]
        log_file = os.path.join('output_files', 'execution_times.csv')
        if not os.path.exists(log_file):
            open(log_file, 'w').close()
        if os.path.getsize(log_file) > 0:
            sup.create_csv_file(times, log_file, mode='a')
        else:
            sup.create_csv_file_header(times, log_file)

    # =============================================================================
    # Support methods
    # =============================================================================
    def _split_timeline(self, size: float, one_ts: bool) -> None:
        """
        Split an event log dataframe by time to peform split-validation.
        prefered method time splitting removing incomplete traces.
        If the testing set is smaller than the 10% of the log size
        the second method is sort by traces start and split taking the whole
        traces no matter if they are contained in the timeframe or not

        Parameters
        ----------
        size : float, validation percentage.
        one_ts : bool, Support only one timestamp.
        """
        # Split log data
        splitter = ls.LogSplitter(self.log.data)
        train, test = splitter.split_log('random', size, one_ts) # empirically verified that random splitting is better than time splitting
        total_events = len(self.log.data)
        # Set splits
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        test = pd.DataFrame(test)
        train = pd.DataFrame(train)
        self._log_train = train
        self.log_test = test
        # print('ee', self.log_test)
        # self.log_test = (test.sort_values(key, ascending=True)
        #                  .reset_index(drop=True))
        # self.log_test = copy.deepcopy(test)
        print('Number of instances in test log: {}'.format(len(self.log_test['caseid'].drop_duplicates())))
        self.log_train = copy.deepcopy(self.log)
        
        
        self.log_train.set_data(train.to_dict('records'))
        print('Number of instances in train log: {}'.format(len(train.sort_values(key, ascending=True)
                                .reset_index(drop=True)['caseid'].drop_duplicates())))

    @staticmethod
    def _get_traces(data, one_timestamp):
        """
        returns the data splitted by caseid and ordered by start_timestamp
        """
        cases = list(set([x['caseid'] for x in data]))
        traces = list()
        for case in cases:
            order_key = 'end_timestamp' if one_timestamp else 'start_timestamp'
            trace = sorted(
                list(filter(lambda x: (x['caseid'] == case), data)),
                key=itemgetter(order_key))
            traces.append(trace)
        return traces
