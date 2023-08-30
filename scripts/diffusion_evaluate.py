import pm4py
import os
import subprocess
import warnings
import click
import sys
import glob

from log_gen_time_utils import *
from utils.support import timeit
from core_modules.instances_generator.prophet_generator import ProphetGenerator

warnings.filterwarnings("ignore")


def transform_gen_data(output_folder, start_time, file, diff_output, simulator_model_folder, id_column, suffix):
    gen = pd.read_csv(diff_output)
    gen = gen.set_index('Unnamed: 0')
    gen['last'] = gen['last'].map(lambda x: [int(each) for each in x.strip('(').strip(')').split(',')] if x!='None' else x)
    gen['next'] = gen['next'].map(lambda x: [int(each) for each in x.strip('(').strip(')').split(',')] if x!='None' else x)
    num_instances = gen[id_column].nunique()
    print(f"num_instances: {num_instances}")

    ia_list = ProphetGenerator._generate(num_instances, start_time, simulator_model_folder, file)
    ia_list = [ia_list.iloc[i][0] for i in range(num_instances)]

    g = pd.DataFrame()
    for key, group in gen.groupby('caseid'):
        group = group.set_index('index')
        for i in range(len(group)):
            cur = group.iloc[i]
            if i == 0:
                group['start_timestamp'].iloc[i] = ia_list[int(key)-1]
                group['end_timestamp'].iloc[i] = group['start_timestamp'].iloc[i] + pd.Timedelta(seconds=0)
            else:
                if len(cur['last']) > 1:
                    last_time = max(pd.to_datetime(group['end_timestamp'].loc[list(cur['last'])]))
                    group['start_timestamp'].iloc[i] = last_time + pd.Timedelta(seconds=group['wait'].iloc[i])
                else:
                    last_time = pd.to_datetime(group['end_timestamp'].loc[cur['last'][0]])
                    group['start_timestamp'].iloc[i] = last_time + pd.Timedelta(seconds=group['wait'].iloc[i])
                group['end_timestamp'].iloc[i] = group['start_timestamp'].iloc[i] + pd.Timedelta(seconds=group['process'].iloc[i])
        g = pd.concat([g, group])
    g['caseid'] = g['caseid'].map(lambda x: 'Case'+str(x))
    g = g[['caseid', 'activity','start_timestamp', 'end_timestamp', 'res']]
    g = g.rename(columns = {'activity':'task', 'res':'resource'})

    final_output_filename = os.path.join(output_folder, f"diffusion_output_{suffix}.csv")
    g.to_csv(final_output_filename, index=False)


@click.command()
@click.option('--experiment-name', required=True, type=str)
@click.option('--import-test-file', required=True, type=str)
@click.option('--file', required=True, type=str)
@click.option('--diff_output_pattern', required=True, type=str)
@click.option('--simulator-model-folder', required=True, type=str)
@click.option('--id-column', default='caseid', type=str)
@click.option('--time-column', default='time:timestamp', type=str)
def main(experiment_name, import_test_file, file, diff_output_pattern, simulator_model_folder, id_column, time_column):

    output_folder = f'diffu_example_outputs\{experiment_name}'

    # get minimum date
    data_test = pm4py.convert_to_dataframe(pm4py.read.read_xes(import_test_file))
    start_time = data_test[time_column].min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

    for suffix, diff_output in enumerate(glob.glob(diff_output_pattern)):
        print(f"processing {diff_output} ...")
        transform_gen_data(output_folder, start_time, file, diff_output, simulator_model_folder, id_column, suffix)


    print("################# start evaluation ########################")

    command_eval = [
        "python",
        "pipeline.py",
        "--file", f"{file}",
        "--evaluation_only", "true",
        "--log_test_filename", f"{import_test_file.replace('xes', 'csv')}",
        "--gen_log_filename_no_extension", f"{os.path.join(os.path.dirname(diff_output_pattern), 'diffusion_output')}",
        "--gen_loop", f"{len(glob.glob(diff_output_pattern))}"
    ]

    @timeit
    def evaluation_script():
        subprocess.run(command_eval)

    # Run the command
    evaluation_script()

if __name__ == "__main__":
    main(sys.argv[1:])