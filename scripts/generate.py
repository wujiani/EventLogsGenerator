import subprocess
from utils.support import timeit
import sys
import click

# N = 10
# FOLDER = "PurchasingExample_0.2"
# CATEGORY = "PurchasingExample"
# TRAIN_SIZE_GENERATION = "false"


@click.command()
@click.option('--n', required=True, type=int)
@click.option('--folder', required=True, type=str)
@click.option('--category', required=True, type=str)
@click.option('--train_size_generation', default=False, required=False, type=bool)
def main(n, folder, category, train_size_generation):
    # Loop through a range of indices
    print("################# Loop started ########################")
    for i in range(0, n):  # Replace 10 with the desired number of iterations
        print(f"Running {i} .....")
        command_seq = [
            "python",
            "scripts\\log_gen_seq.py",
            "--experiment-name", f"{folder}",
            "--import-file", f"example_datasets\\{folder}\\train_{category}.xes",
            "--import-test-file", f"example_datasets\\{folder}\\test_{category}.xes",
            "--id-column", "caseid",
            "--act-column", "task",
            "--time-column", "time:timestamp",
            "--resource-column", "user",
            "--state-column", "lifecycle:transition",
            "--method", "prefix",
            "--train_size_generation", train_size_generation,
            "--suffix", str(i)  # Pass the current loop index as a string
        ]

        @timeit
        def seq_script():
            subprocess.run(command_seq)

        # Run the command
        seq_script()


        command_time = [
            "python",
            "scripts\\log_gen_time.py",
            "--experiment-name", f"{folder}",
            "--import-file", f"example_datasets\\{folder}\\train_{category}.xes",
            "--import-test-file", f"example_datasets\\{folder}\\test_{category}.xes",
            "--file", f"{category}.xes",
            "--simulator-model-folder", f"example_datasets\\{folder}\\ia_gen_models",
            "--embedding-matrix", f"example_datasets\\{folder}\\embedded_matix\\ac_DP_{category}.emb",
            "--id-column", "caseid",
            "--act-column", "concept:name",
            "--time-column", "time:timestamp",
            "--resource-column", "user",
            "--state-column", "lifecycle:transition",
            "--suffix", str(i)  # Pass the current loop index as a string
        ]

        @timeit
        def time_script():
            subprocess.run(command_time)

        # Run the command
        time_script()

        command_res = [
            "python",
            "scripts\\log_gen_res.py",
            "--experiment-name", f"{folder}",
            "--import-file", f"example_datasets\\{folder}\\train_{category}.xes",
            "--id-column", "caseid",
            "--act-column", "task",
            "--time-column", "time:timestamp",
            "--resource-column", "user",
            "--state-column", "lifecycle:transition",
            "--suffix", str(i)
        ]

        @timeit
        def resource_script():
            subprocess.run(command_res)

        # Run the command
        resource_script()

    print("################# Loop finished ########################")

    print("################# start evaluation ########################")

    command_eval = [
        "python",
        "pipeline.py",
        "--file", f"{category}.xes",
        "--evaluation_only", "true",
        "--log_test_filename", f"example_datasets\\{folder}\\test_{category}.csv",
        "--gen_log_filename_no_extension", f"example_outputs\\{folder}\\gen_seq_time_res_train_{category}",
        "--gen_loop", f"{n}"
    ]

    @timeit
    def evaluation_script():
        subprocess.run(command_eval)

    # Run the command
    evaluation_script()


if __name__ == "__main__":
    main(sys.argv[1:])