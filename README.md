# Techniques for Discovering Event-Log Generative Models

#### ❗ NOTE: This repo is cloned from https://github.com/AdaptiveBProcess/DeepSimulator, I included fixes and other improvements. I also included scripts that implements a <u>probability based process logs generation algorithm</u> (details are explained in my thesis).  


DeepSimulator is a hybrid approach able to learn process simulation models from event logs wherein a (stochastic) process model is extracted via DDS techniques and then combined with a DL model to generate timestamped event sequences. This code can perform the next tasks:


* Training generative models using an event log as input.
* Generate full event logs using the trained generative models.
* Assess the similarity between the original log and the generated one.


### Prerequisites

To execute this code you just need to install Anaconda or Conda in your system, then execute 
* in windows: pip install -r req_windows.txt
* in linux with python version < 3.8: pip install -r requirement.txt
* in linux with python version >= 3.9: pip install -r req_py310.txt

## Running Deep Simulator

Once created the environment, you can execute the tool from a terminal specifying the input event log name and any of the following parameters:

* `--file (required)`: event log in XES format, the event log must be previously located in the `input_files/event_logs` folder
* `--update_gen/--no-update_gen (optional, default=False)`: Refers to whether you want to update the sequences generation model previously discovered. If this parameter is added, the entire discovery pipeline will be executed. Additionally, if this parameter is set, the number of repetitions of every experiment can be configured with the parameter `--s_gen_repetitions (optional, default=5)`, and the number of experiments with `--s_gen_max_eval (optional, default=30)`.
* `--update_ia_gen/--no-update_ia_gen (optional, default=False)`: Refers to whether you want to update the inter-arrivals generation model previously discovered. If this parameter is added, the entire discovery pipeline will be executed.
* `--update_times_gen/--no-update_times_gen (optional, default=False)`: Refers to whether you want to update the deep-learning times generation models previously discovered. If this parameter is added, the entire discovery pipeline will be executed. Additionally, if this parameter is set, the number of training epochs can be configured with the parameter `--t_gen_epochs (optional, default=200)`, and the number of experiments with `--t_gen_max_eval (optional, default=12)`.
* `--save_models/--no-save_models (optional, default=True)`: Refers to whether or not you want to save the discovered models.
* `--evaluate/--no-evaluate (optional, default=True)`: Refers to whether or not you want to perform a final assessment of the accuracy of the final simulation model.
* `--s_gen_max_eval, t_gen_epochs, t_gen_max_eval (optional, default values are given)`: parameters for models training (see details of models in my thesis)
* `--train_size_generation (optional, default='False')`: if true, generate training set size simulation event log, otherwise it uses test set size
* `--gen_loop (optional, default=1)`: how many event logs/files to generate.
* `--evaluation_only (optional, default=False), log_test_filename (optional), gen_log_filename_no_extension (optional)`: 
these are used together when one needs only evaluate a generic generated event log with a test event log, combined with gen_loop parameter one can evaluate several generated logs (see `scripts/generate.py`)

Note that this is a training & prediction pipeline consists of many steps in each of them model artifacts or intermediate results are saved in a specific folder (input_files), this can save time if one only want
to generate event traces without retraining models. One have to respect the given folder hierarchy, some program variables/behaviours can
be changed in either in `properties.yml` or by using script launching arguments.

**Example of basic execution:**

`
python .\pipeline.py --file Production.xes --gen_loop 5
`
`
python .\pipeline.py --file PurchasingExample.xes --gen_loop 5
`

this runs training and prediction given a xes file under `input_files/event_logs`
and puts into an output folder (`output_files/Production_{%Y%m%d}`) all intermediate and final results.

**Example of execution updating the deep-learning times generation models**

`
C:\DeepSimulator> python .\pipeline.py --file Production.xes --update_times_gen --t_gen_epochs 20 --t_gen_max_eval 3
`
---
## Running probabilistic algorithm:

First, run the **DeepSimulator** code to get the baseline generated event log, as well as the embedding matrix of the activities.
Since we use the same generated arrival times for each trace as the DeepSimulator and the embedding matrix to compute the nearest activities, we need the output of the baseline model. 
The algorithm is implemented with some python scripts under the folder `scripts` and these scripts take files from `example_datasets` and save results in `example_outputs`.

(In the directory `example_datasets`, we already have the outputs of DeepSimulator with some datasets.)

✅ firstly, generate the sequence of trace activities running `scripts/log_gen_seq.py`, arguments are:
- `--experiment-name`: used for creating output folder
- `--import-file`: full path of training dataset with format xes
- `--import-test-file`: full path of test dataset with format xes
- `--id-column` (optional): id tag name
- `--act-column` (optional): activity tag name
- `--time-column` (optional): timestamp tag name
- `--resource-column` (optional): resource tag name
- `--state-column` (optional): state tag name
- `--method` (optional): n_gram, prefix.
- `--num` (optional): n_gram number.
- `--suffix` (optional): output file name integer suffix
- `--train_size_generation` (optional): if true, generate training set size simulation event log, otherwise it uses test set size

execution example:

`python scripts\log_gen_seq.py --experiment-name "Production_0.2" --import-file "example_datasets\Production_0.2\train_Production.xes" --import-test-file "example_datasets\Production_0.2\test_Production.xes" --id-column caseid --act-column "task" --time-column "time:timestamp" --resource-column user --state-column "lifecycle:transition" --method prefix --suffix 0`

the output file is `example_outputs/Production_0.2/gen_seq_train_Production_0.csv`

✅ Then, generate the time of each activity, both start time and end time, running `scripts/log_gen_time.py`. In this step the prophet model trained above is used.

following are additional arguments w.r.t. the first script:
- `--file`: original event log file name with format xes.
- `--simulator-model-folder`: the ia_gen_models folder containing prophet model artifacts produced in deep simulator
- `--embedding-matrix`: full path of embedding matrix calculated in deep simulator, with format emb.

execution example:

`python scripts\log_gen_time.py --experiment-name "Production_0.2" --import-file "example_datasets\Production_0.2\train_Production.xes" --import-test-file "example_datasets\Production_0.2\test_Production.xes" --file "Production.xes" --simulator-model-folder "example_datasets\Production_0.2\ia_gen_models" --embedding-matrix "example_datasets\Production_0.2\embedded_matix\ac_DP_Production.emb" --id-column caseid --act-column "concept:name" --time-column "time:timestamp" --resource-column user --state-column "lifecycle:transition" --suffix 0`

the output file is `example_outputs/Production_0.2/gen_seq_time_train_Production_0.csv`

✅  Finally, generate the resource of each activity running `scripts/log_gen_res.py`. The script arguments are already explained.

execution example:

`python scripts\log_gen_res.py --experiment-name "Production_0.2" --import-file "example_datasets\Production_0.2\train_Production.xes" --id-column caseid --act-column "task" --time-column "time:timestamp" --resource-column user --state-column "lifecycle:transition" --suffix 0`

the output file is `example_outputs/Production_0.2/gen_seq_time_res_train_Production_0.csv`

📣 For ease of use and in order to evaluate the outputs using the same evaluation function as in deep simulator, we created another script `scripts/generate.py`
in which above three steps will be done sequentially with a final evaluation step.

execution example:

`python scripts/generate --n 10 --folder "PurchasingExample_0.2" --category "PurchasingExample" --train_size_generation false`

---
# Diffusion model outputs evaluation

In my thesis I showed a novel diffusion probabilistic model able to generate event logs, the implementation code is in this repo https://github.com/wujiani/CoDi.git,
here I created `scripts/diffusion_evaluate.py` in which I exploit the trained prophet model to generate start timestamp, then a data transformation is done to have a standard event log csv format. Finally, the same evaluation function of deep simulator above is performed to calculate evaluation metrics.

Example execution:

`python scripts\diffusion_evaluate.py --experiment-name "PurchasingExample_0.2" --import-test-file "example_datasets\PurchasingExample_0.2\test_PurchasingExample.xes" --file "PurchasingExample.xes" --diff_output_pattern "diffu_example_outputs\\PurchasingExample_0.2\\gen_sample_*.csv" --simulator-model-folder "example_datasets\PurchasingExample_0.2\ia_gen_models" --id-column caseid --time-column "time:timestamp"`

One needs to give well known arguments described above but also a full path pattern of filenames (diff_output_pattern), this script iterates all files that match this pattern.