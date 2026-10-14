# Knowledge Tracing Set Transformers (KTSTs)

Code for [_Principled Transformers for Predictive Performance in Knowledge Tracing_](https://jedm.educationaldatamining.org/index.php/JEDM/article/download/961/272) (accepted in the Journal of Educational Data Mining (JEDM)).

Kai Neubauer*, Yannick Rudolph*, and Ulf Brefeld (* equal contribution)

> Knowledge tracing aims to model students' knowledge and abilities over time, which is crucial for intelligent tutoring systems. In this paper, we propose a straightforward model class, knowledge tracing set transformers (KTSTs), specifically addressing predictive performance in knowledge tracing tasks. KTSTs closely follow prominent transformer architectures and use an intuitive set-based representation for student interactions. We introduce learnable ALiBi, which simplifies and improves upon a prevalent attention mechanism in knowledge tracing, and MHSA aggregation, which readily allows incorporating an arbitrary number of additional, potentially more complex features per student interaction. We highlight and discuss flaws present in related approaches, which are overly complex and, in part, based on suboptimal design choices. We validate our design choices for KTSTs in experiments with real-world data and simulated learning sequences. Overall, we address lessons learned and propose a straightforward model that relies on best practices and establishes a new state-of-the-art on standardized benchmark datasets. Ultimately, KTSTs may serve as a simple but effective base model class for future research in knowledge tracing and intelligent tutoring systems.

<!-- Cite this work:
```bibtex
@inproceedings{}
``` -->

## Installation

We use [conda](https://pip.pypa.io/en/stable/) to manage dependencies. 
To create a new conda environment with the necessary dependencies installed, execute the following command:

```bash
conda env create --file env.yaml
```

By default, this will create a conda environment named `knowledge_tracing`. To activate this environment, use:

```bash
conda activate knowledge_tracing
```

## Usage

### Pre-process data

To run the experiments, the datasets must first be downloaded and then pre-processed. 
The code for pre-processing is stored in `preprocess.py`, which has a CLI and internally uses the [pyKT](https://github.com/pykt-team/pykt-toolkit) data pre-processing logic. 
Run the following command for more information about the pre-processing of the data, e.g. the links to the datasets:

```bash
python preprocess.py --help
```

### Train a model
We use [hydra](https://hydra.cc/) to manage {model, data, training, evaluation, reproduction, ...}-configurations. 
All configs are stored in `pkg/config`. 

In order to train a single KTST model with its default config on the assist2009 dataset, execute the following command:

```bash
python train.py +model=ktst +data=assist2009 data_path=<path_to_data_dir>
```

This will create a folder in `outputs` in which all relevant files (hydra configs, model checkpoints, tensorboard files, logs, etc.) are stored. 

### Evaluate a model

The training procedure will log and store model states into a folder called `outputs`. 
Use the path(s) (recursive!) to run evaluation on trained models.

```bash
python eval.py restore_from=<name_of_dir_in_output_folder>
```

### Reproduce results from paper

Configuration files for reproducing the results are stored under `pkg/config/reproduce`.
Each sub-directory has a multirun.yaml that starts all training runs: 

```bash
python train.py -m +reproduce=<path-to-multirun-file>
```

For example, to train the benchmark models run the following command:

```bash
python train.py -m +reproduce=ktst/benchmark/multirun
```

Afterwards, it is necessary to run evaluation for all trained models (see above).

Notebooks containing the code for creating tables and plots can be found under `scripts`. 

## Run pyKT benchmark

We have added a convenient way to run the [pyKT](https://pykt.org/) benchmark and compute its results (for example, to conduct paired _t_-tests) on a GPU cluster: 
The script `pykt_benchmark.py` has a CLI with which both training and evaluation of models in pyKT can be started. 
For more information on the CLI, run the following command: 

```bash
python pykt_benchmark.py --help
```

### (Important) Linking data folder

The computations of the pykt-toolkit depend on a relative reference to the `data` folder. 
To do this, the `data` folder must be linked to the `pykt-toolkit` subfolder via a soft link. 

```bash
ln -s <link_to_data_dir> <link_to_this_repo>/pykt-toolkit/data
```

### Training models

To train a particular model on specific datasets, these can be defined as comma-separated values.
For example, to train the AKT and DKT models on assist2009, the following command can be used:

```bash
python pykt_benchmark.py train --models akt,dkt --datasets assist2009
```

If no specific models or datasets are defined, all available models or datasets are trained. 
If not specified otherwise, training results are stored in `outputs/pykt-models`, organized by model, dataset and fold id. 
The file `done` in the corresponding folder of a run indicates whether the training was completed successfully. 

### Evaluating models

The evaluation can be executed by changing the mode from `train` to `eval`.
For this to work, the models must of course have been trained beforehand and the `done` file needs to be found in the respective directory.

```bash
python pykt_benchmark.py eval --models akt,dkt --datasets assist2009
```

## Misc

### Sweep model parameters

We provide a script with which with you can tune model parameters with [Optuna](https://hydra.cc/docs/plugins/optuna_sweeper/).
You can retrieve information on available parameters via the CLI.

```bash
python sweep.py --help
```

A config defines the model and its corresponding parameter search space. 
Example:

```bash
python sweep.py sweep/ktst=benchmark_mean --datasets statics2011 --folds 0
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
