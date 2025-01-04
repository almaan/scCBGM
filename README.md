# ConceptLab

Code for accompanying the paper ''Single Cell Concept Bottleneck Generative Model'' submitted to ICML 2025.


# Getting started

Setup the environment

```
micromamba env create -f env.yaml
micromamba activate conceptlab
```

Install the package
```
poetry install
```

## Running code:

Running experiments using hydra configs

```
python main.py experiment=demo.yaml
```

## Reproducing Results
We've aimed to make our analysis as reproducible as possible. Hence:

* All results from our wandb sweeps can be found in: `results/sweeps/`
* We provide notebooks to generate all display items and analysis `notebooks/analysis/`. These are as follows:
  * `hbca_immune_continous_state_shift_analysis.ipynb` - reproduces Figure 2
  * `hbca_immune_pathway_scoring.ipynb` - preprocess and assigns pathway scores to the HBCA-I dataset.
  * `kang_rmmd_analysis_processing.ipynb` - reproduces Table 3
  * `kang_zero_shot_CD4T_cell_example.ipynb` - reproduces Figure 1
  * `optimal_hyperparameter_selection.ipynb` - outlines the choice of each set of hyper parameters for each model.
  * `synthetic_data_result_processing.ipynb` - reproduces Table 1 and Table 2.
* Sweep configs to re-run all of the sweeps can be found in `sweeps/`, we also include instructions on how to use these below.
* The weights for the specific models used in the examples shown in Figure 1 and 2 are found in `results/model_params/`.


## Making Sweeps

The code supports sweeps with a combination of `wandb` and `hydra`. Official
documentation from `wandb` can be found
[here](https://wandb.ai/adrishd/hydra-example/reports/Configuring-W-B-Projects-with-Hydra--VmlldzoxNTA2MzQw);
however, we support a more minimal but also more specific set of instructions
here.

### 1. Setting up a sweep config
You need to set up a config file that's compatible with the `wandb sweep`
function. A demo can be found in `sweeps/sweep_demo.yaml`.

### 2. Create a sweep agent
Create a sweep agent by running `wandb sweep sweep_file`; unless you want to
change multiple files, do this in the `root` directory of this repo.

Using the example this command would look like `wandb sweep sweeps/sweep_demo.yaml`.

When you create a sweep agent, a message that looks something like below, will be displayed:

```sh

$ wandb: Creating sweep from: sweeps/sweep_demo.yaml
$ wandb: Creating sweep with ID: wuw4xyb0
$ wandb: View sweep at: https://xxx.wandb.io/user/conceptlab
$ wandb: Run sweep agent with: wandb agent user/conceptlab/wuw4xyb0

```

### 3. Launch the sweep

To launch the sweep you can simply do `wandb agent user/conceptlab/wuw4xyb0`
as the statement above suggests. This needs to be done on a GPU session.

If you have a large sweep with multiple configurations, it might be smarter to
distribute this across multiple GPUs. This is where the beauty of `wandb agents`
come into play, you can launch multiple jobs that all connects to the same
agent. To automate this we also provide a CLI application that is can launch the
sweep agents via a scheduler. Currently we only have support for LSF, but this
can be extended to other schedulers like SLURM on request.

To use the script do (in `root`)


```sh
# in root
python jobs/sweeper.py --sweep_id SWEEP_AGENT --num_jobs NUM_JOBS lsf

```

Where `SWEEP_AGENT` is the sweep command you get from running `wandb sweep` and
`NUM_JOBS` is how many GPU jobs you want to submit (the sweep will be spread out
over them). Using the example above with 10 jobs this would be:



The lsf subparser have multiple additional commands that you can use to specify the SLA, number of cores, name of the job etc.

```
$ python jobs/sweeper.py lsf --help
usage: sweeper.py lsf [-h] [--memory MEMORY] [--num_cores NUM_CORES] [--sla SLA] [--queue QUEUE] [--job_name JOB_NAME] [--out_dir OUT_DIR]

options:
  -h, --help            show this help message and exit
  --memory MEMORY       Memory required (in GB).
  --num_cores NUM_CORES
                        Number of CPU cores required.
  --sla SLA             Service Level Agreement (SLA) name.
  --queue QUEUE         Queue
  --job_name JOB_NAME   Shared jobname
  --out_dir OUT_DIR     Output directory
```

One example of how you could run this is:


```sh
# in root
python jobs/sweeper.py --n_jobs 20 --sweep_id user/conceptlab/wuw4xyb0 lsf --sla gpu_gold --job_name sweep

```
