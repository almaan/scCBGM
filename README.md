# ConceptLab

This is a toy setup for OMICS data to test different architecture


# Getting started

Install the environment

```
mamba env create -f env.yml
conda activate conceptlab
```

## Running code:

Running experiments using hydra configs

```
python main.py experiment=vae.yaml
```

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
$ wandb: View sweep at: https://genentech.wandb.io/andera29/conceptlab/s
$ wandb: Run sweep agent with: wandb agent andera29/conceptlab/wuw4xyb0

```

### 3. Launch the sweep

To launch the sweep you can simply do `wandb agent andera29/conceptlab/wuw4xyb0`
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
python jobs/sweeper.py --n_jobs 50 --sweep_id andera29/conceptlab/2wfnbcy8 lsf --sla gRED_braid_gpu --job_name sweep

```
