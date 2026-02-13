## 📊 Reproducing Results

We’ve aimed to make our analysis **fully reproducible**:

- Fixed results (e.g., sweeps) are located in `results/`
- Analysis notebooks are available in `notebooks/analysis/`
  - File names are self-explanatory and aligned with their respective analyses
- Sweep configs for re-running experiments can be found in `sweeps/` (see instructions below)

## 📂 Data Access

Data is available at: [OSF link](https://osf.io/kfqj8/?view_only=02cfaddc86da47d5b8fca0577628ddf7)

To run existing scripts and notebooks, please:
- Extract `real.zip` into `data/real`
- Extract `synthetic.zip` into `data/synthetic`



## 🔄 Making Sweeps

### 🧪 Real Data Benchmark
See detailed instructions here: [Benchmarking Sweeps](scBenchmark.md).

### 🧪 Synthetic Data with Noisy Concepts
We use config files in `synth_hydra_config` for this.

Our code supports sweeps using a combination of **wandb** and **hydra**.
Official `wandb` documentation can be found [here](https://wandb.ai/adrishd/hydra-example/reports/Configuring-W-B-Projects-with-Hydra--VmlldzoxNTA2MzQw), though we also provide a more minimal and focused guide below.


**1️⃣ Setting up a sweep config**
Prepare a config file compatible with `wandb sweep`.
A demo is available at `sweeps/sweep_demo.yaml`.


**2️⃣ Creating a sweep agent**
Run the following in the repo root:

```sh
wandb sweep sweeps/sweep_demo.yaml
```

This will generate a sweep ID and output something like:

```sh
$ wandb: Creating sweep from: sweeps/sweep_demo.yaml
$ wandb: Creating sweep with ID: wuw4xyb0
$ wandb: View sweep at: https://xxx.wandb.io/user/conceptlab
$ wandb: Run sweep agent with: wandb agent user/conceptlab/wuw4xyb0
```


**3️⃣ Launching the sweep**
To launch, simply run:

```sh
wandb agent user/conceptlab/wuw4xyb0
```

This must be executed on a GPU session.

For large sweeps, you may want to distribute runs across multiple GPUs. `wandb agents` make this easy—multiple jobs can connect to the same agent.

We also provide a CLI tool to automate launching sweep agents via schedulers (currently **LSF**, with **SLURM** support possible on request).

Run it from the repo root:

```sh
python jobs/sweeper.py --sweep_id SWEEP_AGENT --num_jobs NUM_JOBS lsf
```

Where:
- `SWEEP_AGENT` = sweep ID from `wandb sweep`
- `NUM_JOBS` = number of GPU jobs to distribute across

Example (10 jobs):

```sh
python jobs/sweeper.py --sweep_id user/conceptlab/wuw4xyb0 --num_jobs 10 lsf
```


⚙️ **Scheduler Options**

We support both **LSF** and **SLURM**. For LSF, extra options are available:

```sh
$ python jobs/sweeper.py lsf --help
usage: sweeper.py lsf [-h] [--memory MEMORY] [--num_cores NUM_CORES] [--sla SLA] [--queue QUEUE] [--job_name JOB_NAME] [--out_dir OUT_DIR]

options:
  -h, --help            Show this help message and exit
  --memory MEMORY       Memory required (in GB)
  --num_cores NUM_CORES Number of CPU cores required
  --sla SLA             Service Level Agreement (SLA) name
  --queue QUEUE         Queue name
  --job_name JOB_NAME   Shared job name
  --out_dir OUT_DIR     Output directory
```

Example run:

```sh
python jobs/sweeper.py --num_jobs 20 --sweep_id user/conceptlab/wuw4xyb0 lsf --sla gpu_gold --job_name sweep
```
