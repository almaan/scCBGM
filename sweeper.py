import argparse as arp
import uuid
import subprocess
import os.path as osp
import os


if __name__ == "__main__":

    os.environ["PROJECT_ROOT"] = os.getcwd()

    # Create the main parser
    parser = arp.ArgumentParser(description="Main parser with LSF subparser.")

    parser.add_argument("--sweep_id", type=str, required=True, help="ID of the sweep.")

    parser.add_argument(
        "--n_jobs", type=int, default=1, required=False, help="Number of jobs to submit"
    )

    parser.add_argument(
        "--max_count",
        default=None,
        required=False,
        help="Max counts per agent",
    )

    # Create subparsers and define the 'lsf' subparser
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    lsf_parser = subparsers.add_parser("lsf", help="LSF job configuration")

    slurm_parser = subparsers.add_parser("slurm", help="Slurm job configuration")

    # Add arguments to the 'lsf' subparser
    lsf_parser.add_argument(
        "--memory",
        type=str,
        default="25GB",
        required=False,
        help="Memory required (in GB).",
    )
    lsf_parser.add_argument(
        "--num_cores",
        type=str,
        default="8",
        required=False,
        help="Number of CPU cores required.",
    )
    lsf_parser.add_argument(
        "--sla", type=str, default=None, help="Service Level Agreement (SLA) name."
    )

    lsf_parser.add_argument("--queue", type=str, default="short", help="Queue")

    lsf_parser.add_argument("--job_name", type=str, default=None, help="Shared jobname")

    lsf_parser.add_argument(
        "--out_dir", type=str, default="outputs/jobs", help="Output directory"
    )

    # ---- SLURM ----

    slurm_parser.add_argument(
        "--partition",
        type=str,
        required=True,
        help="Slurm_Parser.partition (e.g., gpu, short, long).",
    )
    slurm_parser.add_argument(
        "--account",
        type=str,
        required=True,
        help="Slurm_Parser.account name (e.g., owner_gred_braid_gpu).",
    )
    slurm_parser.add_argument(
        "--mem_per_cpu",
        type=str,
        default="10GB",
        help="Memory per node (Slurm_Parser.`--mem`).",
    )
    slurm_parser.add_argument(
        "--ntasks",
        type=int,
        default=8,
    )
    slurm_parser.add_argument(
        "--qos",
        type=str,
        default=None,
    )

    slurm_parser.add_argument(
        "--cpus_per_task",
        type=int,
        default=6,
    )

    slurm_parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="GPUs per job (Slurm_Parser.`--gres=gpu:<N>`).",
    )
    slurm_parser.add_argument(
        "--time",
        type=str,
        default="10:00:00",
        help="Walltime limit (Slurm_Parser.`--time`), e.g. 2:00:00",
    )
    slurm_parser.add_argument(
        "--job_name",
        type=str,
        default=None,
        help="Shared job name (Slurm_Parser.`--job-name`).",
    )
    slurm_parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/jobs",
        help="Directory for stdout/stderr files.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Example: Access parsed arguments

    if args.command == "lsf":

        memory = args.memory if args.memory.endswith("B") else args.memory + "GB"

        cmd = [
            "bsub",
            "-q",
            args.queue,
            "-oo",
            osp.join(args.out_dir, "stdout.{identifier}"),
            "-eo",
            osp.join(args.out_dir, "stderr.{identifier}"),
            "-M",
            memory,
            "-n",
            args.num_cores,
            "-gpu",
            "num=1:j_exclusive=yes",
        ]
        if args.sla is not None:
            cmd += [
                "-sla",
                args.sla,
            ]
        if args.job_name is not None:
            cmd += ["-J", args.job_name]

    elif args.command == "slurm":

        cmd = [
            "sbatch",
            "--partition",
            args.partition,
            "--account",
            args.account,
            "--mem-per-cpu",
            args.mem_per_cpu,
            "--ntasks",
            str(args.ntasks),
            "--cpus-per-task",
            str(args.cpus_per_task),
            "--gpus",
            str(args.gpus),
            "--output",
            osp.join(args.out_dir, "stdout.{identifier}.out"),
            "--error",
            osp.join(args.out_dir, "stderr.{identifier}.err"),
        ]

        if args.job_name is not None:
            cmd += ["-J", args.job_name]

        if args.qos is not None:
            cmd += ["--qos", args.qos]

    cmd += ["wandb", "agent", args.sweep_id]
    if args.max_count is not None:
        cmd += ["--count", args.max_count]

    for ii in range(args.n_jobs):
        identifier = uuid.uuid4()
        cmd_unique = [x.replace("{identifier}", str(identifier)) for x in cmd]

        print(cmd_unique)

        result = subprocess.run(cmd_unique, capture_output=True, text=True)
        print(result)
