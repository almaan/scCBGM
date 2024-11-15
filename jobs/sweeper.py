import argparse as arp
import uuid
import subprocess
import os.path as osp


if __name__ == "__main__":

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

    cmd += ["wandb", "agent", args.sweep_id]
    if args.max_count is not None:
        cmd += ["--count", args.max_count]

    for ii in range(args.n_jobs):
        identifier = uuid.uuid4()
        cmd_unique = [x.replace("{identifier}", str(identifier)) for x in cmd]

        result = subprocess.run(cmd_unique, capture_output=True, text=True)
