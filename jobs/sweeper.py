import argparse as arp
import uuid
import subprocess


if __name__ == "__main__":

    # Create the main parser
    parser = arp.ArgumentParser(description="Main parser with LSF subparser.")

    parser.add_argument("--sweep_id", type=str, required=True, help="ID of the sweep.")

    parser.add_argument(
        "--n_jobs", type=int, default=1, required=False, help="Number of jobs to submit"
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

    lsf_parser.add_argument(
        "--out_dir", type=str, default="outputs/jobs", help="Output directory"
    )

    # Parse the arguments
    args = parser.parse_args()

    identifier = uuid.uuid4()

    # Example: Access parsed arguments

    if args.command == "lsf":

        memory = args.memory if args.memory.endswith("B") else args.memory + "GB"

        cmd = [
            "bsub",
            "-q",
            args.queue,
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
        cmd += ["wandb", "agent", args.sweep_id]

        for ii in range(args.n_jobs):
            result = subprocess.run(cmd, capture_output=True, text=True)
            # Print the output
            print(result.stdout)
