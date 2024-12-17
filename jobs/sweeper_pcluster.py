import argparse as arp
import uuid
import subprocess
import os
import os.path as osp

JOB_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={stdout_file}
#SBATCH --error={stderr_file}
#SBATCH --partition={partition}
#SBATCH --gpus=1
#SBATCH --cpus-per-task={num_cores}
#SBATCH --mem={memory}
#SBATCH --ntasks=1
#SBATCH --nodes=1

wandb agent {sweep_id}
"""

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

    # Ensure the output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Example: Access parsed arguments

    if args.command == "lsf":

        # Iterate over the number of jobs to create and submit job scripts
        for ii in range(args.n_jobs):
            identifier = uuid.uuid4()
            job_name = args.job_name if args.job_name else f"job_{identifier}"
            stdout_file = osp.join(args.out_dir, f"stdout.{identifier}")
            stderr_file = osp.join(args.out_dir, f"stderr.{identifier}")

            # Create job script content
            job_script_content = JOB_SCRIPT_TEMPLATE.format(
                job_name=job_name,
                stdout_file=stdout_file,
                stderr_file=stderr_file,
                partition=args.sla,
                num_cores=args.num_cores,
                memory=args.memory,
                sweep_id=args.sweep_id,
            )

            # Write job script to a temporary file
            job_script_path = osp.join(args.out_dir, f"job_script.{identifier}.sh")
            with open(job_script_path, "w") as job_script_file:
                job_script_file.write(job_script_content)

            # Submit the job script using sbatch
            cmd = ["sbatch", job_script_path]
            print("Running command:", " ".join(cmd))  # Print the command for debugging
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Print the result for debugging
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
            print("returncode:", result.returncode)

            # Optionally, remove the job script file after submission
            os.remove(job_script_path)
