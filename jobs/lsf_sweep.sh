
for ii in $( seq 1 $2 ); do
    bsub -q short -oo outputs/jobs/output.$ii -eo outputs/jobs/errors.$ii -n 8 -M 25GB  -gpu num=1:j_exclusive=yes -sla gRED_braid_gpu wandb agent $1
done
