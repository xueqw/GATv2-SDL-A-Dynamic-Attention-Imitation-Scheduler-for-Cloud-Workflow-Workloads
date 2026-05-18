#!/bin/bash
# GATv2 + HEFT teacher + direct PPO baseline pipeline
# Usage: chain_heft_baseline.sh <vm_types> <each_vm_type_num> <arr_rate>
set -u

VM_TYPES=$1
EACH_VM=$2
ARR_RATE=$3
GOODRL=/home/xue/GOODRL
PY=/home/xue/anaconda3/envs/drl_sched/bin/python

SCENARIO_TAG=${VM_TYPES}_${EACH_VM}_${ARR_RATE}
ARR_RATE_PER_SEC=$(awk -v ar=$ARR_RATE "BEGIN{printf \"%g\", ar/3600}")
RUN_DIR=$GOODRL/runs/20260518_gatv2_heft_baseline_${SCENARIO_TAG}
LOG=$RUN_DIR/pipeline.log
mkdir -p $RUN_DIR/checkpoints

echo "[$(date)] === GATv2+HEFT baseline: ${SCENARIO_TAG} (arr_per_sec=${ARR_RATE_PER_SEC}) ===" | tee -a $LOG

# Ensure HEFT memory accessible under per-sec name
HEFT_REAL=$GOODRL/validation_data/HEFT/HEFT_memory_500_${VM_TYPES}_${EACH_VM}_${ARR_RATE}.pkl
HEFT_LINK=$GOODRL/validation_data/HEFT/HEFT_memory_500_${VM_TYPES}_${EACH_VM}_${ARR_RATE_PER_SEC}.pkl
if [ ! -e "$HEFT_LINK" ]; then
    ln -s $(basename $HEFT_REAL) $HEFT_LINK
    echo "[$(date)] symlink: $(basename $HEFT_LINK) -> $(basename $HEFT_REAL)" | tee -a $LOG
fi

# Backup paper original actor if present and not already backed up
PAPER_ACTOR=$GOODRL/validation_data/step1/actors/a_${VM_TYPES}_${EACH_VM}_${ARR_RATE_PER_SEC}.pth
PAPER_BAK=$PAPER_ACTOR.bak_paper_GAT
if [ -f "$PAPER_ACTOR" ] && [ ! -f "$PAPER_BAK" ]; then
    cp $PAPER_ACTOR $PAPER_BAK
    echo "[$(date)] backed up: $PAPER_BAK" | tee -a $LOG
fi

# Phase 1: step1 HEFT imitation (GATv2 architecture)
LOGS_ACTOR=$GOODRL/logs/a_${VM_TYPES}_${EACH_VM}_${ARR_RATE_PER_SEC}.pth
rm -f $LOGS_ACTOR
echo "[$(date)] Phase 1: step1 HEFT imitation..." | tee -a $LOG
cd $GOODRL
$PY -u step1.py \
    --vm_types $VM_TYPES --each_vm_type_num $EACH_VM --arr_rate $ARR_RATE \
    --lr_a 0.0001 --log_interval 1 --max_updates 10 \
    --algo_seed 3 \
    --actor_pointer 0 --actor_atten_layers 0 \
    > $RUN_DIR/step1.log 2>&1
if [ ! -f "$LOGS_ACTOR" ]; then
    echo "[$(date)] step1 FAILED (no actor saved)" | tee -a $LOG
    tail -20 $RUN_DIR/step1.log | tee -a $LOG
    exit 2
fi
LAST_VAL=$(grep -a "Vlidation" $RUN_DIR/step1.log | tail -1)
echo "[$(date)] step1 done. $LAST_VAL" | tee -a $LOG

cp $LOGS_ACTOR $PAPER_ACTOR
cp $LOGS_ACTOR $RUN_DIR/step1_actor.pth

# Phase 2: step2 direct PPO (matches 20260515 baseline params, no KL anchor)
echo "[$(date)] Phase 2: step2 direct PPO..." | tee -a $LOG
bash $GOODRL/tools/run_with_resume.sh $RUN_DIR -- \
    --vm_types $VM_TYPES --each_vm_type_num $EACH_VM --arr_rate $ARR_RATE \
    --lr_a 0.0001 --lr_c 0.001 \
    --warmup_critic 500 --max_updates 1000 \
    --grad_control 0 --algo_seed 3 --entloss_coef 0.01 \
    --actor_pointer 0 --actor_atten_layers 0 \
    --checkpoint_interval 20

echo "[$(date)] === ${SCENARIO_TAG} DONE ===" | tee -a $LOG
