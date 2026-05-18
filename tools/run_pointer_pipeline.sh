#!/bin/bash
# Full pipeline: step1 (imitation) + step2 (PPO) with Pointer Attention scoring
set -u
GOODRL=/home/xue/GOODRL
PY=/home/xue/anaconda3/envs/drl_sched/bin/python
RUN_DIR=$GOODRL/runs/20260516_pointer_full
mkdir -p $RUN_DIR/checkpoints

echo "[$(date)] === Pointer pipeline started ===" | tee -a $RUN_DIR/pipeline.log

# Restore the GATv2 baseline step1 actor (in case actor_attn run polluted it)
if [ -f $GOODRL/validation_data/step1/actors/a_5_5_0.0015.pth.bak_pre_actor_attn ]; then
    cp $GOODRL/validation_data/step1/actors/a_5_5_0.0015.pth.bak_pre_actor_attn \
       $GOODRL/validation_data/step1/actors/a_5_5_0.0015.pth
    echo "[$(date)] Restored GATv2 baseline step1 actor" | tee -a $RUN_DIR/pipeline.log
fi

rm -f $GOODRL/logs/a_5_5_0.0015.pth $GOODRL/logs/c_5_5_0.0015.pth

# === Step 1: imitation with Pointer head ===
echo "[$(date)] === Step1 with --actor_pointer 1 ===" | tee -a $RUN_DIR/pipeline.log
cd $GOODRL
$PY -u step1.py \
    --vm_types 5 --each_vm_type_num 5 --arr_rate 5.4 \
    --lr_a 0.0001 --log_interval 1 --max_updates 10 \
    --algo_seed 3 \
    --actor_pointer 1 \
    --actor_atten_layers 0 \
    > $RUN_DIR/step1.log 2>&1

STEP1_EXIT=$?
echo "[$(date)] Step1 exit code: $STEP1_EXIT" | tee -a $RUN_DIR/pipeline.log

if [ "$STEP1_EXIT" != "0" ] || [ ! -f $GOODRL/logs/a_5_5_0.0015.pth ]; then
    echo "[$(date)] Step1 FAILED - aborting" | tee -a $RUN_DIR/pipeline.log
    exit 1
fi

cp $GOODRL/logs/a_5_5_0.0015.pth $GOODRL/validation_data/step1/actors/a_5_5_0.0015.pth
cp $GOODRL/logs/a_5_5_0.0015.pth $RUN_DIR/step1_actor.pth

LAST_STEP1_VAL=$(grep -a "Vlidation" $RUN_DIR/step1.log | tail -1)
echo "[$(date)] Step1 last validation: $LAST_STEP1_VAL" | tee -a $RUN_DIR/pipeline.log

cat > $RUN_DIR/metadata.json <<JSON
{
  "run_id": "20260516_pointer_full",
  "description": "Full pipeline with Pointer Attention scoring head. Previous SelfAttention probe failed (broke imitation). Pointer should preserve per-candidate identity.",
  "architecture": "GATv2 + Pointer Attention scoring",
  "non_default_flags": {
    "actor_pointer": 1,
    "actor_atten_layers": 0,
    "lr_a": 0.0001, "lr_c": 0.001,
    "warmup_critic": 500, "max_updates": 1000,
    "grad_control": 0, "algo_seed": 3, "entloss_coef": 0.01
  },
  "step1_record": "$LAST_STEP1_VAL",
  "started_at": "$(date -Iseconds)"
}
JSON

# === Step 2 with watchdog ===
echo "[$(date)] === Step2 with watchdog ===" | tee -a $RUN_DIR/pipeline.log
bash $GOODRL/tools/run_with_resume.sh $RUN_DIR \
    --vm_types 5 --each_vm_type_num 5 --arr_rate 5.4 \
    --lr_a 0.0001 --lr_c 0.001 \
    --warmup_critic 500 --max_updates 1000 \
    --grad_control 0 --algo_seed 3 --entloss_coef 0.01 \
    --actor_pointer 1 \
    --actor_atten_layers 0 \
    --checkpoint_interval 50

echo "[$(date)] === Pipeline DONE ===" | tee -a $RUN_DIR/pipeline.log
