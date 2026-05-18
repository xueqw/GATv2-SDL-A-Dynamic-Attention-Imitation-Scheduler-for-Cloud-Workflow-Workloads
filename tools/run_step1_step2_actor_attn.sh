#!/bin/bash
# Full pipeline: step1 (imitation) + step2 (PPO) with --actor_atten_layers 1
# step2 uses watchdog for crash resilience
set -u
GOODRL=/home/xue/GOODRL
PY=/home/xue/anaconda3/envs/drl_sched/bin/python
RUN_DIR=$GOODRL/runs/20260516_actor_attn_full
mkdir -p $RUN_DIR/checkpoints

echo "[$(date)] === Pipeline started ===" | tee -a $RUN_DIR/pipeline.log

# Backup current step1 actor (we'll replace it)
BACKUP=$GOODRL/validation_data/step1/actors/a_5_5_0.0015.pth.bak_pre_actor_attn
if [ ! -f "$BACKUP" ]; then
    cp $GOODRL/validation_data/step1/actors/a_5_5_0.0015.pth $BACKUP
    echo "[$(date)] Backed up old step1 actor to $BACKUP" | tee -a $RUN_DIR/pipeline.log
fi

# Clear stale logs/ saves
rm -f $GOODRL/logs/a_5_5_0.0015.pth $GOODRL/logs/c_5_5_0.0015.pth

# === Step 1 ===
echo "[$(date)] === Running Step1 with --actor_atten_layers 1 ===" | tee -a $RUN_DIR/pipeline.log
cd $GOODRL
$PY -u step1.py \
    --vm_types 5 --each_vm_type_num 5 --arr_rate 5.4 \
    --lr_a 0.0001 --log_interval 1 --max_updates 10 \
    --algo_seed 3 \
    --actor_atten_layers 1 \
    > $RUN_DIR/step1.log 2>&1

STEP1_EXIT=$?
echo "[$(date)] Step1 exit code: $STEP1_EXIT" | tee -a $RUN_DIR/pipeline.log

if [ "$STEP1_EXIT" != "0" ]; then
    echo "[$(date)] Step1 FAILED - check $RUN_DIR/step1.log" | tee -a $RUN_DIR/pipeline.log
    exit 1
fi

if [ ! -f $GOODRL/logs/a_5_5_0.0015.pth ]; then
    echo "[$(date)] WARNING: step1 did not save actor!" | tee -a $RUN_DIR/pipeline.log
    exit 2
fi

# Promote step1 output to validation_data/step1/actors/
cp $GOODRL/logs/a_5_5_0.0015.pth $GOODRL/validation_data/step1/actors/a_5_5_0.0015.pth
cp $GOODRL/logs/a_5_5_0.0015.pth $RUN_DIR/step1_actor.pth
echo "[$(date)] Promoted step1 actor to validation_data/" | tee -a $RUN_DIR/pipeline.log

# Show step1 result
LAST_STEP1_VAL=$(grep -a "Vlidation" $RUN_DIR/step1.log | tail -1)
echo "[$(date)] Step1 last validation: $LAST_STEP1_VAL" | tee -a $RUN_DIR/pipeline.log

# Write metadata
cat > $RUN_DIR/metadata.json <<JSON
{
  "run_id": "20260516_actor_attn_full",
  "description": "Full pipeline (step1 + step2) with actor self-attention (--actor_atten_layers=1). Probe for whether attention in actor helps before committing to Pointer attention.",
  "architecture": "GATv2 + Actor SelfAttention",
  "non_default_flags": {
    "actor_atten_layers": 1,
    "lr_a": 0.0001, "lr_c": 0.001,
    "warmup_critic": 500, "max_updates": 1000,
    "grad_control": 0, "algo_seed": 3, "entloss_coef": 0.01
  },
  "step1_actor": "$RUN_DIR/step1_actor.pth",
  "step2_parallel_mode": "all 4 Parallel calls active (HEFT val, update-0 val, rollout, per-20 val)",
  "started_at": "$(date -Iseconds)"
}
JSON

# === Step 2 with watchdog ===
echo "[$(date)] === Running Step2 with watchdog ===" | tee -a $RUN_DIR/pipeline.log
bash $GOODRL/tools/run_with_resume.sh $RUN_DIR \
    --vm_types 5 --each_vm_type_num 5 --arr_rate 5.4 \
    --lr_a 0.0001 --lr_c 0.001 \
    --warmup_critic 500 --max_updates 1000 \
    --grad_control 0 --algo_seed 3 --entloss_coef 0.01 \
    --actor_atten_layers 1 \
    --checkpoint_interval 50

STEP2_EXIT=$?
echo "[$(date)] Step2 watchdog exit code: $STEP2_EXIT" | tee -a $RUN_DIR/pipeline.log
echo "[$(date)] === Pipeline DONE ===" | tee -a $RUN_DIR/pipeline.log
