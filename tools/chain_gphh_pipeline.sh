#!/bin/bash
# Watcher: wait for GPHH training to finish, then chain into full GPHH-teacher pipeline.
#
# Phases:
#  1. WAIT for GP tree to be saved (logs/bestTree_5_5_0.0015.pkl exists + mainGP.py dead)
#  2. Generate GPHH memory (state, action) pairs
#  3. Run step1 with --teacher gphh
#  4. Promote step1 actor to validation_data/
#  5. Run step2 PPO via watchdog
set -u
GOODRL=/home/xue/GOODRL
PY=/home/xue/anaconda3/envs/drl_sched/bin/python
RUN_DIR=$GOODRL/runs/20260517_gphh_teacher_pipeline
GP_TREE=$GOODRL/logs/bestTree_5_5_0.0015.pkl
mkdir -p $RUN_DIR/checkpoints

LOG=$RUN_DIR/pipeline.log
echo "[$(date)] === GPHH-teacher pipeline watcher started ===" | tee -a $LOG

# ============================================================
# Phase 1: Wait for GPHH training to finish
# ============================================================
echo "[$(date)] Phase 1: Waiting for GPHH training to finish..." | tee -a $LOG
WAIT_COUNT=0
while true; do
    # Check if mainGP.py is still running
    MAINGP_RUNNING=$(pgrep -f "python.*mainGP.py" | wc -l)
    GP_TREE_EXISTS=0
    [ -f "$GP_TREE" ] && GP_TREE_EXISTS=1

    if [ "$MAINGP_RUNNING" = "0" ] && [ "$GP_TREE_EXISTS" = "1" ]; then
        echo "[$(date)] GPHH training finished and GP tree exists" | tee -a $LOG
        break
    fi

    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ $((WAIT_COUNT % 30)) = "0" ]; then
        # every 5 min
        echo "[$(date)] still waiting... mainGP_running=$MAINGP_RUNNING, gp_tree_exists=$GP_TREE_EXISTS" | tee -a $LOG
    fi
    sleep 10
done

ls -la $GP_TREE | tee -a $LOG

# ============================================================
# Phase 2: Generate GPHH memory
# ============================================================
MEM_DIR=$GOODRL/validation_data/GPHH
MEM_FILE=$MEM_DIR/GPHH_memory_500_5_5_0.0015.pkl
mkdir -p $MEM_DIR

echo "[$(date)] Phase 2: Generating GPHH memory..." | tee -a $LOG
cd $GOODRL
$PY -u tools/generate_gphh_memory.py \
    --gp_tree $GP_TREE \
    --num_instances 500 \
    --out $MEM_FILE \
    --vm_types 5 --each_vm_type_num 5 --arr_rate 5.4 \
    > $RUN_DIR/gen_memory.log 2>&1

if [ ! -f $MEM_FILE ]; then
    echo "[$(date)] FAILED to generate GPHH memory - see $RUN_DIR/gen_memory.log" | tee -a $LOG
    tail -20 $RUN_DIR/gen_memory.log | tee -a $LOG
    exit 1
fi
echo "[$(date)] Memory generated: $(ls -lh $MEM_FILE)" | tee -a $LOG

# ============================================================
# Phase 3: Restore baseline step1 actor + run step1 with GPHH teacher
# ============================================================
# Make sure validation_data/step1/actors/ has a clean baseline (GATv2 trained earlier)
# In case other experiments polluted it, restore from backup
BAK=$GOODRL/validation_data/step1/actors/a_5_5_0.0015.pth.bak_pre_actor_attn
if [ -f $BAK ]; then
    cp $BAK $GOODRL/validation_data/step1/actors/a_5_5_0.0015.pth
fi

# Clean logs/ so step1 saves fresh
rm -f $GOODRL/logs/a_5_5_0.0015.pth $GOODRL/logs/c_5_5_0.0015.pth

echo "[$(date)] Phase 3: Running step1 with --teacher gphh..." | tee -a $LOG
$PY -u step1.py \
    --vm_types 5 --each_vm_type_num 5 --arr_rate 5.4 \
    --lr_a 0.0001 --log_interval 1 --max_updates 10 \
    --algo_seed 3 \
    --teacher gphh \
    --actor_pointer 0 \
    --actor_atten_layers 0 \
    > $RUN_DIR/step1.log 2>&1

STEP1_EXIT=$?
if [ "$STEP1_EXIT" != "0" ] || [ ! -f $GOODRL/logs/a_5_5_0.0015.pth ]; then
    echo "[$(date)] step1 FAILED - see $RUN_DIR/step1.log" | tee -a $LOG
    tail -30 $RUN_DIR/step1.log | tee -a $LOG
    exit 2
fi

LAST_VAL=$(grep -a "Vlidation" $RUN_DIR/step1.log | tail -1)
echo "[$(date)] step1 done. Last validation: $LAST_VAL" | tee -a $LOG

# Promote
cp $GOODRL/logs/a_5_5_0.0015.pth $GOODRL/validation_data/step1/actors/a_5_5_0.0015.pth
cp $GOODRL/logs/a_5_5_0.0015.pth $RUN_DIR/step1_actor.pth

# Write metadata
cat > $RUN_DIR/metadata.json <<JSON
{
  "run_id": "20260517_gphh_teacher_pipeline",
  "description": "Full pipeline using GPHH-trained tree as imitation teacher (instead of HEFT).",
  "teacher": "gphh",
  "architecture": "GATv2 + MLP scoring (baseline actor)",
  "gp_tree_source": "$GP_TREE",
  "gphh_memory": "$MEM_FILE",
  "step1_record": "$LAST_VAL",
  "non_default_flags": {
    "teacher": "gphh", "lr_a": 0.0001, "lr_c": 0.001,
    "warmup_critic": 500, "max_updates": 1000,
    "grad_control": 0, "algo_seed": 3, "entloss_coef": 0.01
  },
  "started_at": "$(date -Iseconds)"
}
JSON

# ============================================================
# Phase 4: step2 via watchdog
# ============================================================
echo "[$(date)] Phase 4: Running step2 via watchdog..." | tee -a $LOG
bash $GOODRL/tools/run_with_resume.sh $RUN_DIR \
    --vm_types 5 --each_vm_type_num 5 --arr_rate 5.4 \
    --lr_a 0.0001 --lr_c 0.001 \
    --warmup_critic 500 --max_updates 1000 \
    --grad_control 0 --algo_seed 3 --entloss_coef 0.01 \
    --teacher gphh \
    --actor_pointer 0 \
    --actor_atten_layers 0 \
    --checkpoint_interval 50

echo "[$(date)] === Pipeline DONE ===" | tee -a $LOG
