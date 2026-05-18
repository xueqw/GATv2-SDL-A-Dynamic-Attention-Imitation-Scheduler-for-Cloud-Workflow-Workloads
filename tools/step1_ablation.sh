#!/bin/bash
# Step1 imitation learning ablation: GAT vs GATv2 across seeds
# Each run saves trained actor + log under run_dir/<arch>_seed<N>/

set -u
GOODRL=/home/xue/GOODRL
PY=/home/xue/anaconda3/envs/drl_sched/bin/python
ABLATION_DIR=$GOODRL/runs/20260516_step1_imitation_ablation
mkdir -p $ABLATION_DIR

cd $GOODRL

SEEDS="3 7 42"

for ARCH in GAT GATv2; do
    # Set up actor3.py for this architecture
    if [ "$ARCH" = "GAT" ]; then
        cp $GOODRL/policy/actor3.py.gat_backup $GOODRL/policy/actor3.py
    else
        # Start from gat_backup, apply GATv2 patch
        cp $GOODRL/policy/actor3.py.gat_backup $GOODRL/policy/actor3.py
        sed -i 's|from torch_geometric.nn import GINConv, GATConv, global_mean_pool|from torch_geometric.nn import GINConv, GATConv, GATv2Conv, global_mean_pool|' $GOODRL/policy/actor3.py
        sed -i 's|self.conv = GATConv(|self.conv = GATv2Conv(|' $GOODRL/policy/actor3.py
    fi

    # Verify the swap
    GAT_LINE=$(grep -c "GATv2Conv(" $GOODRL/policy/actor3.py)
    echo "[$(date +%H:%M:%S)] Architecture: $ARCH (GATv2Conv occurrences: $GAT_LINE)"

    for SEED in $SEEDS; do
        RUN=$ABLATION_DIR/${ARCH}_seed${SEED}
        mkdir -p $RUN

        echo "[$(date +%H:%M:%S)] ===== Running $ARCH seed=$SEED ====="

        # Write metadata
        cat > $RUN/metadata.json <<JSON
{
  "run_id": "${ARCH}_seed${SEED}",
  "architecture": "$ARCH",
  "algo_seed": $SEED,
  "command": "python step1.py --vm_types 5 --each_vm_type_num 5 --arr_rate 5.4 --lr_a 0.0001 --log_interval 1 --max_updates 10 --algo_seed $SEED",
  "purpose": "Step1 imitation learning ablation: GAT vs GATv2 architecture",
  "started_at": "$(date -Iseconds)"
}
JSON

        # Remove any old saved actor so step1 starts fresh
        rm -f $GOODRL/logs/a_5_5_0.0015.pth

        $PY -u step1.py \
            --vm_types 5 --each_vm_type_num 5 --arr_rate 5.4 \
            --lr_a 0.0001 --log_interval 1 --max_updates 10 \
            --algo_seed $SEED \
            > $RUN/step1.log 2>&1

        EXIT_CODE=$?
        echo "[$(date +%H:%M:%S)]   Exit code: $EXIT_CODE"

        # Save resulting actor
        if [ -f $GOODRL/logs/a_5_5_0.0015.pth ]; then
            cp $GOODRL/logs/a_5_5_0.0015.pth $RUN/final_actor.pth
            echo "[$(date +%H:%M:%S)]   Saved actor to $RUN/final_actor.pth"
        else
            echo "[$(date +%H:%M:%S)]   WARNING: no actor saved! step1 may have failed."
        fi

        # Quick result summary
        LAST_VAL=$(grep -a "Vlidation" $RUN/step1.log | tail -1)
        echo "[$(date +%H:%M:%S)]   Last validation: $LAST_VAL"
    done
done

echo "[$(date +%H:%M:%S)] ===== ALL DONE ====="
echo ""
echo "=== Final summary ==="
for RUN in $ABLATION_DIR/*/; do
    LAST=$(grep -a "Vlidation" $RUN/step1.log 2>/dev/null | tail -1 | grep -oE "mean_flowtime_deterministic: [0-9.]+\+/-[0-9.]+")
    echo "$(basename $RUN): $LAST"
done
