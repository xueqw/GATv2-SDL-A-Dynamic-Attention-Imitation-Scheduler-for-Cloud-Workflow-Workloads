# Research Plan: NCO-Inspired Scalable Actor for Dynamic Workflow Scheduling

> Created: 2026-05-15
> Status: Drafting
> Target venue: ICLR / NeurIPS / ICML workshop, or KDD / AAAI main track

---

## 1. Positioning

### 1.1 Context

Yang et al. (ICLR 2025), **GOODRL** ("Graph Assisted Offline-Online Deep Reinforcement Learning for Dynamic Workflow Scheduling") is the current SOTA on DWS. Key design:

- **Actor**: per-pair pairwise reasoning — constructs `N_VM` separate sub-graphs per decision (one per candidate VM), runs GAT on each, scores each independently via MLP.
- **Critic**: single global graph — one GAT pass over all task nodes + all hypothetical VM placement edges, followed by self-attention + mean pooling.

GOODRL's design is internally consistent: actor produces `N_VM` outputs so it does `N_VM` computations; critic produces 1 output so it does 1 computation.

### 1.2 Our Hypothesis

GOODRL's pairwise actor scales as **O(N_VM)** per decision step. While this is acceptable at the configurations tested in the paper (`vmNum=25, 24`), it limits deployment to larger cloud environments (`vmNum=100, 200, 1000`).

We hypothesize that **NCO-style shared-encoder + pointer-decoder** architectures (Vinyals 2015, Kool 2019), proven effective on TSP/VRP/JSP, can achieve **comparable performance at O(1) actor cost**.

### 1.3 Framing

- **NOT**: "GOODRL's actor is inefficient, we fix it"
- **YES**: "GOODRL achieves SOTA at vmNum ≤ 25. We characterize its scaling behavior up to vmNum=200 and propose an NCO-inspired alternative that maintains O(1) cost regardless of vmNum, enabling deployment in large-scale settings."

Tone: parallel exploration, not antagonistic.

---

## 2. Technical Approach

### 2.1 Actor: From Pairwise Sub-graphs → Pointer-Attention Actor (PAA)

**GOODRL current actor** (`wrapper` in `policy/actor3.py:49`):
- 25 sub-graphs, each with: full workflow DAG nodes + 1 hypothetical VM-placement edge for VM_i + VM_i features stamped on candidate task rows
- GAT runs 25× (batched as one large tensor); 25 different focused embeddings produced
- MLP independently scores each → softmax over 25

**Our PAA**:
1. Single global graph:
   - Task nodes (workflow DAGs, queue chain edges)
   - **VM nodes** (explicit, one per VM, with VM features)
   - **Edge types**:
     - `task -[depends_on]-> task` (DAG, directional)
     - `task -[queued_after]-> task` (queue chain on each VM)
     - `task -[candidate_for]-> vm` (which VMs are options for which ready task)
     - `vm -[same_type]-> vm` (same VM-type pool, optional)
2. **HeteroGATv2 encoder**: 2 layers, one HeteroConv per edge type. Produces task embeddings + VM embeddings.
3. **Pointer scoring head**:
   - Query: `Q = W_q · (ready_task_embed)`
   - Key: `K_i = W_k · (vm_i_embed)`
   - Score: `score_i = Q · K_i / sqrt(d)`, optionally `score_i = C · tanh(score_i)` (Kool 2019 style)
   - Softmax over `i ∈ [0, N_VM)`

**Cost**: 1 HeteroGAT forward pass + 1 pointer attention = O(|E|) vs original O(N_VM · |E|).

### 2.2 Critic: Unchanged

GOODRL critic is already O(1) and has empirically validated SelfAttention (Yang et al. Table 5: ablation shows SelfAttention reduces value loss from 8.56 to 6.25). No reason to modify.

This keeps our ablation single-variable: only actor architecture differs.

### 2.3 Why GATv2 inside HeteroConv

- GATv2 has dynamic attention (proven empirically: see `runs/attention_compare_GAT_vs_GATv2.png` and `quantify_attention.py`)
- Marginal cost over GAT (1.8× more params), but theoretically strictly more expressive
- Compatible with HeteroConv (PyG supports both)

---

## 3. Experimental Design

### 3.1 Settings

Standard configurations from paper:
- `S` workflow size (30-task DAGs)
- 4 DAG types: CyberShake, Montage, Inspiral, Sipht
- `wf_num=30` workflows per instance
- Validation set: `validation_instance_2024.npy` (paper's fixed file)

Scale dimensions (NEW — paper only tested vmNum=24,25):
- vmNum=25: `--vm_types 5 --each_vm_type_num 5`
- vmNum=24: `--vm_types 6 --each_vm_type_num 4`
- vmNum=50: `--vm_types 5 --each_vm_type_num 10`
- vmNum=100: `--vm_types 10 --each_vm_type_num 10`
- vmNum=200: `--vm_types 10 --each_vm_type_num 20`

Arrival rates: 5.4, 9 (paper's rates)

### 3.2 Main Ablation Matrix

| Variant | Actor encoder | Actor scoring | Notes |
|---|---|---|---|
| **A0** GOODRL baseline | 25× sub-graph GAT | MLP | paper config |
| **A1** + GATv2 | 25× sub-graph GATv2 | MLP | single-variable: GAT→GATv2 |
| **A2** + Pointer | 25× sub-graph GATv2 | Pointer | single-variable: MLP→Pointer (still pairwise) |
| **A3** + Single graph | 1× HeteroGATv2 | MLP | drop pairwise |
| **A4** Full PAA (ours) | 1× HeteroGATv2 | Pointer | combined |

Each cell × 3-5 seeds × 5 vmNum configurations × 2 arrival rates ≈ **150-250 runs**.

### 3.3 Metrics

Primary:
- **Mean flowtime** on validation set (paper's primary metric)
- **Wall-clock training time** to convergence
- **Inference time** per decision

Secondary:
- v_loss / v_mre curves (critic stability)
- p_loss / grad_changes curves (actor stability)
- Attention entropy / dynamism (qualitative)

Scaling-specific (key plot):
- Training time vs vmNum
- Inference time vs vmNum
- Memory peak vs vmNum

### 3.4 Statistical Treatment

- Mean ± stderr over seeds
- Paired t-test or Wilcoxon for paired comparisons
- Bootstrap CI for scaling curves

---

## 4. Implementation Plan

### 4.1 Code Structure

```
GOODRL/
  policy/
    actor3.py              # original, untouched (baseline)
    actor3.py.gat_backup   # original GAT backup
    hetero_encoder.py      # NEW: HeteroGATv2 encoder
    hetero_actor.py        # NEW: PAA actor class
    pointer_head.py        # NEW: pointer attention scoring
  env/
    workflow_scheduling_v3/
      lib/
        cloud_env_maxPktNum.py  # +30 lines: expose VM features as separate, build candidate edges
        hetero_state.py         # NEW: build HeteroData from env outputs
  step1.py                  # +10 lines: --use_hetero flag
  step2.py                  # +10 lines: same flag
  step3.py                  # +10 lines: same flag
  config/Params.py          # +5 lines: --use_hetero, --pointer_scaling
  tools/
    analyze.py              # existing log parser
    compare_attention.py    # existing
    quantify_attention.py   # existing
    scaling_eval.py         # NEW: run baseline + ours across vmNum
    train_with_resume.py    # NEW: checkpoint resume wrapper
```

Total NEW: ~600 lines
Total CHANGED: ~80 lines (mostly Param flags and switch)

### 4.2 File-by-file TODO

#### NEW: `policy/hetero_encoder.py` (~150 lines)
```python
class HeteroGATv2Encoder(nn.Module):
    def __init__(self, task_dim, vm_dim, hidden_dim, num_layers, heads, dropout):
        # Linear projections from raw features to hidden_dim
        # Stack of HeteroConv layers, each with GATv2Conv per edge type
        # Edge types: ('task','dag','task'), ('task','queue','task'),
        #             ('task','cand','vm'), ('vm','rev_cand','task'),
        #             ('vm','same_type','vm')
    def forward(self, x_dict, edge_index_dict):
        # Project, multi-layer HeteroConv with ELU, return dict
```

#### NEW: `policy/pointer_head.py` (~80 lines)
```python
class PointerHead(nn.Module):
    def __init__(self, hidden_dim, tanh_scaling=10.0):
        self.W_q, self.W_k = Linear, Linear
        self.C = tanh_scaling
    def forward(self, query_emb, key_embs):
        # query_emb: (B, d), key_embs: (B, N_VM, d)
        # returns: (B, N_VM) scores
```

#### NEW: `policy/hetero_actor.py` (~200 lines)
```python
class HeteroPointerActor(nn.Module):
    def __init__(self, ..., hidden_dim, vmNum):
        self.encoder = HeteroGATv2Encoder(...)
        self.pointer = PointerHead(hidden_dim)
    def forward(self, data, candidate_task_idx, candidate_vm_idx, deterministic=False):
        x_dict = self.encoder(data.x_dict, data.edge_index_dict)
        task_emb = x_dict['task'][candidate_task_idx]   # ready task emb
        vm_emb = x_dict['vm'][candidate_vm_idx]          # candidate VMs
        scores = self.pointer(task_emb, vm_emb)
        pi = F.softmax(scores, dim=-1)
        # sample / argmax, log_prob, entropy
        return action_id, log_prob, entropy
    def eval_actions(self, ...): pass
```

#### NEW: `env/.../hetero_state.py` (~100 lines)
```python
def build_hetero_state(env_state):
    """Build PyG HeteroData from env state tuple."""
    data = HeteroData()
    data['task'].x = task_feats              # (N_tasks, 3) - drop VM cols
    data['vm'].x = vm_feats                  # (N_vms, 4)
    data['task', 'dag', 'task'].edge_index = dag_edges
    data['task', 'queue', 'task'].edge_index = queue_edges
    data['task', 'cand', 'vm'].edge_index = candidate_edges  # (2, N_cand × N_vm)
    # rev edges added by PyG ToHetero or manually
    return data
```

#### MODIFY: `env/.../cloud_env_maxPktNum.py` (+30 lines)
- Expose `vm_features` separately (don't stamp into wf_features)
- Build `candidate_edges` list: (ready_task_idx, vm_idx) pairs

#### MODIFY: `step2.py`, `step1.py`, `step3.py` (+10 each)
- `--use_hetero` flag selects between PPO-original / PPO-with-PAA
- Branch on flag for encoder class

#### NEW: `tools/train_with_resume.py` (~150 lines)
```python
# Wrapper around step2 that saves checkpoint every K updates
# and resumes from last checkpoint if interrupted
# Stores: actor.pth, critic.pth, optimizer.pth, episode_counter.pth
```

#### NEW: `tools/scaling_eval.py` (~100 lines)
```python
# Run baseline + PAA on grid of vmNum × arrival_rate
# Collect: mean_flowtime, training_time, inference_time
# Output: CSV + scaling curve PNG
```

### 4.3 Phase Ordering

```
Phase 0: Prerequisites (1-2 days)
  ✓ RAM fix (physical action by user)
  ✓ Verify memtester clean after RAM swap
  ✓ Implement checkpoint resume (tools/train_with_resume.py)
  ✓ Verify resume works on a short run
  
Phase 1: HeteroEncoder + PAA prototype (3-4 days)
  ✓ Write hetero_encoder.py, hetero_actor.py, pointer_head.py
  ✓ Write hetero_state.py
  ✓ Step1 imitation runs end-to-end without crash
  ✓ Step1 imitation loss is reasonable (CE ~2.5 on vmNum=25)
  
Phase 2: Step2 training + baseline (1 week)
  ✓ A0 (GOODRL baseline): 3 seeds × vmNum=25,24 → 6 runs (~3 days)
  ✓ A4 (PAA): 3 seeds × vmNum=25,24 → 6 runs
  ✓ Compare: A0 vs A4 mean flowtime at vmNum ≤ 25 (sanity check parity)
  
Phase 3: Scaling experiments (1 week)
  ✓ A0 at vmNum=50,100,200 (likely OOM or very slow on paper code)
  ✓ A4 at vmNum=50,100,200
  ✓ Scaling curves
  
Phase 4: Ablation refinement (1 week)
  ✓ A1, A2, A3 intermediate variants
  ✓ Multi-seed runs for statistical significance
  
Phase 5: Writing (2 weeks)
  ✓ Method section
  ✓ Experiments + figures
  ✓ Related work positioning
  ✓ Ablation tables
```

Total: ~6 weeks if Phase 0 hardware completes in 1-2 days.

---

## 5. Pre-existing Findings (Worth Mentioning in Paper)

These are things we already discovered during initial exploration:

### 5.1 Released code bug: critic not updating

GOODRL's released `step2.py` + `policy/actor3.py` train() method, under default `grad_control=1`, calls only `optimizer_actor.step()`, **never updating critic** despite `loss.backward()` filling its gradients. v_mre stays at ~110% throughout warmup.

Fix: `--grad_control 0` or properly call `optimizer_critic.step()`. We verified the fix produces v_loss drop from ~2.5 to ~0.1.

**Position in paper**: a footnote or appendix item, noting "we discovered and fixed an unintended bug in the released code; results below use the fixed version for fair baseline".

### 5.2 GATv2 attention quantification

Already done (see `tools/quantify_attention.py`):
- GAT Spearman rank correlation = 1.0000 (perfectly static)
- GATv2 Spearman = 0.9592, rank reversal rate 6.76%
- Top-1 agreement: GAT 100%, GATv2 92.7%

**Position in paper**: motivation / preliminary analysis for why GATv2 is used.

### 5.3 Hyperparameter tuning candidates

From exploratory runs:
- `--entloss_coef 0.01` adds soft entropy regularization (paper has it at 0)
- `--warmup_critic 500` instead of 200 lets critic learn better before actor unlocks

Note: these are training tricks, not architectural; mention briefly but don't lean on them.

---

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| RAM hardware not fixed in time | Medium | High | Use checkpoint resume; pay for cloud GPU as fallback |
| PAA performance worse than baseline at vmNum=25 | Medium | High | Have intermediate variants A2/A3 to localize the issue; may need pointer-head tuning |
| HeteroData/HeteroConv has bugs in PyG | Low | Medium | Test forward pass thoroughly in Phase 1 |
| GOODRL baseline OOM at vmNum=100 | High | Medium | This is actually our point! Document failure mode |
| Multiple seeds needed for stat sig but expensive | Medium | Medium | Start with 3 seeds, increase if results are noisy |
| Reviewers say "GOODRL was already O(N_VM), this is obvious" | Medium | Medium | Frame as scaling characterization + algorithmic alternative, cite NCO lineage strongly |

---

## 7. Open Questions / Design Choices

These need decisions before/during Phase 1:

### 7.1 Pointer scoring formula
- Option a: `score = (W_q·t) · (W_k·v) / sqrt(d)` (Bello 2017 style)
- Option b: `score = C · tanh((W_q·t) · (W_k·v) / sqrt(d))` (Kool 2019 style, prevents extreme softmax)
- Choice: start with (b), `C=10`, ablate if needed.

### 7.2 Multi-head pointer
- Single-head or multi-head pointer?
- Choice: start single-head; if performance gap to baseline at vmNum=25, add multi-head.

### 7.3 VM-VM "same_type" edges
- Include or not?
- Choice: include in default; ablate in supplementary.

### 7.4 SelfAttention in actor (hybrid)
- Should we also add candidate-candidate SelfAttention before Pointer scoring?
- Choice: NO in main config (keep change minimal); SUPP appendix variant if time allows.

### 7.5 Imitation learning for hetero
- Does step1 still imitate HEFT? HEFT action format must map to hetero action format.
- Choice: yes, same HEFT supervision; map HEFT VM choice to pointer target index.

### 7.6 Action space at large vmNum
- At vmNum=200, softmax over 200 may be challenging
- Choice: consider temperature scaling or top-k pruning if needed.

---

## 8. Deliverables (writing-ready outputs)

By Phase 4 end, we should have:

1. **Figure: scaling curves** — training time and inference time vs vmNum, for A0 and A4
2. **Figure: validation curves** — mean flowtime over training updates, A0 vs A4 at vmNum=25
3. **Table: main results** — A0 through A4 across all vmNum, mean flowtime ± stderr
4. **Table: critic ablation reproduction** — show our A0 reproduces paper Table 5 (sanity)
5. **Figure: attention dynamism** — already have via `compare_attention.py`
6. **Appendix: bug report** — the critic-not-updating issue and fix verification
7. **Appendix: hyperparameter sensitivity** — entloss_coef, warmup_critic, lr_a small grid

---

## 9. Citation Lineage (Related Work positioning)

Strongest connections we should cite:

- **GOODRL** (Yang et al. 2025): baseline we extend
- **Pointer Networks** (Vinyals et al. 2015): foundational pointer-attention paradigm
- **Attention Model for routing** (Kool et al. 2019): closest NCO precedent for our actor design
- **L2D for JSP** (Cheng et al. 2023): in GOODRL's reference list, uses pointer for job-shop
- **POMO** (Kwon et al. 2020): if we add multiple-pointer trick
- **GATv2** (Brody et al. 2022): for attention quality
- **HGT** (Hu et al. 2020): heterogeneous graph foundation
- **PPO** (Schulman et al. 2017): training algorithm
- **DAPG** (Rajeswaran 2018): for behavior cloning regularization (if we add it)

---

## 10. Decisions Pending User Input

- [ ] Confirm story v2 (scaling-focused) is the right framing
- [ ] Decide target venue (affects writing tone, page limits)
- [ ] Set physical RAM action timeline (when to swap/test)
- [ ] Approve compute budget for ~250 runs (RAM permitting)
- [ ] Decide on co-authorship / advisor involvement (if applicable)

---

*End of plan*
