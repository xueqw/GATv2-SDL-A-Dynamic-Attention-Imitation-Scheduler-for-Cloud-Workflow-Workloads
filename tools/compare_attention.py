#!/usr/bin/env python3
"""Compare GAT vs GATv2 attention patterns on a real workflow DAG.

Loads Montage_25 (one of the 4 DAGs used in the project), builds:
- GATConv (original GAT)
- GATv2Conv (our modification)
both with IDENTICAL random initialization. Then forwards the same input
and compares attention weights to show:

1. GAT: attention rank of neighbors is identical across all query nodes
   (static attention - alpha_ij depends only on j).
2. GATv2: attention rank can differ per query node (dynamic attention).
"""
import sys
import os
sys.path.insert(0, '/home/xue/GOODRL/env/workflow_scheduling_v3/lib')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv, GATv2Conv
import processDAG

# --- 1. Load a real DAG from the project ---
DAX = '/home/xue/GOODRL/env/workflow_scheduling_v3/dax/Montage_25.xml'
G, _ = processDAG.buildGraph('Montage', DAX)
print(f"Loaded Montage_25: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Build edge_index tensor
edges = list(G.edges())
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
# Make undirected (project does this in actor3.py:565)
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

# --- 2. Synthesize input features (7-dim, matching project: 3 wf + 4 vm) ---
N = G.number_of_nodes()
torch.manual_seed(42)
IN_DIM = 7
HID = 32
HEADS = 1

x = torch.randn(N, IN_DIM)

# --- 3. Initialize GATConv and GATv2Conv with same seed ---
torch.manual_seed(0)
gat = GATConv(IN_DIM, HID, heads=HEADS, dropout=0.0, concat=False, add_self_loops=False)

torch.manual_seed(0)
gatv2 = GATv2Conv(IN_DIM, HID, heads=HEADS, dropout=0.0, concat=False, add_self_loops=False)

print(f"GATConv  param count: {sum(p.numel() for p in gat.parameters())}")
print(f"GATv2Conv param count: {sum(p.numel() for p in gatv2.parameters())}")

# --- 4. Forward with return_attention_weights ---
gat.eval(); gatv2.eval()
with torch.no_grad():
    _, (ei_gat, alpha_gat) = gat(x, edge_index, return_attention_weights=True)
    _, (ei_v2,  alpha_v2)  = gatv2(x, edge_index, return_attention_weights=True)

# alpha shape: (num_edges, num_heads). With heads=1, squeeze.
alpha_gat = alpha_gat.squeeze(-1).numpy()
alpha_v2  = alpha_v2.squeeze(-1).numpy()

# Build attention MATRIX [target_node, source_node] = alpha
# edge_index is (2, E). Each col is [src, dst] (depends on PyG convention).
# PyG: message flows src -> dst, alpha is on edge (src, dst), normalized over src per dst.
def alpha_matrix(ei, alpha, N):
    """Return matrix A[query=dst, neighbor=src] = alpha."""
    M = np.zeros((N, N))
    src = ei[0].numpy(); dst = ei[1].numpy()
    for s, d, a in zip(src, dst, alpha):
        M[d, s] = a
    return M

A_gat = alpha_matrix(ei_gat, alpha_gat, N)
A_v2  = alpha_matrix(ei_v2,  alpha_v2,  N)

# --- 5. Find query nodes with shared neighbors (good for comparison) ---
# Pick query nodes that have at least 2 incoming edges; look for pairs that share neighbors.
in_neighbors = {n: set() for n in range(N)}
for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
    in_neighbors[d].add(s)

# Find 3 query nodes with the most shared neighbors with each other
candidates = sorted([n for n in range(N) if len(in_neighbors[n]) >= 2], key=lambda n: -len(in_neighbors[n]))
queries = candidates[:4]
print(f"\nPicked query nodes: {queries}")
for q in queries:
    print(f"  node {q}: in-neighbors = {sorted(in_neighbors[q])}")

# --- 6. Plot: 4 subplots ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (top-left): full attention heatmap for GAT
ax = axes[0, 0]
im = ax.imshow(A_gat, cmap='hot', aspect='auto')
ax.set_title(f'GAT  attention matrix\n(rows = query / dst, cols = source)', fontsize=11)
ax.set_xlabel('Source node'); ax.set_ylabel('Query node')
plt.colorbar(im, ax=ax, fraction=0.046)

# (top-right): same for GATv2
ax = axes[0, 1]
im = ax.imshow(A_v2, cmap='hot', aspect='auto')
ax.set_title(f'GATv2 attention matrix', fontsize=11)
ax.set_xlabel('Source node'); ax.set_ylabel('Query node')
plt.colorbar(im, ax=ax, fraction=0.046)

# (bottom-left): bar chart - GAT, multiple query nodes, attention over shared neighbors
ax = axes[1, 0]
# All neighbors across the 4 selected queries
all_neighbors = sorted(set().union(*[in_neighbors[q] for q in queries]))
width = 0.18
xpos = np.arange(len(all_neighbors))
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(queries)))
for i, q in enumerate(queries):
    vals = [A_gat[q, n] for n in all_neighbors]
    ax.bar(xpos + i * width, vals, width, label=f'query node {q}', color=colors[i])
ax.set_xticks(xpos + width * (len(queries)-1) / 2)
ax.set_xticklabels([str(n) for n in all_neighbors])
ax.set_xlabel('Source (neighbor) node'); ax.set_ylabel('Attention weight alpha')
ax.set_title('GAT: same neighbor → same attention regardless of query\n(if both query nodes see this neighbor, the bars are identical-shaped)', fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (bottom-right): same chart but GATv2
ax = axes[1, 1]
for i, q in enumerate(queries):
    vals = [A_v2[q, n] for n in all_neighbors]
    ax.bar(xpos + i * width, vals, width, label=f'query node {q}', color=colors[i])
ax.set_xticks(xpos + width * (len(queries)-1) / 2)
ax.set_xticklabels([str(n) for n in all_neighbors])
ax.set_xlabel('Source (neighbor) node'); ax.set_ylabel('Attention weight alpha')
ax.set_title('GATv2: same neighbor can get DIFFERENT attention\ndepending on which query is asking', fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.suptitle(f'GAT vs GATv2 attention on Montage_25 workflow ({N} tasks)\n'
             f'Random init (seed=0), same input features, identical edge structure',
             fontsize=12)
plt.tight_layout()
out_path = '/home/xue/GOODRL/runs/attention_compare_GAT_vs_GATv2.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")

# --- 7. Quantify the "staticness" of GAT vs GATv2 ---
print("\n=== Quantitative analysis ===")
# For each pair of query nodes that share a neighbor n,
# compare their normalized attention to n (after softmax/just look at raw alpha).
# In GAT: should be a constant ratio (since alpha_in = f(n) / sum...).
# Better metric: for each pair of queries (q1, q2) and their shared neighbors,
# compute Pearson correlation of their alpha distributions.
shared_pairs = []
for i, q1 in enumerate(queries):
    for q2 in queries[i+1:]:
        shared = sorted(in_neighbors[q1] & in_neighbors[q2])
        if len(shared) >= 2:
            shared_pairs.append((q1, q2, shared))

if shared_pairs:
    print(f"Query pairs with >=2 shared neighbors: {len(shared_pairs)}")
    for q1, q2, sh in shared_pairs[:5]:
        a1_gat = np.array([A_gat[q1, n] for n in sh])
        a2_gat = np.array([A_gat[q2, n] for n in sh])
        a1_v2  = np.array([A_v2[q1, n] for n in sh])
        a2_v2  = np.array([A_v2[q2, n] for n in sh])
        # Pearson correlation between query 1 and query 2 attention over their shared neighbors
        c_gat = np.corrcoef(a1_gat, a2_gat)[0, 1] if len(sh) >= 2 else float('nan')
        c_v2  = np.corrcoef(a1_v2,  a2_v2 )[0, 1] if len(sh) >= 2 else float('nan')
        print(f"  Queries {q1} vs {q2} (shared={sh}):")
        print(f"    GAT   attn correlation = {c_gat:+.4f}   (closer to 1.0 = more static)")
        print(f"    GATv2 attn correlation = {c_v2:+.4f}")
else:
    print("No shared-neighbor pairs found (rare). Try a different DAG.")

# Also: per-query, what fraction of attention goes to top-1 neighbor?
print("\nAttention concentration (top-1 share):")
for q in queries:
    neigh = sorted(in_neighbors[q])
    g_attn = np.array([A_gat[q, n] for n in neigh])
    v_attn = np.array([A_v2[q, n] for n in neigh])
    g_top = g_attn.max() / g_attn.sum() if g_attn.sum() > 0 else 0
    v_top = v_attn.max() / v_attn.sum() if v_attn.sum() > 0 else 0
    print(f"  q={q}: GAT top1 = {g_top:.3f}, GATv2 top1 = {v_top:.3f}")

print("\nDone.")
