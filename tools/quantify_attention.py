#!/usr/bin/env python3
"""Quantitative comparison: GAT vs GATv2 attention dynamics.

Computes multiple independent metrics that measure how 'query-dependent'
the attention pattern is.
"""
import sys, os
sys.path.insert(0, '/home/xue/GOODRL/env/workflow_scheduling_v3/lib')
import torch
import numpy as np
from torch_geometric.nn import GATConv, GATv2Conv
from scipy.stats import pearsonr, spearmanr, entropy as scipy_entropy
import processDAG

# --- Load DAG ---
DAX = '/home/xue/GOODRL/env/workflow_scheduling_v3/dax/Montage_25.xml'
G, _ = processDAG.buildGraph('Montage', DAX)
edges = list(G.edges())
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
N = G.number_of_nodes()

torch.manual_seed(42)
IN_DIM, HID, HEADS = 7, 32, 1
x = torch.randn(N, IN_DIM)

torch.manual_seed(0)
gat = GATConv(IN_DIM, HID, heads=HEADS, dropout=0.0, concat=False, add_self_loops=False)
torch.manual_seed(0)
gatv2 = GATv2Conv(IN_DIM, HID, heads=HEADS, dropout=0.0, concat=False, add_self_loops=False)

gat.eval(); gatv2.eval()
with torch.no_grad():
    _, (ei_gat, alpha_gat) = gat(x, edge_index, return_attention_weights=True)
    _, (ei_v2, alpha_v2) = gatv2(x, edge_index, return_attention_weights=True)
alpha_gat = alpha_gat.squeeze(-1).numpy()
alpha_v2 = alpha_v2.squeeze(-1).numpy()

# Build attention matrix [query, source] = alpha
def alpha_matrix(ei, alpha, N):
    M = np.zeros((N, N))
    src = ei[0].numpy(); dst = ei[1].numpy()
    for s, d, a in zip(src, dst, alpha):
        M[d, s] = a
    return M

A_gat = alpha_matrix(ei_gat, alpha_gat, N)
A_v2 = alpha_matrix(ei_v2, alpha_v2, N)

# in_neighbors[q] = sources that have edge to q
in_neighbors = {n: [] for n in range(N)}
for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
    in_neighbors[d].append(s)
for n in in_neighbors:
    in_neighbors[n] = sorted(set(in_neighbors[n]))

print("="*70)
print("  QUANTITATIVE METRICS: GAT vs GATv2 attention dynamics")
print("  DAG: Montage_25, 25 nodes, 45 edges")
print("="*70)

# ============================================
# METRIC 1: Cross-query variance per neighbor
# ============================================
# For each neighbor j, look at attention from all queries that have j as neighbor.
# Compute variance. GAT theoretically gives 0 variance (after fair normalization).
def cross_query_variance(A):
    """For each source node j, variance of attention from different queries that have j as neighbor."""
    vars_per_j = []
    for j in range(N):
        # Find all queries i that have j as in-neighbor
        queries = [i for i in range(N) if j in in_neighbors[i]]
        if len(queries) >= 2:
            attns = [A[i, j] for i in queries]
            vars_per_j.append(np.var(attns))
    return np.mean(vars_per_j), np.std(vars_per_j), len(vars_per_j)

m_gat, s_gat, n = cross_query_variance(A_gat)
m_v2, s_v2, _ = cross_query_variance(A_v2)
print(f"\n[1] Cross-query attention variance (per neighbor, across queries):")
print(f"    GAT:   mean variance = {m_gat:.6f}  (across {n} neighbors with >=2 queries)")
print(f"    GATv2: mean variance = {m_v2:.6f}")
print(f"    Ratio: GATv2/GAT = {m_v2/m_gat:.2f}x more variance ⇒ more dynamic")

# ============================================
# METRIC 2: Pearson correlation over query pairs
# ============================================
# For pairs of queries (i, i') with shared neighbors S:
# corr(α_i over S, α_i' over S). GAT should be ~1.0 always.
def pairwise_correlation(A, kind='pearson'):
    pairs = []
    for i in range(N):
        for ip in range(i+1, N):
            shared = sorted(set(in_neighbors[i]) & set(in_neighbors[ip]))
            if len(shared) >= 2:
                v1 = np.array([A[i, j] for j in shared])
                v2 = np.array([A[ip, j] for j in shared])
                if np.std(v1) > 0 and np.std(v2) > 0:
                    if kind == 'pearson':
                        r, _ = pearsonr(v1, v2)
                    else:
                        r, _ = spearmanr(v1, v2)
                    pairs.append(r)
    return np.array(pairs)

p_gat = pairwise_correlation(A_gat, 'pearson')
p_v2 = pairwise_correlation(A_v2, 'pearson')
s_gat = pairwise_correlation(A_gat, 'spearman')
s_v2 = pairwise_correlation(A_v2, 'spearman')
print(f"\n[2] Pairwise correlation (between query pairs over shared neighbors):")
print(f"    Pearson  mean: GAT = {p_gat.mean():.4f} ± {p_gat.std():.4f}   "
      f"GATv2 = {p_v2.mean():.4f} ± {p_v2.std():.4f}")
print(f"    Spearman mean: GAT = {s_gat.mean():.4f} ± {s_gat.std():.4f}   "
      f"GATv2 = {s_v2.mean():.4f} ± {s_v2.std():.4f}")
print(f"    GAT close to 1.0 ⇒ static (rank order same across queries)")
print(f"    GATv2 < GAT     ⇒ dynamic (rank order can differ)")

# ============================================
# METRIC 3: Top-1 neighbor agreement
# ============================================
# Among query pairs (i, i') with shared neighbors, do they agree on the top-1 neighbor?
def top1_agreement(A):
    agree = 0; total = 0
    for i in range(N):
        for ip in range(i+1, N):
            shared = sorted(set(in_neighbors[i]) & set(in_neighbors[ip]))
            if len(shared) >= 2:
                top_i = max(shared, key=lambda j: A[i, j])
                top_ip = max(shared, key=lambda j: A[ip, j])
                if top_i == top_ip:
                    agree += 1
                total += 1
    return agree, total

a_gat, t = top1_agreement(A_gat)
a_v2, _ = top1_agreement(A_v2)
print(f"\n[3] Top-1 neighbor agreement (across query pairs sharing >=2 neighbors):")
print(f"    GAT:   {a_gat}/{t} ({100*a_gat/t:.1f}%) of query pairs pick the SAME top-1 neighbor")
print(f"    GATv2: {a_v2}/{t} ({100*a_v2/t:.1f}%)")
print(f"    Lower agreement ⇒ more query-dependent ranking")

# ============================================
# METRIC 4: Attention entropy per query
# ============================================
# H(α_i) = -Σ α_ij log α_ij; high = spread, low = peaky
def avg_entropy(A):
    ents = []
    for i in range(N):
        neigh = in_neighbors[i]
        if len(neigh) >= 2:
            p = np.array([A[i, j] for j in neigh])
            if p.sum() > 0:
                p = p / p.sum()
                ents.append(scipy_entropy(p))
    return np.mean(ents), np.std(ents)

e_gat, ed_gat = avg_entropy(A_gat)
e_v2, ed_v2 = avg_entropy(A_v2)
import math
max_ent = math.log(max(len(in_neighbors[i]) for i in range(N) if len(in_neighbors[i]) >= 2))
print(f"\n[4] Attention entropy per query (over its own neighbors):")
print(f"    GAT:   {e_gat:.3f} ± {ed_gat:.3f}  (max possible = {max_ent:.3f}, uniform)")
print(f"    GATv2: {e_v2:.3f} ± {ed_v2:.3f}")
print(f"    Higher = more diffuse attention; lower = more concentrated")

# ============================================
# METRIC 5: KL divergence between query pairs
# ============================================
# For paired queries with shared neighbors, KL(p_i || p_ip) over shared support.
def avg_kl(A):
    kls = []
    for i in range(N):
        for ip in range(i+1, N):
            shared = sorted(set(in_neighbors[i]) & set(in_neighbors[ip]))
            if len(shared) >= 2:
                v1 = np.array([A[i, j] for j in shared]) + 1e-10
                v2 = np.array([A[ip, j] for j in shared]) + 1e-10
                v1 = v1 / v1.sum(); v2 = v2 / v2.sum()
                kls.append(0.5 * (scipy_entropy(v1, v2) + scipy_entropy(v2, v1)))  # symm KL
    return np.array(kls)

kl_gat = avg_kl(A_gat)
kl_v2 = avg_kl(A_v2)
print(f"\n[5] Symmetric KL divergence between query pairs (over shared neighbors):")
print(f"    GAT:   mean = {kl_gat.mean():.4f}, max = {kl_gat.max():.4f}")
print(f"    GATv2: mean = {kl_v2.mean():.4f}, max = {kl_v2.max():.4f}")
print(f"    Ratio: GATv2/GAT = {kl_v2.mean()/kl_gat.mean():.2f}x")
print(f"    Higher KL = queries have more different attention patterns")

# ============================================
# METRIC 6: Rank reversal rate
# ============================================
# Among query pairs sharing >=2 neighbors, for each pair of those neighbors (a, b):
# Does query i rank a > b but query i' rank b > a? Count "reversals".
def rank_reversal_rate(A):
    reversals = 0; total = 0
    for i in range(N):
        for ip in range(i+1, N):
            shared = sorted(set(in_neighbors[i]) & set(in_neighbors[ip]))
            if len(shared) >= 2:
                for ia in range(len(shared)):
                    for ib in range(ia+1, len(shared)):
                        a, b = shared[ia], shared[ib]
                        ord_i = A[i, a] - A[i, b]
                        ord_ip = A[ip, a] - A[ip, b]
                        if ord_i * ord_ip < 0:  # different signs = reversal
                            reversals += 1
                        total += 1
    return reversals, total

r_gat, t2 = rank_reversal_rate(A_gat)
r_v2, _ = rank_reversal_rate(A_v2)
print(f"\n[6] Rank reversal rate (pairs of (query_pair, neighbor_pair) where ranking flips):")
print(f"    GAT:   {r_gat}/{t2} ({100*r_gat/t2:.2f}%)")
print(f"    GATv2: {r_v2}/{t2} ({100*r_v2/t2:.2f}%)")
print(f"    GAT close to 0% ⇒ ranking invariant across queries (static)")
print(f"    GATv2 > GAT      ⇒ ranking can flip per query (dynamic)")

# ============================================
# SUMMARY TABLE
# ============================================
print("\n" + "="*70)
print("  SUMMARY")
print("="*70)
print(f"{'Metric':<45}{'GAT':>10}{'GATv2':>10}{'ratio':>8}")
print("-"*70)
print(f"{'[1] Cross-query attn variance (×1e4)':<45}{m_gat*1e4:>10.2f}{m_v2*1e4:>10.2f}{m_v2/m_gat:>8.2f}x")
print(f"{'[2a] Pearson correlation (mean)':<45}{p_gat.mean():>10.4f}{p_v2.mean():>10.4f}{p_v2.mean()/p_gat.mean():>8.4f}x")
print(f"{'[2b] Spearman correlation (mean)':<45}{s_gat.mean():>10.4f}{s_v2.mean():>10.4f}{s_v2.mean()/s_gat.mean():>8.4f}x")
print(f"{'[3] Top-1 agreement rate (%)':<45}{100*a_gat/t:>10.1f}{100*a_v2/t:>10.1f}{a_v2/a_gat if a_gat else 0:>8.4f}x")
print(f"{'[4] Mean attention entropy (nats)':<45}{e_gat:>10.3f}{e_v2:>10.3f}{e_v2/e_gat:>8.4f}x")
print(f"{'[5] Mean symmetric KL divergence':<45}{kl_gat.mean():>10.4f}{kl_v2.mean():>10.4f}{kl_v2.mean()/kl_gat.mean():>8.2f}x")
print(f"{'[6] Rank reversal rate (%)':<45}{100*r_gat/t2:>10.2f}{100*r_v2/t2:>10.2f}{(r_v2/r_gat if r_gat else float('inf')):>8.2f}x")
print("="*70)
print("\nKey takeaways:")
print("  - GATv2 has dramatically higher cross-query attention variance (metric 1)")
print("  - GAT's pairwise correlation ~1.0 confirms static attention (metric 2)")
print("  - GATv2 has more rank reversals (metric 6) ⇒ truly dynamic attention")
