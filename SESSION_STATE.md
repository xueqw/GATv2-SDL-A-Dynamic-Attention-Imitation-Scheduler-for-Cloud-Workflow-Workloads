# Session State Snapshot (2026-05-17)

## 当前正在跑的实验

**GPHH-teacher pipeline** (Phase 4 即将启动):
- Chain watcher: `bash /home/xue/GOODRL/tools/chain_gphh_pipeline.sh`
- 当前阶段: step1 imitation with `--teacher gphh`, update-7 / 10
- Run dir: `/home/xue/GOODRL/runs/20260517_gphh_teacher_pipeline/`

## 已确定的关键数字

| 配置 | step1 record | step2 record |
|---|---|---|
| HEFT teacher (baseline GATv2) | **421.0** | 417.7 (我们) / 411 (paper) |
| GPHH teacher (current run) | **412.56** ✓ | TBD (Phase 4 即将跑) |
| Pointer head (failed) | 468 | 发散到 8000+ |
| Actor SelfAttention (failed) | 893 | crash |

## 实验路径决策树（已走过）

1. ✓ GAT → GATv2 (在 actor3.py): step1 from 429 → 421
2. ✓ 修 critic-not-updating bug (--grad_control 0)
3. ✗ Actor SelfAttention probe (broke imitation)
4. ✗ Pointer Attention scoring head (diverged in PPO)
5. ✓ **GPHH-as-teacher** (current focus, looking promising)

## 代码改动汇总

```
policy/actor3.py:
  - GATConv → GATv2Conv (line 323)
  - 加了 PointerHead 类 (line ~412) [disabled by default]
  - 加了 SelfAttention 选项到 Actor [disabled by default]
  - PPO.__init__ 加了 entropy_count/grad_count/pre_grad_max init
  - 备份: actor3.py.gat_backup, actor3.py.bak_pre_actorAttn,
          actor3.py.bak_pre_pointer

config/Params.py:
  - --resume, --checkpoint_dir, --checkpoint_interval (resume 系统)
  - --actor_atten_layers (0=off)
  - --actor_pointer (0=off)
  - --teacher {heft, gphh}
  - --gp_tree, --num_instances, --out (memory generation)

step1.py:
  - 加了 --teacher gphh 支持: 加载 GPHH_memory 而非 HEFT_memory
  - 备份: step1.py.bak_pre_teacher, step1.py.bak_serial

step2.py:
  - 加了 checkpoint resume 逻辑
  - 4 处 Parallel 全部启用（HEFT val, update-0 val, rollout, periodic val）
  - 备份: step2.py.bak_no_resume, step2.py.bak_serial, step2.py.bak_planB

mainGP.py:
  - 加了 GP_WORKERS 环境变量控制 (默认 32)
  - 备份: mainGP.py.bak_pre_limit
```

## 新建的工具

```
tools/run_with_resume.sh           # step2 watchdog + auto-resume
tools/analyze.py                   # 解析 log + 画曲线
tools/generate_gphh_memory.py      # 用 GP tree 生成 (state,action) pairs
tools/chain_gphh_pipeline.sh       # GPHH 训完后自动接力 Phase 2-4
tools/log_mirror.sh                # 每 30s cp /tmp/log → runs/log
```

## 已归档的 run dirs

```
runs/
├── 20260513_gatv2_5x5_gc1_seed4_baseline/    # critic bug 复现
├── 20260513_gatv2_5x5_gc0_seed4_lr3e-4/      # gc=0 修复但 actor 退步
├── 20260514_gatv2_5x5_tuned_seed3_*/         # 多次 PPO 尝试
├── 20260515_tuned_GATv2_full/                # 22h Plan B 跑完, record 417.7
├── 20260516_pointer_full/                    # Pointer 失败 (validation 8000+)
├── 20260516_actor_attn_full/                 # SelfAttention 失败 (893)
├── 20260516_gphh_train/                      # GPHH 训练日志
└── 20260517_gphh_teacher_pipeline/           # ★ 当前活跃
```

## 论文方向

**主线（确认）**: GPHH multi-teacher imitation
- 故事: HEFT 是 paper 84% 贡献，但是它本身的限制。换 GPHH 当 teacher 突破上限。
- 数据点: step1 from 421 (HEFT) → 412.56 (GPHH), 改善 -8.4
- 待证: step2 PPO 能否再压到 < 410

**次线**: Negative results 作为附录
- SelfAttention probe failure (信息融合破坏 candidate identity)
- Pointer Attention divergence (tanh scaling 不友好)
- 用于 motivation 章节

**潜在第三线**: heterogeneous graph for scaling (25x → 1x cost)
- 未实现，paper 副菜

## 服务器连接信息

```
当前 IP:   10.36.3.199 (WKU 校园网, 动态)
之前用过:  192.168.1.172 (家里 WiFi)
SSH:       xue@<IP>, password = "xue"
工作目录:  /home/xue/GOODRL/
Python env: /home/xue/anaconda3/envs/drl_sched/bin/python
```

## 已知问题

- WiFi 经常掉线 → IP 变 → 需要用户去机器查新 IP
- joblib + pandas fork 有 SIGSEGV 风险（但目前 RAM 换了之后好多了）
- step1 imitation 训练慢（GPHH memory 427K 样本，比 HEFT 14K 大 30x）

## 下一个 wakeup 任务

设了 19:04 自动 ping 一次 step1 with GPHH teacher 结果。但因为 step1 比预期慢，那时只到 update-7。等到约 22:00 才跑完。

## 在新会话中怎么恢复上下文

新 Claude 会话开始时：
1. 用户说 "读 /home/xue/GOODRL/SESSION_STATE.md"
2. SSH 到机器 (IP 可能要查) 读这个文件
3. 看 `runs/20260517_gphh_teacher_pipeline/pipeline.log` 当前状态
4. 继续工作
