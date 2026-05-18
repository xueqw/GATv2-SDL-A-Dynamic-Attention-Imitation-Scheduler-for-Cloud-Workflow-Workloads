#!/usr/bin/env python3
"""Generate GPHH memory (state, action pairs) for step1 imitation.

Uses env.reset()/step() to get 9-tuple actor-format states (compatible with step1.py),
but uses env.gp_feature_construct() to extract 6-feature matrix for GP tree decision.

Outputs pickle in same format as HEFT_memory_500_*.pkl so step1.py can swap teachers.
"""
import argparse, os, sys, pickle, time, random
import numpy as np
import operator
from functools import partial

# Add project root
sys.path.insert(0, '/home/xue/GOODRL')

from config.Params import configs
from env.workflow_scheduling_v3.simulator_wf import WFEnv
from env.workflow_scheduling_v3.lib.poissonSampling import sample_poisson_shape
from deap import base, creator, tools, gp


def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


def setup_pset():
    pset = gp.PrimitiveSet("main1", 6)
    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(np.minimum, 2)
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(protected_div, 2, name='div')
    pset.addEphemeralConstant("rand101", partial(random.randint, -1, 1))
    pset.renameArguments(ARG0='TS', ARG1='RW', ARG2='ET',
                         ARG3='FT', ARG4='CU', ARG5='UL')
    if not hasattr(creator, 'FitnessMin'):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    return pset


def gp_pick_action(rule, gp_features, normalize_features=None):
    """Pick action using GP tree on 6-feature matrix (vmNum, 6)."""
    obs = gp_features
    if normalize_features is not None:
        obs = obs / np.array(normalize_features[1:])
    priorities = rule(*obs.T)
    if isinstance(priorities, int) or priorities.ndim == 0:
        priorities = np.full((len(obs),), 1, dtype=np.float64)
    min_indices = np.where(priorities == np.min(priorities))[0]
    return int(np.random.choice(min_indices))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gp_tree', required=True)
    ap.add_argument('--num_instances', type=int, default=500)
    ap.add_argument('--out', required=True)
    args, _ = ap.parse_known_args()  # let configs (from Params.py) own the rest

    random.seed(configs.env_seed)
    np.random.seed(configs.env_seed)

    # Set up datasets like step1 main() does
    wf_types = 4
    configs.train_dataset = np.random.randint(0, wf_types, (args.num_instances+1, configs.num_envs, configs.wf_num))
    configs.arr_times_train = sample_poisson_shape(configs.arr_rate, configs.train_dataset.shape)
    configs.valid_dataset = np.load('/home/xue/GOODRL/validation_data/validation_instance_2024.npy').reshape((1,-1,configs.wf_num))[:, :configs.valid_num]
    configs.arr_times = sample_poisson_shape(configs.arr_rate, configs.valid_dataset.shape)
    configs.GENindex = 0
    configs.indEVALindex = 0

    # Load GP tree
    pset = setup_pset()
    toolbox = base.Toolbox()
    toolbox.register("compile", gp.compile, pset=pset)
    tree = pickle.load(open(args.gp_tree, 'rb'))
    rule = toolbox.compile(expr=tree)
    print(f'[GPHH-mem] Loaded GP tree from {args.gp_tree}', flush=True)
    print(f'[GPHH-mem] GP tree: {tree}', flush=True)
    print(f'[GPHH-mem] Generating {args.num_instances} training trajectories...', flush=True)

    all_states = []     # list of 9-tuples
    all_actions = []    # list of ints
    all_rewards = []
    all_dones = []
    all_episode_flowtimes = []

    t0 = time.time()
    for inst_idx in range(args.num_instances):
        configs.GENindex = inst_idx
        env = WFEnv(configs.env_name, configs, True)  # True = train mode

        # Use reset() to get 9-tuple (actor format)
        state_9tuple = env.reset()

        ep_count = 0
        while True:
            # Extract GP-format features (6-col matrix per candidate)
            gp_feats = env.gp_feature_construct()

            # Pick action via GP tree
            action = gp_pick_action(rule, gp_feats,
                                    normalize_features=configs.normalize_features if configs.normalize else None)

            # Record (state_9tuple, action) pair for imitation
            all_states.append(state_9tuple)
            all_actions.append(action)
            all_rewards.append(0)

            # Step env with that action
            result = env.step(action)
            state_9tuple, reward, done = result[0], result[1], result[2]
            ep_count += 1

            if done:
                all_dones.append(True)
                # Record episode-level metric
                if hasattr(env, 'all_flowTime'):
                    all_episode_flowtimes.append(np.mean(env.all_flowTime))
                else:
                    all_episode_flowtimes.append(0)
                break
            else:
                all_dones.append(False)

        if (inst_idx+1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (inst_idx+1) / elapsed
            eta = (args.num_instances - inst_idx - 1) / rate
            recent_mean = np.mean(all_episode_flowtimes[-20:])
            print(f'[GPHH-mem] inst {inst_idx+1}/{args.num_instances}, '
                  f'samples={len(all_states)}, '
                  f'last20 mean flowtime={recent_mean:.1f}, '
                  f'elapsed={elapsed:.0f}s, ETA={eta:.0f}s', flush=True)

    # Save in same format as HEFT memory: 5-element list
    out_data = [all_states, all_actions, all_rewards, all_dones, all_episode_flowtimes]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(out_data, f)
    print(f'[GPHH-mem] Saved {len(all_states)} (state,action) pairs to {args.out}', flush=True)
    print(f'[GPHH-mem] Mean episode flowtime: {np.mean(all_episode_flowtimes):.2f}', flush=True)


if __name__ == '__main__':
    main()
