from copy import deepcopy
import torch, pickle
import time, random
import numpy as np
from config.Params import configs
from env.workflow_scheduling_v3.lib.poissonSampling import sample_poisson_shape
import os, sys, argparse as _argparse

def _parse_resume_args():
    """Parse resume-related args without disturbing existing configs."""
    ap = _argparse.ArgumentParser(add_help=False)
    ap.add_argument('--resume', type=str, default=None)
    ap.add_argument('--checkpoint_dir', type=str, default='./checkpoints_default')
    ap.add_argument('--checkpoint_interval', type=int, default=50)
    known, _ = ap.parse_known_args()
    return known

def _save_checkpoint(path, i_update, algos, record, configs_obj):
    state = {
        'i_update': i_update,
        'actor_state_dict': algos.actor.state_dict(),
        'critic_state_dict': algos.critic.state_dict(),
        'optimizer_state_dict': algos.optimizer.state_dict() if hasattr(algos, 'optimizer') else None,
        'optimizer_actor_state_dict': algos.optimizer_actor.state_dict(),
        'optimizer_critic_state_dict': algos.optimizer_critic.state_dict(),
        'record': record,
        'pre_grad_max': algos.pre_grad_max,
        'entropy_count': algos.entropy_count,
        'grad_count': algos.grad_count,
        'np_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'python_random_state': random.getstate(),
    }
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)
    print(f'[ckpt] saved {path} at i_update={i_update}', flush=True)

def _load_checkpoint(path, algos):
    state = torch.load(path, map_location='cpu', weights_only=False)
    algos.actor.load_state_dict(state['actor_state_dict'])
    algos.critic.load_state_dict(state['critic_state_dict'])
    if state.get('optimizer_state_dict') is not None and hasattr(algos, 'optimizer'):
        algos.optimizer.load_state_dict(state['optimizer_state_dict'])
    algos.optimizer_actor.load_state_dict(state['optimizer_actor_state_dict'])
    algos.optimizer_critic.load_state_dict(state['optimizer_critic_state_dict'])
    algos.pre_grad_max = state['pre_grad_max']
    algos.entropy_count = state['entropy_count']
    algos.grad_count = state['grad_count']
    np.random.set_state(state['np_random_state'])
    torch.set_rng_state(state['torch_random_state'])
    random.setstate(state['python_random_state'])
    print(f"[ckpt] resumed from {path} at i_update={state['i_update']}, record={state['record']}", flush=True)
    return state['i_update'], state['record']

from joblib import Parallel, delayed

from policy.actor3 import PPO, BatchGraph, Memory, RolloutBuffer
from env.workflow_scheduling_v3.simulator_wf import WFEnv

device = torch.device(configs.device)
# file_writing_a = './logs/actor_log_' + str(configs.epochs_c) + '_' + str(configs.lr_c) + '_' + str(configs.window_steps) + '.npy'
# file_writing_c = './logs/critic_log_' + str(configs.epochs_c) + '_' + str(configs.lr_c) + '_' + str(configs.window_steps) + '.npy'
# file_writing_ = './logs/actor_log_' + str(configs.epochs_c) + '_' + str(configs.lr_c) + '_' + str(configs.window_steps) + '.pkl'
# file_writing_g = './logs/grad_log_' + str(configs.epochs_c) + '_' + str(configs.lr_c) + '_' + str(configs.window_steps) + '.pkl'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)    

def compute_returns( rewards):
    norm_reward_mb = [i/configs.normalize_rewards for i in rewards]
    dis_returns = []
    next_values = 0
    for step in reversed(range(len(rewards))):  
        next_values = norm_reward_mb[step] + configs.gamma * next_values
        dis_returns.insert(0, next_values)    
    return dis_returns


def compute_values(critic, bufferdata):
    start_idx = 0
    values = []
    while start_idx <  len(bufferdata):
        end_idx = min(start_idx + configs.batch_size, len(bufferdata))
        batch_states = BatchGraph(configs.normalize).batch_process(bufferdata[start_idx:end_idx]) 
        vals = critic(state_wf = batch_states.wf_features,      ## (batch_szie^2, 2)             
                        edge_index_wf = batch_states.wf_edges,       
                        mask_wf = batch_states.wf_masks,
                        batch_wf = batch_states.wf_batchs,
                        candidate_task_index = batch_states.candidate_taskID,
                        deterministic = False) 
        values.extend(vals.tolist())
        start_idx += configs.batch_size    
    return values


def collect_whole_rollouts( policy, args, k):   
    # terminated = False
    args.GENindex = k
    args.indEVALindex = 0
    env = WFEnv(args.env_name, args, True)
    state_list = env.reset()    
    memory = [[],[],[],[],[],[]]    # s,a,r,d,l,vs
    batch_states = BatchGraph(args.normalize)
    v_states = BatchGraph(args.normalize)
    returns = 0           ## 使用policy与环境交互
    while True:
        with torch.no_grad():
            v_states.form_v_state(*state_list)  
            batch_states.wrapper(*state_list)   
            action, prob, _ = policy(state_wf = batch_states.wf_features,   
                                state_vm = batch_states.vm_features,                 
                                edge_index_wf = batch_states.wf_edges,    
                                edge_index_vm = batch_states.vm_edges,   
                                mask_wf = batch_states.wf_masks,
                                mask_vm = batch_states.vm_masks,
                                batch_wf = batch_states.wf_batchs,
                                batch_vm = batch_states.vm_batchs,
                                candidate_task_index = batch_states.candidate_taskID,
                                deterministic = False)

        state_list, reward, done = env.step(action.item())  
        append_to_nested_lists(memory, [deepcopy(batch_states), action, reward, done, deepcopy(prob), deepcopy(v_states)])
        returns += reward

        if done:
            break

    return memory, np.mean(env.all_flowTime)  

def validation_Org(i, args, model, deterministic):

    en_loss = []
    args.GENindex = 0
    args.indEVALindex = i
    envs = WFEnv(args.env_name, args, False)
    state_list = envs.reset()
    batch_states = BatchGraph(args.normalize)
    while True:
        with torch.no_grad():
            batch_states.wrapper(*state_list)   # 处理(s,a)
            action, _, entropy = model(state_wf = batch_states.wf_features,      ## (batch_szie^2, 2)
                                state_vm = batch_states.vm_features,                 
                                edge_index_wf = batch_states.wf_edges,    
                                edge_index_vm = batch_states.vm_edges,   
                                mask_wf = batch_states.wf_masks,
                                mask_vm = batch_states.vm_masks,
                                batch_wf = batch_states.wf_batchs,
                                batch_vm = batch_states.vm_batchs,
                                candidate_task_index = batch_states.candidate_taskID,
                                deterministic = deterministic)
        en_loss.append(entropy.item())
        state_list, _, done = envs.step(action.item())  
        if done:  
            return np.mean(envs.all_flowTime), np.mean(en_loss)  

def validation_H(i, args):
    set_seed(args.env_seed)
    args.GENindex = 0
    args.indEVALindex = i
    env = WFEnv(args.env_name, args, False)
    state_list = env.resetGP()
    while True:
        action = env.HEFT()
        state_list, _, done = env.stepGP(action)   ## 比env.rest()多了reward, done
        if done:  
            return np.mean(env.all_flowTime)   

def collect_parallel(i_update, actor, critic):
    memory, returni = collect_whole_rollouts(actor, configs, i_update)#i)
    temp_memory = Memory(len(memory[0]))
    temp_memory.update(*memory)
    if configs.gae_lambda == 1:        
        period_returns = compute_returns(temp_memory.reward_mb)
        buffer = RolloutBuffer(temp_memory, period_returns, period_returns, len(period_returns))
    else:
        temp_vals = compute_values(critic, temp_memory.v_state_mb)
        period_returns, period_adv = temp_memory.compute_gae(temp_vals)
        buffer = RolloutBuffer(temp_memory, period_returns, period_adv, len(period_returns))    
    return returni,buffer

def append_to_nested_lists(nested_lists, new_lists):
    for i in range(len(nested_lists)):
        nested_lists[i].append(new_lists[i])

def extend_to_nested_lists(nested_lists, new_lists):
    for i in range(len(nested_lists)):
        nested_lists[i].extend(new_lists[i])

def main():

    # Load dataset
    record = 1e10
    wf_types=4  
    set_seed(configs.env_seed)

    configs.valid_dataset = np.load('./validation_data/validation_instance_2024.npy').reshape((1,-1, configs.wf_num)) [:, :(configs.valid_num)]
    configs.GENindex = 0
    configs.indEVALindex = 0
    configs.arr_times = sample_poisson_shape(configs.arr_rate, configs.valid_dataset.shape)
    configs.train_dataset = np.random.randint(0,wf_types,(configs.max_updates+configs.warmup_critic+1, 1, configs.wf_num ))    
    configs.arr_times_train = sample_poisson_shape(configs.arr_rate, configs.train_dataset.shape)

    # parallel HEFT validation
    meanFlowTimes = Parallel(n_jobs=-1)(delayed(validation_H)(t, configs) for t in range(configs.valid_num))

    t1 = time.time()
    print('Vlidation at HEFT: mean_flowtime_deterministic: {:.6f}+/-{:.6f}\t time_elapsed: {:.6f}'.\
            format(np.mean(meanFlowTimes), np.std(meanFlowTimes), (t1 - total1)/3600), flush=True)    

    # Build policy
    set_seed(configs.algo_seed)
    algos = PPO(input_dim_wf = configs.input_dim_wf,
                    input_dim_vm= configs.input_dim_vm,           
                    hidden_dim= configs.hidden_dim,
                    c_hidden_dim= configs.c_hidden_dim,
                    gnn_layers= configs.gnn_layers,
                    atten_layers = configs.atten_layers,  
                    mlp_layers= configs.mlp_layers,                                                                               
                    heads= configs.heads,
                    dropout= configs.dropout,     
                    activate_fn = configs.activate_fn,   
                    )
    algos.actor.load_state_dict(torch.load('./validation_data/step1/actors/a_{}_{}_{}.pth'.format(configs.vm_types, configs.each_vm_type_num, configs.arr_rate),\
                                            map_location=torch.device(device),weights_only=True))
    # NEW: KL anchor reference actor (frozen copy of step1 imitation)
    if configs.beta_kl > 0:
        import copy as _copy
        algos.ref_actor = _copy.deepcopy(algos.actor)
        for _p in algos.ref_actor.parameters():
            _p.requires_grad_(False)
        algos.ref_actor.eval()
        print(f"[kl-anchor] ref_actor created, beta_kl={configs.beta_kl}", flush=True)
    algos.optimizer = torch.optim.Adam([
                        {'params': algos.actor.parameters(), 'lr': configs.lr_a},
                        {'params': algos.critic.parameters(), 'lr': configs.lr_c}])

    # non-parallel
    # parallel update-0 validation
    valids = Parallel(n_jobs=-1)(delayed(validation_Org)(t, configs, algos.actor, True) for t in range(configs.valid_num))
    valids = np.array(valids)
    meanFlowTimes, meanEntropies = valids[:,0], valids[:,1]

    if np.mean(meanFlowTimes) < record:
        record = deepcopy(np.mean(meanFlowTimes))
    t1 = time.time()
    print('Vlidation at update-{}: mean_flowtime_deterministic: {:.6f}+/-{:.6f}\t mean_Entropy: {:.6f}+/-{:.6f}\t record: {:.6f}\t time_elapsed: {:.6f}'.\
            format(str(0), np.mean(meanFlowTimes), np.std(meanFlowTimes), np.mean(meanEntropies), np.std(meanEntropies) ,record, (t1 - total1)/3600), flush=True)

    # Training loop
    log = []
    # aloss_log = [[], [], []]
    algos.pre_grad_max = 0
    
    # === Checkpoint resume logic ===
    _ckpt_dir = configs.checkpoint_dir
    _ckpt_interval = configs.checkpoint_interval
    os.makedirs(_ckpt_dir, exist_ok=True)
    _start_iter = 0
    if configs.resume and os.path.exists(configs.resume):
        _start_iter, record = _load_checkpoint(configs.resume, algos)
        _start_iter += 1   # next iter to run

    for i_update in range(_start_iter, configs.max_updates+configs.warmup_critic):

        algos.update_idx = i_update+1 

        # parallel (rollout only - Plan B)
        outputs = Parallel(n_jobs=configs.num_envs)(delayed(collect_parallel)(i_update, algos.actor, algos.critic) for _ in range(configs.num_envs))
        ep_returns = [item[0] for item in outputs]
        memories = [item[1] for item in outputs]

        new_buffer = memories[0]
        new_buffer.merge(memories[1:])
        allloss, grad_changes = algos.train(new_buffer)
        # extend_to_nested_lists(aloss_log, allloss)

        t1 = time.time()
        log.append([i_update+1, np.mean(ep_returns)]+ [np.mean(allloss[s1]) for s1 in range(len(allloss))]+
                    [np.mean(grad_changes[s2]) for s2 in range(len(grad_changes))] + [(t1-total1)/3600])
        # with open(file_writing_, 'wb') as f:
        #     pickle.dump(aloss_log, f)
        print('Episode-{}: ep_meanFlowTime: {:.6f}\t all_loss: {:.6f}\t p_loss: {:.6f}\t e_loss: {:.6f}\t v_loss: {:.6f}\t v_mre: {:.6f}\t grad_changes: {:.6f}\t time_elapsed: {:.4f}'.format(*log[-1]), flush=True)        
        # with open('./logs/train_log_{}-{}-{}.txt'.format(configs.vm_types, configs.each_vm_type_num ,configs.arr_rate), 'w') as f:
        #     for entry in log:
        #         f.write(str(entry) + '\n')

        if (i_update + 1) % configs.log_interval == 0:

            # parallel per-update validation
            valids = Parallel(n_jobs=-1)(delayed(validation_Org)(t, configs, algos.actor, True) for t in range(configs.valid_num))
            valids = np.array(valids)
            meanFlowTimes, meanEntropies = valids[:,0], valids[:,1]

            if np.mean(meanFlowTimes) < record:
                torch.save(algos.actor.state_dict(), './logs/a_{}_{}_{}.pth'.format(configs.vm_types, configs.each_vm_type_num, configs.arr_rate) )
                torch.save(algos.critic.state_dict(), './logs/c_{}_{}_{}.pth'.format(configs.vm_types, configs.each_vm_type_num, configs.arr_rate) )        
                record = deepcopy(np.mean(meanFlowTimes))
            t1 = time.time()
            print('Vlidation at update-{}: mean_flowtime_deterministic: {:.6f}+/-{:.6f}\t mean_Entropy: {:.6f}+/-{:.6f}\t record: {:.6f}\t time_elapsed: {:.4f}'.\
                    format(str(i_update+1), np.mean(meanFlowTimes), np.std(meanFlowTimes), np.mean(meanEntropies), np.std(meanEntropies), record, (t1 - total1)/3600), flush=True)

            # === Periodic checkpoint save ===
            if (i_update + 1) % _ckpt_interval == 0 or (i_update + 1) >= configs.max_updates + configs.warmup_critic:
                _save_checkpoint(os.path.join(_ckpt_dir, 'state_latest.pth'),
                                 i_update, algos, record, configs)
                _save_checkpoint(os.path.join(_ckpt_dir, f'state_{i_update+1}.pth'),
                                 i_update, algos, record, configs)

    # with open(file_writing_, 'wb') as f:
    #     pickle.dump(aloss_log, f)

if __name__ == '__main__':
    total1 = time.time()
    main()
    total2 = time.time()
    print('>>>Overall Runtime is ', (total2 - total1)/3600, ' hours', flush=True)