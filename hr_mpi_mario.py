import numpy as np
import gym
import time
from mpi4py import MPI
import cv2
from ludus.utils import discount_rewards
import heapq
import tensorflow as tf
from model import HASL
import multiprocessing
import pickle
from sklearn.neighbors import LSHForest

# Super Mario stuff
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

OBS_DIM = 42

def init_logger(lp):
    global log_path
    log_path = lp

    f = open(log_path, 'w+')
    f.close()

def log(string):
    with open(log_path, 'a') as f:
        f.write(string + '\n')

def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
    return env

def filter_obs(obs, obs_shape=(OBS_DIM, OBS_DIM)):
    obs = cv2.resize(obs, obs_shape, interpolation=cv2.INTER_LINEAR)
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    return obs / 255
    
def worker(action_sets, hasl, max_steps=1000):
    """
    Performs the game simulation, and is called across all processes
    """
    train_data = []
    full_data = []
    env = make_env()
    obs = env.reset()
    obs = filter_obs(obs)
    encoded_obs = hasl.apply_encoder(obs.reshape(1, OBS_DIM, OBS_DIM, 1))
    
    ep_reward = 0
    step = 0
    while step < max_steps:
        # act_idx = np.random.randint(len(action_sets))
        # act_set = action_sets[act_idx]
        act_set = [hasl.choose_action(encoded_obs)]
        
        train_data.append([encoded_obs])
        step_reward = 0
        for act in act_set:
            obs_p, r, d, _ = env.step(act)
            obs_p = filter_obs(obs_p)
            full_data.append([obs, act, r, obs_p])
            obs = obs_p
            step_reward += r
            step += 1
            if d or step >= max_steps:
                break
        ep_reward += step_reward
        
        encoded_obs = hasl.apply_encoder(obs.reshape(1, OBS_DIM, OBS_DIM, 1))
        train_data[-1].extend([act_set, step_reward, encoded_obs])
        
        if d:
            break
    
    train_data = np.asarray(train_data)
    full_data = np.asarray(full_data)

    return train_data, full_data

def find_neighbors(samples, all_data, n=500):
    lshf = LSHForest(n_estimators=20, n_candidates=200,
                 n_neighbors=n).fit(all_data)

    neighbors = []
    for sample in samples:
        neighbors.append([])
        sn = lshf.kneighbors(np.array(sample).reshape(1, -1))[1][0]
        for n in sn:
            neighbors[-1].append(all_data[n])

    return np.asarray(neighbors)

if __name__ == '__main__':
    ### Setp for MPI ###
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_processes = comm.Get_size()
    controller = 0

    ### Define starting parameters ###
    n_epochs = 1000
    n_train_batches = 8
    n_process_batches = int(n_train_batches / n_processes)
    top_frac = 0.1
    top_x = int(np.ceil(n_processes * top_frac))
    act_top_x = 2
    min_branch, max_branch = 2, 3
    log_freq = 60
    n_as_proposals = 5
    train_act_sets = [[i] for i in range(0, 7)]

    init_logger('progress.log')

    if rank == controller:
        device_config = tf.ConfigProto()
        use_device = tf.device('/cpu:0')
    else:
        device_config = tf.ConfigProto(device_count={'GPU': 0})
        # n_cpus = multiprocessing.cpu_count()
        # device_config = tf.ConfigProto(device_count={'GPU': 0, 'CPU': n_cpus},
        #                 inter_op_parallelism_threads=n_cpus,
        #                 intra_op_parallelism_threads=1,
        #                 log_device_placement=True)
        # print(rank % n_cpus)
        # use_device = tf.device(f'/cpu:{rank % n_cpus}')

    # with use_device:
    hasl = HASL(comm, controller, rank, sess_config=device_config)

    for epoch in range(1, n_epochs+1):
        train_data = []
        encoder_data = []
        for _ in range(n_process_batches):
            ### Simulate more episodes to gain training data ###
            if rank == controller:
                all_data = comm.gather(worker(train_act_sets, hasl), controller)
                new_train_data = [x[0] for x in all_data]
                train_data.extend(new_train_data)
                encoder_data.extend([x[1] for x in all_data])
            else:
                comm.gather(worker(train_act_sets, hasl), controller)

        if rank == controller:
            encoder_data = np.concatenate(encoder_data)
            cat_train_data = np.concatenate(train_data)
            
        if rank == controller:
            print(f'----- Epoch {epoch} -----')

            if epoch % log_freq == 0:
                log(f'Epoch {epoch} train action sets:')
                log(str(train_act_sets) + '\n')

            ### Train reverse dynamics encoder model ###
            assert encoder_data.shape[1] == 4
            train_states = encoder_data[:, 0]
            train_actions = encoder_data[:, 1]
            train_state_ps = encoder_data[:, 3]

            accuracy = hasl.train_encoder(train_states, train_state_ps, train_actions)
            print(f'Inverse Dynamics Accuracy: {str(accuracy*100)[:5]}%')

            # TODO: Change the way the actions are passed in to train the policy
            hasl.train_policy(cat_train_data[:,0], cat_train_data[:,1], cat_train_data[:,2])

            ### Pull rewards and action sequences from the training data ###
            reward_list = []
            # action_seq_list = []
            for i in range(len(train_data)):
                reward_list.append(sum(train_data[i][:,2]))
                # action_seq_list.append(train_data[i][:,1])

            print(f'Avg Reward: {np.mean(reward_list)}, Min: {np.min(reward_list)}, Max: {np.max(reward_list)}, Std: {np.std(reward_list)}')

            ### Calculate state differences ###
            state_changes = []
            start_states = []
            act_seqs = []
            ss = []
            seq_len = 3
            for ep in range(len(train_data)):
                real_step = 0
                for step in range(seq_len, len(train_data[ep])):
                    real_step += len(train_data[ep][step][1])
                    ss.append([real_step, train_data[ep][step][0]])
                    # state_changes.append([real_step, train_data[ep][step][0] - train_data[ep][step-seq_len][0]])
                    state_changes.append(train_data[ep][step][0] - train_data[ep][step-seq_len][0])
                    start_states.append(train_data[ep][step-seq_len][0])
                    act_seqs.append(train_data[ep][step-seq_len:step,1])

            state_changes = np.asarray(state_changes).squeeze()
            start_states = np.asarray(start_states).squeeze()
            act_seqs = np.asarray(act_seqs)
            print(state_changes.shape, start_states.shape, act_seqs.shape)

            # print(f'Count: {len(state_changes)}')
            # with open('state_changes.pickle', 'wb') as f:
            #     pickle.dump(state_changes, f)

            # with open('ss.pickle', 'wb') as f:
            #     pickle.dump(ss, f)

            ### Use softmax on rewards to stochastically choose which episodes to pull actions from ###
            max_reward = max(reward_list)
            scaled_rewards = [r - max_reward for r in reward_list]
            reward_sum = sum(scaled_rewards)
            scaled_rewards = [max(r / reward_sum, 1e-9) for r in scaled_rewards]

            selected_ids = np.random.choice(range(len(train_data)), size=top_x, replace=False, p=scaled_rewards)
            top_data = [train_data[idx] for idx in selected_ids]

            ### Gather and format data for action sequence proposals ###

            state_change_samples = []
            for i in range(n_as_proposals):
                state_change_samples.append(state_changes[np.random.randint(len(state_changes))])

            as_net_train_data = find_neighbors(state_change_samples, state_changes)

            ### Count all the actions of specified sizes from the chosen episodes ###
            strain_act_sets = set([tuple(x) for x in train_act_sets])
            branch_dicts = {}
            for seq_len in range(min_branch, max_branch+1):
                count_dict = {}
                for episode in top_data:
                    # TODO: Change the way acts work after I make the seq nets pipeline
                    ep_acts = episode[:,1]
                    for step_idx in range(seq_len-1, len(ep_acts)):
                        new_act_set = tuple(np.concatenate(ep_acts[step_idx-seq_len+1:step_idx+1]))
                        if tuple(new_act_set) not in strain_act_sets:
                            if new_act_set in count_dict:
                                count_dict[new_act_set] += 1
                            else:
                                count_dict[new_act_set] = 1
                
                branch_dicts[seq_len] = count_dict

            ### Choose the new action sequences to be added to be used ###
            top_acts = []
            for n_branch in range(min_branch, max_branch+1):
                top_acts.extend([list(x[0]) for x in heapq.nlargest(act_top_x, list(branch_dicts[n_branch].items()), key=lambda x: x[1])])
                
            for act in top_acts:
                train_act_sets.append(act)

            ### Send the new action sequences to each process and sync models ###
            comm.bcast(train_act_sets, controller)
            hasl.sync_weights()
        else:
            ### Incorporate the new action sequences into its list for further training and sync models ###
            train_act_sets = comm.bcast(None, controller)
            hasl.sync_weights()

    if rank == 0:
        print('done')