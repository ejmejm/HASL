import argparse
import numpy as np
import gc
import gym
import time
import os
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

OBS_DIM = 64 # Must be a multiple of 2^4 with the current mdoel
OBS_DEPTH = 3

class ObsStack():
    def __init__(self, obs_shape, stack_size=4, fill_first=True):
        """
        Stack of observations that only keeps n observations at a time.

        Args:
            obs_shape (tuple): Shape of individual observations inserted
                into the stack.
            stack_size (int): Max number of observations that can be pushed
                into the stack before the oldest observations get replaced.
            fill_first (bool): The stack is always initialized to all zeros.
                When this variable is true, the very first push of an observation
                will be pushed `stack_size` times to fill the stack.

        """
        self.expected_shape = tuple(obs_shape)
        self.stack_size = stack_size
        self.stack = np.zeros(shape=list(obs_shape)+[stack_size])
        self.fill_first = True

    def push(self, obs):
        """
        Pushes a new state onto the stack at index 0.
        """
        assert obs.shape == self.expected_shape
        assert type(obs) == list or type(obs) == np.ndarray

        if type(obs) == list:
            obs = np.asarray(obs)

        self.stack[..., 1:] = self.stack[..., :-1]
        self.stack[..., 0] = obs

        if self.fill_first:
            self.fill_first = False
            for _ in range(self.stack_size - 1):
                self.push(obs)

    def get_stack(self):
        return self.stack.copy()

    def get_flat_stack(self):
        return self.stack.copy().reshape(-1)


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


def worker(action_sets, hasl, max_steps=1000, act_epsilon=0.1, obs_stack_size=4):
    """
    Performs the game simulation, and is called across all processes.

    Returns:
        train_data (np.ndarray): Array with the format, [[enc_ obs_stack, act_set, step_reward, encoded_obs], ...].
            The array contains an entry for every high-level step.
            enc_obs_stack: Encoded stack of n observations, where the shape = output shape of encoder.
            act_set: `list` of low-level actions taken in the high-level step.
            step_reward: Reward gained over the entire high-level step after being discounted.
            encoded_obs: Resulting, encoded state from taking act_set actions given obs_stack.
            
        full_data (np.ndarray): Array with the format, [[obs, act, r, obs_p], ...].
            The array contains an entry for every low-level step.
            obs: Filtered (but not encoded) observation at the beginning of the step.
            act: `int` representing the low-level action taken over the step.
            r: Reward gained for the low-level step after being discounted.
            obs_p: Resulting, fitlered (but not encoded) state from taking act in state obs.

        rewards (np.ndarray): Array containing the low-level rewards at each step.
    """
    train_data = []
    full_data = []
    env = make_env()

    obs = env.reset()
    obs = filter_obs(obs)
    h_obs_stack = ObsStack(obs.shape, obs_stack_size) # High-level obs stack
    h_obs_stack.push(obs)
    enc_full_obs = hasl.apply_encoder([h_obs_stack.get_stack()])

    l_obs_stack = ObsStack(obs.shape, obs_stack_size) # Low-level obs stack
    l_obs_stack.push(obs)
    
    ep_reward = 0
    step = 0
    while step < max_steps:
        # act_idx = np.random.randint(len(action_sets))
        # act_set = action_sets[act_idx]

        # TODO: The method for generating act_sets seems strange
        act_set = [hasl.choose_action(enc_full_obs, epsilon=act_epsilon)]
        if act_set[0] > 11:
            print(f'AAA: {act_set[0]}')
            act_set[0] = 0

        train_data.append([enc_full_obs])
        step_reward = 0
        for act in act_set:
            full_data.append([l_obs_stack.get_stack()])
            obs_p, r, d, _ = env.step(act)
            obs_p = filter_obs(obs_p)
            l_obs_stack.push(obs_p)
            full_data[-1].extend([act, r, l_obs_stack.get_stack()])
            step_reward += r
            step += 1
            if d or step >= max_steps:
                break
        ep_reward += step_reward

        h_obs_stack.push(obs_p)
        enc_full_obs = hasl.apply_encoder([h_obs_stack.get_stack()])
        train_data[-1].extend([act_set, step_reward, enc_full_obs])

        if d or step >= max_steps:
            break

    train_data = np.asarray(train_data)
    full_data = np.asarray(full_data)

    rewards = np.copy(full_data[:, 2])

    # Discount rewards
    train_data[:, 2] = discount_rewards(train_data[:, 2])
    full_data[:, 2] = discount_rewards(full_data[:, 2])

    return train_data, full_data, rewards

def find_neighbors(samples, all_data, n=500, return_idx=True):
    lshf = LSHForest(n_estimators=20, n_candidates=200,
                     n_neighbors=n).fit(all_data)

    neighbors = []
    for sample in samples:
        neighbors.append([])
        sn = lshf.kneighbors(np.array(sample).reshape(1, -1))[1][0]
        for n in sn:
            if return_idx:
                neighbors[-1].append(n)
            else:
                neighbors[-1].append(all_data[n])

    return np.asarray(neighbors)

### List Arguments ###

parser = argparse.ArgumentParser()

parser.add_argument('-ne', '--n_epochs', dest='n_epochs', type=int,
    default=1000, help='Number of overall training epochs')
parser.add_argument('-ms', '--max_rollout_steps', dest='max_rollout_steps', type=int,
    default=1000, help='Max number of steps in a single rollout')
parser.add_argument('-nr', '--n_rollouts', dest='n_rollouts', type=int,
    default=4, help='Number of rollouts per epoch.')
parser.add_argument('-nar', '--n_asn_rollouts', dest='n_asn_rollouts', type=int,
    default=32, help='Number of rollouts used for training an ASN')
parser.add_argument('-nag', '--n_asn_gen', dest='n_asn_gen', type=int,
    default=5, help='Number of ASNs created in one training cycle')
parser.add_argument('-lf', '--log_freq', dest='log_freq', type=int,
    default=10, help='Number of epochs between logging (currently unused)')
parser.add_argument('-apd', '--asn_proposal_delay', dest='asn_proposal_delay', type=int,
    default=8, help='Number of epochs in between ASN generation')
parser.add_argument('-nas', '--n_asn_train_samples', dest='n_asn_train_samples', type=int,
    default=512, help='Number of smaples drawn to train an individual ASN')
parser.add_argument('-b', '--act_branch_factor', dest='act_branch_factor', type=int,
    default=3, help='Number of actions each ASN should execute')
parser.add_argument('-ee', '--train_encoder_epochs', dest='train_encoder_epochs', type=int,
    default=10, help='Number of epochs to train the encoder before policy training begins')
parser.add_argument('-ae', '--act_epsilon', dest='act_epsilon', type=float,
    default=1., help='Initial chance of taking a random action')
parser.add_argument('-tae', '--target_act_epsilon', dest='target_act_epsilon', type=float,
    default=0.1, help='The final target probability of taking a random action')

if __name__ == '__main__':
    ### Setp for MPI ###
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_processes = comm.Get_size()
    controller = 0

    ### Define starting parameters ###
    args = parser.parse_args()

    n_epochs = args.n_epochs
    max_rollout_steps = args.max_rollout_steps
    n_train_batches = args.n_rollouts
    n_asn_train_batches = args.n_asn_rollouts
    n_process_batches = int(n_train_batches / n_processes)
    n_asn_process_batches = int(n_asn_train_batches / n_processes)
    n_as_proposals = args.n_asn_gen
    # min_branch, max_branch = 2, 3
    log_freq = args.log_freq
    asn_proposal_delay = args.asn_proposal_delay # How many epochs in between new ASNs being added
    n_asn_train_samples = args.n_asn_train_samples
    act_branch_factor = args.act_branch_factor # How many actions each model should output
                                               # TODO: Change models to LSTMs and make this variable
    train_encoder_epochs = args.train_encoder_epochs
    act_epsilon = args.act_epsilon
    target_act_epsilon = args.target_act_epsilon
    train_act_sets = [[i] for i in range(0, 7)]
    encoder_save_path = 'models/encoder.h5'

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
    hasl = HASL(comm, controller, rank, state_shape=(OBS_DIM, OBS_DIM), state_depth=OBS_DEPTH, sess_config=device_config)

    if rank == controller:
        if os.path.exists(encoder_save_path + '.index') and os.path.isfile(encoder_save_path + '.index'):
            hasl.load_encoder(encoder_save_path)
            train_encoder_epochs = 0
            print('Loaded HASL autoencoder model!')

    hasl.sync_weights()

    for epoch in range(1, n_epochs+1):
        if epoch == train_encoder_epochs + 1:
            if rank == controller:
                print('Starting policy training!')
            act_epsilon = target_act_epsilon

        if epoch % asn_proposal_delay == 0:
            n_rollouts = n_asn_process_batches
        else:
            n_rollouts = n_process_batches

        train_data = []
        encoder_data = []
        all_rewards = []
        for _ in range(n_rollouts):
            ### Simulate more episodes to gain training data ###
            if rank == controller:
                all_data = comm.gather(
                    worker(train_act_sets, hasl, act_epsilon=act_epsilon, obs_stack_size=OBS_DEPTH, 
                           max_steps=max_rollout_steps), controller)
                new_train_data = [x[0] for x in all_data]
                train_data.extend(new_train_data)
                encoder_data.extend([x[1] for x in all_data])
                all_rewards.extend([sum(x[2]) for x in all_data])
            else:
                comm.gather(worker(train_act_sets, hasl, act_epsilon=act_epsilon,
                                   obs_stack_size=OBS_DEPTH, max_steps=max_rollout_steps), controller)

        if rank == controller:
            encoder_data = np.concatenate(encoder_data)
            cat_train_data = np.concatenate(train_data)
        
            print(f'----- Epoch {epoch} -----')

            print('# micro steps: {}'.format(len(encoder_data)))
            print('# macro steps: {}'.format(len(cat_train_data)))

        ###### End of data gathering, start of training ######

            if epoch % log_freq == 0:
                log(f'Epoch {epoch} train action sets:')
                log(str(train_act_sets) + '\n')
            
            if epoch <= train_encoder_epochs:
                ### Train auto encoder model ###
                assert encoder_data.shape[1] == 4, 'The encoder data must have a shape of (?, 4)!'
                train_states = encoder_data[:, 0]
                train_actions = encoder_data[:, 1]
                train_state_ps = encoder_data[:, 3]
                    
                loss = hasl.train_encoder(train_states, batch_size=128, save_path=encoder_save_path)
                print(f'Auto encoder loss: {loss}')
            else:
                ### Pull rewards and action sequences from the training data ###

                print(
                    f'Avg Reward: {np.mean(all_rewards)}, Min: {np.min(all_rewards)}, Max: {np.max(all_rewards)}, Std: {np.std(all_rewards)}')

                if epoch % asn_proposal_delay == 0:
                    ### Calculate state differences ###

                    state_changes = []
                    all_states = []
                    act_seqs = []
                    reward_list = []
                    # ss = []
                    seq_len = act_branch_factor
                    for ep in range(len(train_data)):
                        real_step = 0
                        for step in range(seq_len, len(train_data[ep])):
                            real_step += len(train_data[ep][step][1])
                            # ss.append([real_step, train_data[ep][step][0]])
                            # state_changes.append([real_step, train_data[ep][step][0] - train_data[ep][step-seq_len][0]])
                            state_changes.append(
                                train_data[ep][step][0] - train_data[ep][step-seq_len][0])
                            all_states.append(train_data[ep][step-seq_len:step, 0])
                            act_seqs.append(train_data[ep][step-seq_len:step, 1])
                            reward_list.append(
                                sum(train_data[ep][step-seq_len:step, 2]))

                    state_changes = np.asarray(state_changes).squeeze()
                    all_states = np.asarray(all_states).squeeze()
                    act_seqs = np.asarray(act_seqs)

                    ### Use scaled on rewards to stochastically choose which episodes to pull actions from ###

                    scaled_rewards = np.asarray(reward_list)
                    zero_dist = min(reward_list)
                    scaled_rewards -= zero_dist
                    total_reward = sum(scaled_rewards)
                    scaled_rewards /= total_reward

                    top_ids = np.random.choice(
                        range(len(scaled_rewards)), size=n_as_proposals, replace=False, p=scaled_rewards)
                    top_samples = [state_changes[i] for i in top_ids] # List of the central samples

                    ### Gather and format data for action sequence proposals ###

                    # Get a list with an entry for each top_sample
                    # Each list contains n indices of the other closest samples
                    as_net_train_data = find_neighbors(
                        top_samples, state_changes, n=n_asn_train_samples)

                    # with open('sc.pickle', 'wb') as f:
                    #     pickle.dump([state_changes, all_states, act_seqs, as_net_train_data], f)
                    
                    ### Formatting training data for new act set models ###

                    obs = []
                    acts = []
                    for cluster in as_net_train_data:
                        obs.append([])
                        acts.append([])
                        for idx in cluster:
                            obs[-1].append(np.vstack(all_states[idx]))
                            acts[-1].append(np.hstack(act_seqs[idx]))

                    # Collapsing the branches of x over n samples to just be n*x examples
                    # It is no longer useful to have those dimensions separated for training
                    # However, it will be useful to bring back if this is switched to an LSTM
                    obs = np.asarray(obs)
                    obs = obs.reshape(obs.shape[0], -1, obs.shape[-1])
                    acts = np.asarray(acts).reshape(obs.shape[0], -1)

                    print(obs.shape, acts.shape)

                    # TODO: Make an initial period where this doesn't happen for x epochs
                    # so that the autoencoder has time to learn more stabely
                    for i in range(n_as_proposals):
                        print('Training new ASN #{}'.format(i+1))
                        hasl.create_as_net(obs[i], acts[i], n_acts=act_branch_factor, n_epochs=10,
                            hidden_dims=(64,32,))
                    hasl.set_act_seqs()
                    hasl.sync_asns()
                    
                else:
                    hasl.train_policy(
                        cat_train_data[:, 0], cat_train_data[:, 1], cat_train_data[:, 2])
                    hasl.sync_weights()

            ### Send the new action sequences to each process and sync models ###
            comm.bcast(train_act_sets, controller)
        else:
            if epoch % asn_proposal_delay == 0:
                hasl.sync_asns()
            else:
                hasl.sync_weights()
            ### Incorporate the new action sequences into its list for further training and sync models ###
            train_act_sets = comm.bcast(None, controller)

        # synced = hasl.is_model_synced()
        # if rank == controller:
        #     print(f'Model Synced: {synced}')

        gc.collect()

    if rank == controller:
        print('done')