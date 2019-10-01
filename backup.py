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
        assert obs.shape == self.expected_shape
        assert type(obs) == list or type(obs) == np.ndarray

        if type(obs) == list:
            obs = np.asarray(obs)

        self.stack[..., 1:] = self.stack[..., :-1]
        self.stack[..., 0] = obs

        if self.fill_first:
            self.fill_first = False
            for _ in range(stack_size - 1):
                self.push(obs)

    def get_stack(self):
        return self.stack

    def get_flat_stack(self):
        return self.stack.reshape(-1)


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
            enc_obs_stack: Flattened stack of n encoded observations.
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
    encoded_obs = hasl.apply_encoder(obs.reshape(1, OBS_DIM, OBS_DIM, 1))
    obs_stack = ObsStack(encoded_obs.shape, obs_stack_size)
    obs_stack.push(encoded_obs)

    ep_reward = 0
    step = 0
    while step < max_steps:
        # act_idx = np.random.randint(len(action_sets))
        # act_set = action_sets[act_idx]
        full_obs = obs_stack.get_flat_stack().reshape(1, -1)
        # TODO: The method for generating act_sets seems strange
        act_set = [hasl.choose_action(
            full_obs, epsilon=act_epsilon, possible_acts=list(range(env.action_space.n)))]

        train_data.append([full_obs])
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
        obs_stack.push(encoded_obs)

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
    top_x = 5
    act_top_x = 2
    min_branch, max_branch = 2, 3
    log_freq = 60
    n_as_proposals = 5
    n_as_train_samples = 512
    rand_explore_epochs = 50
    act_epsilon = 1
    target_act_epsilon = 0.1
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
    hasl = HASL(comm, controller, rank, state_depth=4, sess_config=device_config)

    for epoch in range(1, n_epochs+1):
        if epoch > rand_explore_epochs:
            act_epsilon = target_act_epsilon

        train_data = []
        encoder_data = []
        all_rewards = []
        for _ in range(n_process_batches):
            ### Simulate more episodes to gain training data ###
            if rank == controller:
                all_data = comm.gather(
                    worker(train_act_sets, hasl, act_epsilon=act_epsilon), controller)
                new_train_data = [x[0] for x in all_data]
                train_data.extend(new_train_data)
                encoder_data.extend([x[1] for x in all_data])
                all_rewards.extend([sum(x[2]) for x in all_data])
            else:
                comm.gather(worker(train_act_sets, hasl,
                                   act_epsilon=act_epsilon), controller)

        if rank == controller:
            encoder_data = np.concatenate(encoder_data)
            cat_train_data = np.concatenate(train_data)

        ###### End of data gathering, start of training ######

        if rank == controller:
            print(f'----- Epoch {epoch} -----')

            if epoch % log_freq == 0:
                log(f'Epoch {epoch} train action sets:')
                log(str(train_act_sets) + '\n')

            ### Train reverse dynamics encoder model ###

            assert encoder_data.shape[1] == 4, 'The ecoder data must have a shape of (?, 4)!'
            train_states = encoder_data[:, 0]
            train_actions = encoder_data[:, 1]
            train_state_ps = encoder_data[:, 3]

            accuracy = hasl.train_encoder(
                train_states, train_state_ps, train_actions)
            print(f'Inverse Dynamics Accuracy: {str(accuracy*100)[:5]}%')

            

        #     # TODO: Change the way the actions are passed in to train the policy
        #     hasl.train_policy(
        #         cat_train_data[:, 0], cat_train_data[:, 1], cat_train_data[:, 2])

        #     ### Pull rewards and action sequences from the training data ###

        #     print(
        #         f'Avg Reward: {np.mean(all_rewards)}, Min: {np.min(all_rewards)}, Max: {np.max(all_rewards)}, Std: {np.std(all_rewards)}')

        #     ### Calculate state differences ###

        #     state_changes = []
        #     start_states = []
        #     act_seqs = []
        #     reward_list = []
        #     # ss = []
        #     seq_len = 3
        #     for ep in range(len(train_data)):
        #         real_step = 0
        #         for step in range(seq_len, len(train_data[ep])):
        #             real_step += len(train_data[ep][step][1])
        #             # ss.append([real_step, train_data[ep][step][0]])
        #             # state_changes.append([real_step, train_data[ep][step][0] - train_data[ep][step-seq_len][0]])
        #             state_changes.append(
        #                 train_data[ep][step][0] - train_data[ep][step-seq_len][0])
        #             start_states.append(train_data[ep][step-seq_len][0])
        #             act_seqs.append(train_data[ep][step-seq_len:step, 1])
        #             reward_list.append(
        #                 sum(train_data[ep][step-seq_len:step, 2]))

        #     state_changes = np.asarray(state_changes).squeeze()
        #     start_states = np.asarray(start_states).squeeze()
        #     act_seqs = np.asarray(act_seqs)

        #     ### Use softmax on rewards to stochastically choose which episodes to pull actions from ###

        #     zero_dist = -min(reward_list)
        #     scaled_rewards = [r + zero_dist for r in reward_list]
        #     total_reward = sum(scaled_rewards)
        #     scaled_rewards = [r / total_reward for r in scaled_rewards]

        #     top_ids = np.random.choice(
        #         range(len(scaled_rewards)), size=top_x, replace=False, p=scaled_rewards)
        #     top_samples = [state_changes[i] for i in top_ids]

        #     ### Gather and format data for action sequence proposals ###

        #     as_net_train_data = find_neighbors(
        #         top_samples, state_changes, n=n_as_train_samples)

        #     obs = np.array([start_states[x] for x in as_net_train_data[0]])
        #     acts = np.array([np.hstack(act_seqs[x])
        #                      for x in as_net_train_data[0]])

        #     # TODO: Make an initial period where this doesn't happen for x epochs
        #     # so that the autoencoder has time to learn more stabely
        #     if epoch % 10 == 0:
        #         hasl.create_as_net(obs, acts)

        #     # print(f'Count: {len(state_changes)}')
        #     # with open('state_changes.pickle', 'wb') as f:
        #     #     pickle.dump(state_changes, f)

        #     # with open('ss.pickle', 'wb') as f:
        #     #     pickle.dump(ss, f)

        #     ### Send the new action sequences to each process and sync models ###
        #     comm.bcast(train_act_sets, controller)
        #     hasl.sync_weights()
        # else:
        #     ### Incorporate the new action sequences into its list for further training and sync models ###
        #     train_act_sets = comm.bcast(None, controller)
        #     hasl.sync_weights()

    if rank == 0:
        print('done')
