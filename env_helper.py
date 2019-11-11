import cv2
import numpy as np
from sklearn.neighbors import LSHForest

from utils import OBS_DIM
from hasl_model import HASL

# Super Mario stuff
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

### Data formatting ###

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

def filter_obs(obs, obs_shape=(OBS_DIM, OBS_DIM)):
    obs = cv2.resize(obs, obs_shape, interpolation=cv2.INTER_LINEAR)
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    return obs / 255

def discount_rewards(rewards, gamma=0.99):
    """Discounts an array of rewarwds, generally used after
    an episode ends and before training.
    Args:
        rewards (:obj:`list` of float): The rewards to be discounted.
        gamma (float, optional): Gamma in the reward discount function.
            Higher gamma = higher importance on later rewards.
    Returns:
        (:obj:`list` of float): List of the disounted rewards.
    Examples:
        >>> print([round(x) for x in discount_rewards([1, 2, 3], gamma=0.99)])
            [5.92, 4.97, 3.0]
    """
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return new_rewards[::-1]

### Environment tasks ###

def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
    return env

def worker(hasl, max_steps=1000, act_epsilon=0.1, obs_stack_size=4, return_micro_data=True, return_macro_data=True):
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
    if return_macro_data:
        train_data = []
    else:
        train_data = None
    if return_micro_data:
        full_data = []
    else:
        full_data = None

    env = make_env()

    obs = env.reset()
    obs = filter_obs(obs)
    h_obs_stack = ObsStack(obs.shape, obs_stack_size) # High-level obs stack
    h_obs_stack.push(obs)
    enc_full_obs = hasl.apply_encoder([h_obs_stack.get_stack()])
    rewards = []

    if return_micro_data:
        l_obs_stack = ObsStack(obs.shape, obs_stack_size) # Low-level obs stack
        l_obs_stack.push(obs)
    
    ep_reward = 0
    step = 0
    while step < max_steps:
        act, act_stack, master_act = hasl.choose_action(
            enc_full_obs, act_stack=None, epsilon=act_epsilon)

        if return_macro_data:
            train_data.append([enc_full_obs])
        step_reward = 0
        while act is not None:
            # Continuously called until the macro-action is over
            if return_micro_data:
                full_data.append([l_obs_stack.get_stack()])
            obs_p, r, d, _ = env.step(act)
            obs_p = filter_obs(obs_p)
            if return_micro_data:
                l_obs_stack.push(obs_p)
            if return_macro_data:
                full_data[-1].extend([act, r, l_obs_stack.get_stack()])
            rewards.append(r)
            step_reward += r
            step += 1
            if d or step >= max_steps:
                break
            act, act_stack, _ = hasl.choose_action(
                enc_full_obs, act_stack=act_stack, epsilon=act_epsilon)

        ep_reward += step_reward

        if return_macro_data:
            h_obs_stack.push(obs_p)
            enc_full_obs = hasl.apply_encoder([h_obs_stack.get_stack()])
            train_data[-1].extend([master_act, step_reward, enc_full_obs])

        if d or step >= max_steps:
            break

    ### Shape rewards ###

    if return_macro_data:
        train_data = np.asarray(train_data)
        train_data[:, 2] = discount_rewards(train_data[:, 2])

    if return_micro_data:
        full_data = np.asarray(full_data)
        full_data[:, 2] = discount_rewards(full_data[:, 2])

    return train_data, full_data, rewards

### Training support algorithms ###

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