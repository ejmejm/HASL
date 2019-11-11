import argparse
import numpy as np
import gc
import os
from mpi4py import MPI
import tensorflow as tf
from hasl_model import HASL
import multiprocessing
import pickle

from env_helper import *
from utils import *

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
parser.add_argument('-nag', '--n_asn_gen', dest='n_asn_proposals', type=int,
    default=5, help='Number of ASNs created in one training cycle')
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
parser.add_argument('-l', '--log_path', dest='log_path', type=str,
    default='logs/training.log', help='Path to save the log to')
parser.add_argument('-d', '--dump_train_data', dest='dump_train_data', type=bool,
    default=False, help='Whether or not to dump the training data to a pickle')
parser.add_argument('-esp', '--encoder_save_path', dest='encoder_save_path', type=str,
    default='models/encoder.h5', help='Path to save encoder model to if one does not already exist')
parser.add_argument('-ate', '--n_asn_train_epochs', dest='n_asn_train_epochs', type=int,
    default=5, help='Number of epochs to train each ASN')

if __name__ == '__main__':
    ### Setp for MPI ###
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_processes = comm.Get_size()
    controller = 0

    ### Define starting parameters ###
    args = parser.parse_args()

    init_logger(args.log_path)

    n_process_batches = int(args.n_rollouts / n_processes)
    n_asn_process_batches = int(args.n_asn_rollouts / n_processes)
    # min_branch, max_branch = 2, 3

    if rank == controller:
        device_config = tf.ConfigProto()
        use_device = tf.device('/cpu:0')
    else:
        device_config = tf.ConfigProto(device_count={'GPU': 0})

    hasl = HASL(comm, controller, rank, state_shape=(OBS_DIM, OBS_DIM), state_depth=OBS_DEPTH, sess_config=device_config)

    ### Load enoder model if it exists ###
    if rank == controller:
        if os.path.exists(args.encoder_save_path + '.index') and \
                 os.path.isfile(args.encoder_save_path + '.index'):
            hasl.load_encoder(args.encoder_save_path)
            args.train_encoder_epochs = 0
            log('Loaded HASL autoencoder model!')

    hasl.sync_weights()

    if rank == controller:
        log('Args: ' + str(args))

    ### Main training loop ###
    for epoch in range(1, args.n_epochs+1):
        if epoch == args.train_encoder_epochs + 1:
            if rank == controller:
                log('Starting policy training!')
            args.act_epsilon = args.target_act_epsilon

        if epoch % args.asn_proposal_delay == 0:
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
                    worker(hasl, act_epsilon=args.act_epsilon, obs_stack_size=OBS_DEPTH, 
                           max_steps=args.max_rollout_steps), controller)
                new_train_data = [x[0] for x in all_data]
                train_data.extend(new_train_data)
                encoder_data.extend([x[1] for x in all_data])
                all_rewards.extend([sum(x[2]) for x in all_data])
            else:
                comm.gather(worker(hasl, act_epsilon=args.act_epsilon,
                                   obs_stack_size=OBS_DEPTH, max_steps=args.max_rollout_steps), controller)

        if rank == controller:
            encoder_data = np.concatenate(encoder_data)
            cat_train_data = np.concatenate(train_data)
        
            log(f'----- Epoch {epoch} -----')

            log('# micro steps: {}'.format(len(encoder_data)))
            log('# macro steps: {}'.format(len(cat_train_data)))

            ###### End of data gathering, start of training ######
            
            if epoch <= args.train_encoder_epochs:
                ### Train auto encoder model ###
                assert encoder_data.shape[1] == 4, 'The encoder data must have a shape of (?, 4)!'
                train_states = encoder_data[:, 0]
                train_actions = encoder_data[:, 1]
                train_state_ps = encoder_data[:, 3]
                    
                loss = hasl.train_encoder(train_states, batch_size=128, save_path=args.encoder_save_path)
                log(f'Auto encoder loss: {loss}')
            else:
                log(f'Avg Reward: {np.mean(all_rewards)}, Min: {np.min(all_rewards)}, Max: {np.max(all_rewards)}, Std: {np.std(all_rewards)}')

                if epoch % args.asn_proposal_delay == 0:
                    ### Create and train new ASNs ###

                    ### Calculate state differences ###

                    state_changes = []
                    all_states = []
                    act_seqs = []
                    reward_list = []
                    seq_len = args.act_branch_factor
                    for ep in range(len(train_data)):
                        for step in range(seq_len, len(train_data[ep])):
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
                        range(len(scaled_rewards)), size=args.n_asn_proposals, replace=False, p=scaled_rewards)
                    top_samples = [state_changes[i] for i in top_ids] # List of the central samples

                    ### Gather and format data for action sequence proposals ###

                    # Get a list with an entry for each top_sample
                    # Each list contains n indices of the other closest samples
                    as_net_train_data = find_neighbors(
                        top_samples, state_changes, n=args.n_asn_train_samples)

                    if args.dump_train_data:
                        with open('sc.pickle', 'wb') as f:
                            pickle.dump([state_changes, all_states, act_seqs, as_net_train_data], f)
                    
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

                    log(str(obs.shape) + ' ' + str(acts.shape))

                    # TODO: Make an initial period where this doesn't happen for x epochs
                    # so that the autoencoder has time to learn more stabely
                    for i in range(args.n_asn_proposals):
                        log('Training new ASN #{}'.format(i+1))
                        hasl.create_asn(obs[i], acts[i], n_acts=args.act_branch_factor, n_epochs=args.n_asn_train_epochs,
                            hidden_dims=(32,32,))
                    hasl.set_act_seqs()
                    hasl.sync_asns()
                    
                else:
                    ### Train master network policy ###

                    hasl.train_policy(
                        cat_train_data[:, 0], cat_train_data[:, 1], cat_train_data[:, 2])
                    hasl.sync_weights()
        else:
            if epoch % args.asn_proposal_delay == 0:
                hasl.sync_asns()
            else:
                hasl.sync_weights()

        # synced = hasl.is_model_synced()
        # if rank == controller:
        #     print(f'Model Synced: {synced}')

        gc.collect()

    if rank == controller:
        log('done')