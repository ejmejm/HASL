import argparse
import numpy as np
import gc
import os
from mpi4py import MPI
import tensorflow as tf
from hasl_model import HASL, ImgEncoder, RobotEncoder
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
    default=4, help='Number of rollouts used for training an ASN per batch')
parser.add_argument('-nag', '--n_asn_proposals', dest='n_asn_proposals', type=int,
    default=10, help='Number of ASNs created in one training cycle')
parser.add_argument('-nat', '--n_asn_keep', dest='n_asn_keep', type=int,
    default=2, help='Number of ASNs kept from the generated ones per training cycle')
parser.add_argument('-apd', '--asn_proposal_delay', dest='asn_proposal_delay', type=int,
    default=8, help='Number of epochs in between ASN generation')
parser.add_argument('-nas', '--n_asn_train_samples', dest='n_asn_train_samples', type=int,
    default=512, help='Number of smaples drawn to train an individual ASN per sub-batch')
parser.add_argument('-ate', '--n_asn_train_epochs', dest='n_asn_train_epochs', type=int,
    default=3, help='Number of epochs to train each ASN')
parser.add_argument('-atb', '--n_asn_train_batches', dest='n_asn_train_batches', type=int,
    default=5, help='Number of bacthes to gather data and train ASNs')
parser.add_argument('-b', '--act_branch_factor', dest='act_branch_factor', type=int,
    default=3, help='Number of actions each ASN should execute')
parser.add_argument('-ee', '--n_encoder_epochs', dest='n_encoder_epochs', type=int,
    default=10, help='Number of epochs to train the  encoder before policy training begins')
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
parser.add_argument('-env', '--env_name', dest='env_name', type=str,
    default='SuperMarioBros-v0', help='Which environment to use')

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

    if args.env_name == 'SuperMarioBros-v0':
        encoder_type = ImgEncoder
        n_base_acts = 12
    elif args.env_name == 'FetchPickAndPlace-v1':
        encoder_type = lambda sess, state_ph, state_depth, optimizer: \
            RobotEncoder(sess, state_ph, state_depth, optimizer, enc_dim=28*OBS_STACK_SIZE)
        n_base_acts = 9
    elif args.env_name == 'FetchReach-v1':
        encoder_type = lambda sess, state_ph, state_depth, optimizer: \
            RobotEncoder(sess, state_ph, state_depth, optimizer, enc_dim=13*OBS_STACK_SIZE)
        n_base_acts = 9
    else:
        raise ValueError('env_name must be either SuperMarioBros-v0 or FetchPickAndPlace-v1 or FetchReach-v1')

    hasl = HASL(comm, controller, rank, encoder_type, n_base_acts=n_base_acts,
        state_shape=(OBS_DIM, OBS_DIM), state_depth=OBS_DEPTH, sess_config=device_config)

    if rank == controller:
        log('Args: ' + str(args))

    if hasl.encoder.train_req:
        ### Load enoder model if it exists ###
        if rank == controller:
            if os.path.exists(args.encoder_save_path + '.index') and \
                        os.path.isfile(args.encoder_save_path + '.index'):
                hasl.load_encoder(args.encoder_save_path)
                args.n_encoder_epochs = 0
                comm.bcast(args.n_encoder_epochs, controller)
                log('Loaded HASL autoencoder model!')
        else:
            args.n_encoder_epochs = comm.bcast(None, controller)

        hasl.sync_weights()

        ### Encoder training loop ###

        for encoder_epoch in range(1, args.n_encoder_epochs+1):
            if rank == controller:
                log(f'----- Encoder Training Epoch {epoch} -----')
                log('# micro steps: {}'.format(len(encoder_data)))

                # Run rollouts
                _, encoder_data, _ = \
                    gather_data(comm, rank, controller, hasl, args, n_data_batches, data_type='encoder')

                ### Train auto encoder model ###
                assert encoder_data.shape[1] == 4, 'The encoder data must have a shape of (?, 4)!'
                train_states = encoder_data[:, 0]
                train_actions = encoder_data[:, 1]
                train_state_ps = encoder_data[:, 3]
                    
                loss = hasl.train_encoder(train_states, batch_size=128, save_path=args.encoder_save_path)
                log(f'Auto encoder loss: {loss}')

            hasl.sync_weights()
            gc.collect()

    hasl.sync_weights()

    ### Policy training loop ###

    if rank == controller:
        log('Starting policy training!')
    args.act_epsilon = args.target_act_epsilon

    for epoch in range(1, args.n_epochs+1):
        if epoch % args.asn_proposal_delay == 0:

            ### Do ASN Proposals and Training ###

            if rank == controller:
                log(f'----- Epoch {epoch} -----')

            new_asn_ids = []
            for i in range(args.n_asn_proposals):
                asn_id = hasl.create_asn_ops(n_acts=args.act_branch_factor, hidden_dims=(32,32,))
                new_asn_ids.append(asn_id)

            top_samples = None
            for asn_epoch in range(args.n_asn_train_batches):
                # Run rollouts
                train_data, _, all_rewards = gather_data(
                    comm, rank, controller, hasl, args, n_asn_process_batches, data_type='policy', concat_data=False)

                if rank == controller:
                    cat_train_data = np.concatenate(train_data)

                    log('# macro steps: {}'.format(len(cat_train_data)))
                    log(f'Avg Reward: {np.mean(all_rewards)}, Min: {np.min(all_rewards)}, Max: {np.max(all_rewards)}, Std: {np.std(all_rewards)}')

                    ### Create and train new ASNs ###

                    # Calculate state differences
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

                    ### Use scaled rewards to stochastically choose which episodes to pull actions from ###

                    if not top_samples:
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

                    asn_accuracy = []
                    for i in range(args.n_asn_proposals):
                        log(f'Training ASN #{new_asn_ids[i]}'.format(i+1))
                        train_acc, test_acc = hasl.train_asn(new_asn_ids[i], obs[i], acts[i], batch_size=32, n_epochs=args.n_asn_train_epochs)
                        asn_accuracy.append((new_asn_ids[i], train_acc, test_acc))

            if rank == controller:
                keep_asn_ids = list(range(hasl.n_base_acts, hasl.n_curr_acts))
                log(str(keep_asn_ids))
                
                sort_key = lambda x: x[1]
                if asn_accuracy[0][2] is not None:
                    sort_key = lambda x: x[2]

                best_asns = sorted(asn_accuracy, key=sort_key, reverse=True)[:args.n_asn_keep]
                best_asn_ids = [x[0] for x in best_asns]
                keep_asn_ids.extend(best_asn_ids)
                log(str(((str(best_asn_ids),  str(keep_asn_ids)))))
                hasl.set_act_seqs(asn_keep_ids=keep_asn_ids)
                log(str((hasl.n_curr_acts, hasl.n_act_seqs, len(hasl.asns))))

            hasl.sync_asns()
        else:
            # Run rollouts
            cat_train_data, _, all_rewards = \
                gather_data(comm, rank, controller, hasl, args, n_process_batches, data_type='policy')

            if rank == controller:
                log(f'----- Epoch {epoch} -----')
                log('# macro steps: {}'.format(len(cat_train_data)))
                log(f'Avg Reward: {np.mean(all_rewards)}, Min: {np.min(all_rewards)}, Max: {np.max(all_rewards)}, Std: {np.std(all_rewards)}')

                ### Train master network policy ###

                hasl.train_policy(
                    cat_train_data[:, 0], cat_train_data[:, 1], cat_train_data[:, 2])

            hasl.sync_weights()

        # synced = hasl.is_model_synced()
        # if rank == controller:
        #     print(f'Model Synced: {synced}')

        gc.collect()

    if rank == controller:
        log('done')