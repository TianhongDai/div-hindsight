import numpy as np
from scipy.stats import rankdata
import random
from baselines.her.dpp_module.dpp import DPP

def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if (replay_strategy == 'future') or (replay_strategy == 'final'):
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions
        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}
        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        if replay_strategy == 'final':
            future_t[:] = T
        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # Reconstruct info dictionary for reward computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value
        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}
        assert(transitions['u'].shape[0] == batch_size_in_transitions)
        return transitions
    return _sample_her_transitions

def make_sample_her_transitions_diversity(replay_strategy, replay_k, reward_fun):
    if (replay_strategy == 'future') or (replay_strategy == 'final'):
        future_p = 1 - (1. / (1 + replay_k))
    else:
        future_p = 0
    def _sample_her_transitions(episode_batch, batch_size_in_transitions, update_stats=False):
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        if not update_stats:
            div_trajectory = episode_batch['div']
            # calculate the priority
            p_trajectory = div_trajectory.copy()
            p_trajectory = p_trajectory / p_trajectory.sum()
            episode_idxs_div = np.random.choice(rollout_batch_size, size=batch_size, replace=True, p=p_trajectory.flatten())
            episode_idxs = episode_idxs_div
        transitions = {}
        for key in episode_batch.keys():
            if not key == 's' and not key == 'div':
                transitions[key] = episode_batch[key][episode_idxs, t_samples].copy()
        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        if replay_strategy == 'final':
            future_t[:] = T
        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # Reconstruct info dictionary for reward computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value
        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}
        assert(transitions['u'].shape[0] == batch_size_in_transitions)
        return transitions
    return _sample_her_transitions

def make_sample_her_transitions_diversity_with_kdpp(replay_strategy, replay_k, subset_size, goal_type, sigma, reward_fun):
    """
    sample the transitions using k-DPP
    """
    if (replay_strategy == 'future') or (replay_strategy == 'final'):
        future_p = 1 - (1. / (1 + replay_k))
    else:
        future_p = 0
    def _sample_her_transitions(episode_batch, batch_size_in_transitions, update_stats=False):
        """
        the subset decide how much data will be used for the dpp sampling

        if not update the stats, just use the normal batch_size
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size if update_stats else subset_size)
        if not update_stats:
            div_trajectory = episode_batch['div']
            # cal the priority
            p_trajectory = div_trajectory.copy()
            p_trajectory = p_trajectory / p_trajectory.sum()
            episode_idxs_div = np.random.choice(rollout_batch_size, size=subset_size, replace=True, p=p_trajectory.flatten())
            episode_idxs = episode_idxs_div
        transitions = {}
        for key in episode_batch.keys():
            if not key == 's' and not key == 'div':
                transitions[key] = episode_batch[key][episode_idxs, t_samples].copy()
        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        if not update_stats:
            # recalculate the future p for balance!
            future_p_ = 1 - (batch_size * (1 - future_p)) / subset_size
        else:
            future_p_ = future_p
        her_indexes = np.where(np.random.uniform(size=batch_size if update_stats else subset_size) < future_p_)
        future_offset = np.random.uniform(size=batch_size if update_stats else subset_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        if replay_strategy == 'final':
            future_t[:] = T
        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # start to select the goals using dpp, if update stats, just ignore
        if not update_stats:
            if goal_type == 'full':
                f_vector = transitions['g'].copy()
            elif goal_type == 'rotate':
                f_vector = transitions['g'][:, 3:].copy()
            else:
                raise NotImplementedError
            dpp = DPP(f_vector)
            #dpp.compute_kernel(kernel_type='cos-sim')
            dpp.compute_kernel(kernel_type='rbf', sigma=sigma)
            #dpp.compute_kernel(kernel_type='rbf', sigma=1)
            dpp_idx = dpp.sample_k(batch_size)
            # it may have the problem that dpp idx is not equal to the batch size
            if len(dpp_idx) != batch_size:
                # get the total index
                total_idx = [i for i in range(subset_size)]
                unselected_idx = list(set(total_idx) - set(dpp_idx))
                rest_idx = np.random.choice(unselected_idx, size=batch_size - len(dpp_idx), replace=False)
                sample_idx = np.concatenate([dpp_idx, rest_idx])
                # built up the transitions
            else:
                sample_idx = dpp_idx
            transitions_ = {}
            for key in transitions.keys():
                if not key == 's' and not key == 'div':
                    transitions_[key] = transitions[key][sample_idx].copy()
            # assign back
            transitions = transitions_
        # Reconstruct info dictionary for reward computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value
        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}
        assert(transitions['u'].shape[0] == batch_size_in_transitions)
        return transitions
    return _sample_her_transitions