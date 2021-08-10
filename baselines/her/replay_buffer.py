import threading
import numpy as np
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
import math
from scipy.stats import rankdata
import json

class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions
        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}
        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]
        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]
        transitions = self.sample_transitions(buffers, batch_size)
        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key
        return transitions

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]
        with self.lock:
            idxs = self._get_storage_idx(batch_size)
            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]
            self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        # update replay size
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx

class ReplayBufferDiversity:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions, prioritization, env_name, goal_type):
        """
        Creates a replay buffer for measuring the diversity
        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}
        self.buffers['div'] = np.empty([self.size, 1]) # diversity
        # the prioritization is dpp now
        self.prioritization = prioritization
        self.env_name = env_name
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.current_size_test = 0
        self.n_transitions_stored_test = 0
        self.goal_type = goal_type
        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}
        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]
        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]
        transitions = self.sample_transitions(buffers, batch_size)
        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            if not key == 'div':
                assert key in transitions, "key %s missing from transitions" % key
        return transitions

    def store_episode(self, episode_batch, clip_div):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]
        buffers = {}
        for key in episode_batch.keys():
            buffers[key] = episode_batch[key]
        # start to calculate the diversity
        if self.prioritization == 'diversity':
            # we only consider the fetch environment now
            if self.goal_type == 'full':
                traj = buffers['ag'].copy().astype(np.float32)
            elif self.goal_type == 'rotate':
                # if use the rotate...
                traj = buffers['ag'][:, :, 3:].copy().astype(np.float32)
            else:
                raise NotImplementedError 
            # normalize the vector
            traj = traj / np.linalg.norm(traj, axis=2, keepdims=True)
            diversity = []
            for i in range(traj.shape[1] - 1):
                result = np.einsum("ijk, ilk -> ijl" , traj[:, i:i+2, :], traj[:, i:i+2, :])
                diversity.append(np.linalg.det(result))
            diversity = np.array(diversity)
            diversity[diversity < 0] = 0
            diversity = np.sum(diversity, 0)
            # calculate the accumulate rewards
            mean_rewards = np.mean(buffers['info_is_success'].squeeze(), axis=1, keepdims=True)
            # rewrad gains is useless here... can be removed
            reward_gain = np.exp(mean_rewards)
            # clip the diversity - 0.001
            episode_batch['div'] = np.clip(diversity.reshape(-1, 1), 0, clip_div)
        # write the data
        with self.lock:
            idxs = self._get_storage_idx(batch_size)
            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]
            self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            # here we shouldn't randomly pick the trajectory to replace
            idx = np.random.randint(0, self.size, inc)
        # update replay size
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx