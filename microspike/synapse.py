import numpy as np
from typing import Union


class Synapse():
    def __init__(
                self,
                w: np.array,
                w_max: float,
                w_min: float,
                A_pre: float,
                A_post: float,
                tau_pre: float,
                tau_post: float,
                approximate: bool = True,
                dt : float = None
                ) -> None:
        self.w = np.copy(w)
        self.w_max = w_max
        self.w_min = w_min
        self.A_pre_value = A_pre
        self.A_post_value = A_post
        self.A_pre, self.A_post = self.set_A()
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        ## if True do nearest spike approximation, else consider all the contributions of the previous presynaptic spikes
        self.approximate = approximate
        
        self.a_pre = np.zeros_like(self.w)
        self.a_post = np.zeros_like(self.w)
        self.dt = dt

        if not (self.A_pre > 0).all().item():
            raise ValueError("A_pre should be > 0")
        if not (self.A_post < 0).all().item():
            raise ValueError("A_post should be < 0")
        if len(self.w.shape) < 2:
            raise ValueError("w should be 2D")
    
    def set_A(self):
        num_neurons = self.w.shape[1]
        A_pre = np.ones((num_neurons,)) * self.A_pre_value
        A_post = np.ones((num_neurons,)) * self.A_post_value
        return A_pre, A_post 
    
    def get_w_tmp(self, spikes_i: np.array, i):
        """
        spikes_i contains only the correct window of spikes_i
        """
        return self.w[spikes_i, i].squeeze()

    def on_pre_w(self,idx):
        """
        idx is the index of presynaptic neurons that spiked at the current time and now increasing their a_pre,
        idx is a python list
        """
        self.w[idx, :] = np.clip(self.w[idx, :] + self.a_post[idx, :], self.w_min, self.w_max)
        self.a_post[idx, :] = 0.

    def on_post_w(self, idx):
        """
        idx is the index of postsynaptic neurons not in refractory period and their membrane potential above threshold
        """
        self.w[:, idx] = np.clip(self.w[:, idx] + self.a_pre[:, idx], self.w_min, self.w_max)
        self.a_pre[:, idx] = 0.


    def on_pre_a(self, idx):
        """
        idx is the index of presynaptic neurons that spiked at the current time and now increasing their a_pre,
        idx is a python list
        """
        if self.approximate:
            self.a_pre[idx, :] = self.A_pre
        else: self.a_pre[idx, :] += self.A_pre
        
    def on_post_a(self, idx):
        """
        idx is the index of postsynaptic neurons not in refractory period and their membrane potential above threshold
        """
        if self.approximate:
            self.a_post[:, idx] = self.A_post[idx]
        else: self.a_post[:, idx] += self.A_post[idx]

    def update_a(self):
        self.a_pre = self.a_pre - self.dt / self.tau_pre * self.a_pre
        self.a_post = self.a_post - self.dt / self.tau_post * self.a_post

    def update_synapse(self,idx_post_spikinig, idx_pre_spiking, inside_wrong_pattern:Union[bool, None]):
        self.on_post_w(idx_post_spikinig)
        self.on_pre_w(idx_pre_spiking)
        if inside_wrong_pattern is None or inside_wrong_pattern == False:
            self.on_post_a(idx_post_spikinig)
            self.on_pre_a(idx_pre_spiking)
            self.update_a()

        else:
            self.on_post_a_punishment(idx_post_spikinig)
            self.on_pre_a_punishment(idx_pre_spiking)
            self.update_a_punishment()

    def on_pre_a_punishment(self, idx):
        if self.approximate:
            self.a_pre[idx, :] = self.A_post
        else:
            self.a_pre[idx, :] += self.A_post

    def on_post_a_punishment(self, idx):
        if self.approximate:
            self.a_post[:, idx] = self.A_pre[idx]
        else:
            self.a_post[:, idx] += self.A_pre[idx]

    def update_a_punishment(self):
        self.a_pre = self.a_pre - self.dt / self.tau_post * self.a_pre
        self.a_post = self.a_post - self.dt / self.tau_pre * self.a_post

    def init_weight(self, M, N):
        """
        Creates a weight matrix of shape (M, N) with elements drawn from a uniform distribution (0, 1).

        Args:
            M: Number of presynaptic neurons (rows in the weight matrix).
            N: Number of postsynaptic neurons (columns in the weight matrix).

        Returns:
            A weight matrix of shape (M, N) with random values between 0 (inclusive) and 1 (exclusive).
        """
        weight = np.random.uniform(low=0.0, high=1.0, size=(M, N))
        return weight

    def add_new_neurons(self,num_new_neurons):
        num_original_neurons = self.w.shape[1]
        new_neurons_weight = self.init_weight(M=2000, N=num_new_neurons)
        total_neurons = num_original_neurons + num_new_neurons
        
        self.w = np.concatenate((self.w, new_neurons_weight), axis=1)

        self.A_post = np.ones((total_neurons,)) * self.A_post_value
        self.A_post[:num_original_neurons] = 0
        self.A_pre = np.ones((total_neurons,)) * self.A_pre_value
        self.A_pre[:num_original_neurons] = 0
        
        self.a_pre = np.zeros_like(self.w)
        self.a_post = np.zeros_like(self.w)