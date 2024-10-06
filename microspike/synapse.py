# import numpy as np
import torch
from typing import List, Optional
class Synapse:
    """
    Represents a synapse in the network, with weight update mechanisms for STDP.

    Attributes:
        w (torch.Tensor): Synaptic weight matrix.
        w_max (float): Maximum synaptic weight.
        w_min (float): Minimum synaptic weight.
        A_pre (float): Learning rate for pre-synaptic activity.
        A_post (float): Learning rate for post-synaptic activity.
        tau_pre (float): Time constant for pre-synaptic activity decay.
        tau_post (float): Time constant for post-synaptic activity decay.
        approximate (bool): Whether to use nearest spike approximation.
        dt (float): Time step for updates.
    """
    def __init__(
            self,
            w: torch.Tensor,
            w_max: float,
            w_min: float,
            A_pre: float,
            A_post: float,
            tau_pre: float,
            tau_post: float,
            approximate: bool = False,
            dt: float = None,
            device: Optional[str]=None,
        ) -> None:
        self.device = device
        if not self.device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.w = w.clone().to(self.device)
        self.w_max = w_max
        self.w_min = w_min
        self.A_pre = A_pre
        self.A_post = A_post
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.approximate = approximate
        self.a_pre = torch.zeros_like(self.w, device=self.device)
        self.a_post = torch.zeros_like(self.w, device=self.device)
        self.dt = dt

        self.t_j = - torch.ones_like(self.w, device=self.device) * float('inf')
        self.t_i = - torch.ones_like(self.w, device=self.device) * float('inf')
        self.w_log = torch.zeros((self.w.shape[0], 100), device=self.device)

    def get_w_tmp(self, spikes_i: torch.Tensor, i: int) -> torch.Tensor:
        """Returns the temporary weights for the given pre-synaptic neurons and post-synaptic neuron index."""
        return self.w[spikes_i, i].squeeze()

    def on_pre_w(self, idx: List[int]) -> None:
        """Updates synaptic weights for pre-synaptic spikes."""
        self.w[idx, :] = torch.clamp(self.w[idx, :] + self.a_post[idx, :], self.w_min, self.w_max)
        self.a_post[idx, :] = 0.

    def on_post_w(self, idx: List[int]) -> None:
        """Updates synaptic weights for post-synaptic spikes."""
        self.w[:, idx] = torch.clamp(self.w[:, idx] + self.a_pre[:, idx], self.w_min, self.w_max)
        self.a_pre[:, idx] = 0.

    def on_pre_a(self, idx: List[int]) -> None:
        """Updates pre-synaptic eligibility traces for spikes."""
        if self.approximate:
            self.a_pre[idx, :] = self.A_pre
        else:
            self.a_pre[idx, :] += self.A_pre

    def on_post_a(self, idx: List[int]) -> None:
        """Updates post-synaptic eligibility traces for"""
        if self.approximate:
            self.a_post[:, idx] = self.A_post
        else: self.a_post[:, idx] += self.A_post

    def update_a(self):
        self.a_pre = self.a_pre - self.dt / self.tau_pre * self.a_pre
        self.a_post = self.a_post - self.dt / self.tau_post * self.a_post

    def update_synapse(self,idx_post_spikinig, idx_pre_spiking):
        self.on_post_w(idx_post_spikinig)
        # synapse.on_post_test(idx, self.current_t)
        self.on_pre_w(idx_pre_spiking)
        # synapse.on_pre_test(spikes_i[idx_left:idx_right], self.current_t)    
        self.on_post_a(idx_post_spikinig)
        self.on_pre_a(idx_pre_spiking)
        # synapses.update_ti_tj_test(idx_pre=spikes_i[idx_left:idx_right], idx_post=idx_spike, current_t=self.current_t)
        self.update_a()

    def init_weight(self, M, N):
        """
        Creates a weight matrix of shape (M, N) with elements drawn from a uniform distribution (0, 1).

        Args:
            M: Number of presynaptic neurons (rows in the weight matrix).
            N: Number of postsynaptic neurons (columns in the weight matrix).

        Returns:
            A weight matrix of shape (M, N) with random values between 0 (inclusive) and 1 (exclusive).
        """
        weight = torch.random.uniform(low=0.0, high=1.0, size=(M, N))
        return weight

    def add_new_neurons(self,num_new_neurons):
        num_original_neurons = self.w.shape[1]
        new_neurons_weight = self.init_weight(M=2000, N=num_new_neurons)
        total_neurons = num_original_neurons + num_new_neurons
        
        self.w = torch.concatenate((self.w, new_neurons_weight), axis=1)

        self.A_post = torch.ones((total_neurons,)) * self.A_post
        self.A_post[:num_original_neurons] = 0
        self.A_pre = torch.ones((total_neurons,)) * self.A_pre
        self.A_pre[:num_original_neurons] = 0
        
        self.a_pre = torch.zeros_like(self.w)
        self.a_post = torch.zeros_like(self.w)

        self.on_post_a = self.on_post_a_new_neurons
    
    def on_post_a_new_neurons(self, idx):
        """
        idx is the index of postsynaptic neurons not in refractory period and their membrane potential above threshold
        """
        try:
            if self.approximate:
                self.a_post[:, idx] = self.A_post[idx]
            else: self.a_post[:, idx] += self.A_post[idx]
        except Exception as e:
            print(idx, type(idx))
            print(e)
            pass


    # def on_pre_test(self, idx, current_t):
    #     """
    #     LTD
    #     """
    #     # self.t_j[idx, :] = current_t
    #     delta_t = current_t - self.t_i[idx, :]
    #     dw = self.A_post * np.exp(-delta_t/self.tau_post)
    #     self.w[idx, :] = np.clip(self.w[idx, :] + dw, self.w_min, self.w_max)
    #     self.t_i[idx, :] = - np.inf

    # def on_post_test(self, idx, current_t):
    #     """
    #     LTP, A_pre
    #     """
    #     # self.t_i[:, idx.squeeze()] = current_t
    #     delta_t = (self.t_j[:, idx.squeeze()] - current_t)
    #     dw = self.A_pre * np.exp(delta_t/self.tau_pre)
    #     self.w[:, idx.squeeze()] = np.clip(self.w[:, idx.squeeze()] + dw, self.w_min, self.w_max)
    #     self.t_j[:, idx.squeeze()] = - np.inf

    # def update_ti_tj_test(self, idx_pre, idx_post, current_t):
    #     self.t_i[:, idx_post.squeeze()] = current_t
    #     self.t_j[idx_pre, :] = current_t

