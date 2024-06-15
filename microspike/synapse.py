import numpy as np
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
                approximate: bool = False,
                dt : float = None
                ) -> None:
        self.w = np.copy(w)
        self.w_max = w_max
        self.w_min = w_min
        self.A_pre = A_pre
        self.A_post = A_post
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        ## if True do nearest spike approximation, else consider all the contributions of the previous presynaptic spikes
        self.approximate = approximate
        
        self.a_pre = np.zeros_like(self.w)
        self.a_post = np.zeros_like(self.w)
        self.dt = dt

        ### for test:
        self.t_j = - np.ones_like(self.w) * np.inf # for presynaptic neurons
        self.t_i = - np.ones_like(self.w) * np.inf # for postsynaptic neurons
        ########################################################################
        ### for test:
        self.w_log = np.zeros((self.w.shape[0], 100))

        if self.A_pre < 0:
            raise ValueError("A_pre should be > 0")
        if self.A_post > 0:
            raise ValueError("A_post should be < 0")
        if len(self.w.shape) < 2:
            raise ValueError("w should be 2D")
    
    def get_w_tmp(self, spikes_i: np.array, i):
        """
        spikes_i contains only the correct window of spikes_i
        """
        return self.w[spikes_i, i].squeeze().T

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
        self.w[:, idx.squeeze()] = np.clip(self.w[:, idx.squeeze()] + self.a_pre[:, idx.squeeze()], self.w_min, self.w_max)
        self.a_pre[:, idx.squeeze()] = 0.


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
            self.a_post[:, idx.squeeze()] = self.A_post
        else: self.a_post[:, idx.squeeze()] += self.A_post

    def update_a(self):
        self.a_pre = self.a_pre - self.dt / self.tau_pre * self.a_pre
        self.a_post = self.a_post - self.dt / self.tau_post * self.a_post

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
