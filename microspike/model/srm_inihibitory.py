import numpy as np
from .base_model import BaseModel

class SRMInhibitory(BaseModel):
    def __init__(
            self, 
            N: int = None,
            inhibitory_connection: bool = None,
            threshold: float = None,
            reset: float = None,
            refractory: int = None, # refractory is an int number of time steps
            alpha: float = None,
            k : int = None,
            tau_m: float = None,
            tau_s: float = None,
            K1: float = None,
            K2: float = None,
            window_time: float = None, # Maximum time [second] to ignore inputs before it by kernels 
            dt: float = None,
        ) -> None:
        """
        """
        assert N is not None, "Number of neurons must be provided." # TODO: write better check for the class 

        self.N = N
        self.inhibitory_connection = inhibitory_connection
        self.threshold = threshold
        self.reset = reset
        self.refractory = refractory
        self.ref_counter = np.zeros((self.N,1))
        self.alpha = alpha
        self.k = k
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.K1 = K1
        self.K2 = K2
        self.window_time = window_time
        ## TODO:dt should be given by user in net = Networ()
        self.dt = dt
        self.last_spike_time = - np.ones((self.N,1)) * 10e10 # * np.inf
        self.potential = np.zeros((self.N, 1))
        self.inhibitory_weights = np.ones((self.N, self.N))
        self.monitor = None
    
            
    def forward(self, spikes_t: np.array, w_tmp: np.array, current_t: float, i: int):
        """
        i : postsynaptic neuron index
        """
        s = current_t + self.dt - spikes_t
        eps = self.eps_kernel(s)
        eps = eps * w_tmp
        eps = eps.sum()
        s = current_t + self.dt - self.last_spike_time[i]
        eta = self.eta_kernel(s)
        mu = self.mu_kernel(current_t, i)
        potential = eta + eps + mu
        return potential
        
    def mu_kernel(self, current_t: float, i: int):
        s = (current_t + self.dt - self.last_spike_time) * (self.last_spike_time > self.last_spike_time[i])
        s = s * self.inhibitory_weights[:, i:i+1]
        mu = - self.alpha * self.threshold * self.eps_kernel(s)
        mu = np.sum(mu)
        return mu
    
    
    def check_refractory(self):
        """
        returns the indices of the neurons that are not at refractory time.
        """
        idx = self.ref_counter > 0
        self.ref_counter[idx] -= 1
        # self.ref_counter[idx] = self.ref_counter[idx]
        return ~idx  

    def start_refractory(self,idx, current_t):
        self.ref_counter[idx] = self.refractory
        self.last_spike_time[idx] = current_t
    
    def get_kwta(self):
        """
        implementation of K-winner-take-all algorithm
        this function will return the index of post-neuron that can spike and has the most potential
        """
        if self.inhibitory_connection:
            idx_bool = self.check_refractory()
            idx_bool = idx_bool * (self.potential >= self.threshold)
            # if no neuron can spike, return an empty array
            if sum(idx_bool) == 0:
                return np.array([], dtype=np.int32)
            ## return the index of the neuron that is spiking
            idx = (idx_bool * self.potential).squeeze(axis=-1).argsort()[-self.k:]
        else:
            idx_bool = self.check_refractory()
            idx_bool = idx_bool * (self.potential >= self.threshold)
            idx = np.where(idx_bool)[0]

        return idx
        
    ## TODO: add reset to the network
    def reset_neuron(self,):
        self.t = 0
        self.last_spike_time = - np.ones((self.N,1)) * np.inf
    
    def compute_K(self):
        """
        K is chosen such that the maximum value of epsilon kernel will be 1, based on the tau_m and tau_s.
        """
        s_max = (self.tau_m * self.tau_s) / (self.tau_s - self.tau_m) * np.log(self.tau_s / self.tau_m)
        max_val = (np.exp(-s_max/self.tau_m) - np.exp(-s_max/self.tau_s))
        self.K = 1 / max_val
            
    def eps_kernel(self, s: np.array):
        """"
        s = t - t_j 
        time difference between thre current time and spike time of presynaptic neuron (t_j)
        """
        if not hasattr(self, 'K'):
            self.compute_K()
        return self.K * (np.exp(-s/self.tau_m) - np.exp(-s/self.tau_s))

    def eta_kernel(self, s: np.array):
        """
        s = t - t_i
        """
        positive_pulse = self.K1 * np.exp(-s/self.tau_m)
        negative_spike_afterpotential = self.K2 * (np.exp(-s/self.tau_m) - np.exp(-s/self.tau_s))
        return self.threshold * (positive_pulse - negative_spike_afterpotential)
    
    def add_new_neurons(self, num_new_neurons):
        self.N += num_new_neurons
        self.num_new_neurons = num_new_neurons
        ## attributes that need update
        self.reset_internal_params()
        ## change how kwta algorithm works. (it prioritizes spiking of original neurons over new ones)
        self.get_kwta = self.get_kwta_new_neurons
        
    def reset_internal_params(self):
        self.ref_counter = np.zeros((self.N,1))
        self.last_spike_time = - np.ones((self.N,1)) * 10e10
        self.potential = np.zeros((self.N,1))
        self.inhibitory_weights = np.ones((self.N, self.N))
        ## new neurons cannot send inhibitory signals to the original neurons
        self.inhibitory_weights[self.N - self.num_new_neurons: , :self.N - self.num_new_neurons] = 0

    def get_kwta_new_neurons(self):
        """
        implementation of K-winner-take-all algorithm
        this function will return the index of post-neuron that can spike and has the most potential
        """
        if self.inhibitory_connection:
            idx_bool = self.check_refractory()
            idx_bool = idx_bool * (self.potential >= self.threshold)
            # if no neuron can spike, return an empty array
            if sum(idx_bool) == 0:
                return np.array([], dtype=np.int32)
            #####################################################################################
            # ## if at least one of the original neurons can spike
            # ## since we don't want to train it anymore then we will return empty array
            # elif sum(idx_bool[:self.N - self.num_new_neurons]) > 0:
            #     return np.array([], dtype=np.int32 )
            #####################################################################################
            ## return the index of the neuron that is spiking
            ## if at least one of original neurons can spike 
            elif sum(idx_bool[:self.N - self.num_new_neurons]) > 0:
                idx = (idx_bool[:self.N - self.num_new_neurons] * self.potential[:self.N - self.num_new_neurons]).squeeze(axis=-1).argsort()[-self.k:]
                return idx
            
            ## if no original neurons can spike
            else:
                idx = (idx_bool * self.potential).squeeze(axis=-1).argsort()[-self.k:]
        else:
            idx_bool = self.check_refractory()
            idx_bool = idx_bool * (self.potential >= self.threshold)
            idx = np.where(idx_bool)[0]

        return idx