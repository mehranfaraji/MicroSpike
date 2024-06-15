import numpy as np
from .base_model import BaseModel

class SRMInhibitory(BaseModel):
    def __init__(
            self, 
            N: int = None,
            threshold: float = None,
            reset: float = None,
            refractory: int = None, # refractory is an int number of time steps
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
        self.threshold = threshold
        self.reset = reset
        self.refractory = refractory
        self.ref_counter = np.zeros((self.N,1))
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.K1 = K1
        self.K2 = K2
        self.window_time = window_time
        ## TODO:dt should be given by user in net = Networ()
        self.dt = dt
        self.last_spike_time = - np.ones((self.N,1)) * np.inf

        self.monitor = None
    
            
    def forward(self, spikes_t: np.array, w_tmp: np.array, current_t: float, current_it, i: int):
        """
        i : postsynaptic neuron index
        """
        s = current_t + self.dt - spikes_t
        eps = self.eps_kernel(s)
        eps = eps * w_tmp
        eps = eps.sum()
        s = current_t + self.dt - self.last_spike_time[i]
        eta = self.eta_kernel(s)
        potential = eta + eps
        return potential
        
    def inhibitory_pulse(self, i):
        self.last_spike_time[i]

    
    
    def check_refractory(self):
        idx = self.ref_counter > 0
        self.ref_counter[idx] -= 1
        # self.ref_counter[idx] = self.ref_counter[idx]
        return ~idx  

    def start_refractory(self,idx, current_t):
        self.ref_counter[idx] = self.refractory
        self.last_spike_time[idx] = current_t
    
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