import numpy as np
from .model.base_model import BaseModel

class Monitor():
    def __init__(self,
                 layer: BaseModel,
                 ) -> None:
        
        layer.monitor = self
        ## TODO: Is there a way to know how many times each of the postsynaptic neurons will spike?
        self.spikes_t = np.array([])
        self.spikes_i = np.array([])
        self.potential_rec = np.array([])
        self.N = layer.N
        self.dt = layer.dt

    def init_records(self, time):
        T = time + 1
        self.potential_rec = np.zeros((self.N, T))

    def record_spike(self, current_t, idx):
        """
        idx is the index number of the neuron (one neuron!) that is spiking
        """
        tmp_t = [current_t] * len(idx)
        self.spikes_t = np.append(self.spikes_t, tmp_t)
        self.spikes_i = np.append(self.spikes_i, idx)  

    def record_potential(self, current_it, potential):
        self.potential_rec[:, current_it] = potential