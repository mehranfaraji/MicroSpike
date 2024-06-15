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
        tmp_t = [current_t] * (idx)
        tmp_t = tmp_t[idx]
        self.spikes_t = np.append(self.spikes_t, tmp_t)
        tmp_i = np.where(idx)[0]
        self.spikes_i = np.append(self.spikes_i, tmp_i)  

    def record_potential(self, current_it, potential):
        self.potential_rec[:, current_it] = potential