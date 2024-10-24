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

    def reset_records(self, time):
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

    def get_latencies(self,position_copypaste, patternlength):
        latencies = [[] for _ in range(self.N)]

        for spike_time, spike_index in zip(self.spikes_t, self.spikes_i):
            indx_position = int(spike_time / patternlength)
            is_pattern = position_copypaste[indx_position]

            latency = np.round(spike_time - (indx_position) * patternlength, decimals=3).item()
            latency = latency * is_pattern
            latencies[int(spike_index)].append(latency)
        
        latencies = [np.array(lat_list) * 1000 for lat_list in latencies]


        return latencies