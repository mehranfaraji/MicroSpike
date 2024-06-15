from .model.base_model import BaseModel
from .synapse import Synapse
from .input_train import InputTrain
from .monitor import Monitor
import numpy as np
from typing import List, Optional

DEFAULT_DT = 0.001

class NetworkBase():
    def __init__(
            self,
            train_mode: bool = True,
            dt = DEFAULT_DT
            ) -> None:
        self.synapses: List[Synapse] = []
        self.layers: List[BaseModel] = []
        self.potential: List[np.ndarray] = []
        self.input_train: Optional[InputTrain] = None
        self.dt: float = dt
        self.current_t: float = 0.0 ## also add in reset()!
        self.train_mode: bool = train_mode
        
    def add_synapse(self, synapse: Synapse):
        self.update_dt(synapse)
        self.synapses.append(synapse)
      
    def add_layer(self, layer: BaseModel):
        self.update_dt(layer)
        if layer.monitor:
            self.update_dt(layer.monitor)
        self.layers.append(layer)
        self.potential.append(np.zeros((layer.N, 1)))

    def add_input_train(self, input_train: InputTrain):
        self.input_train = input_train

    def init_records(self,time):
        # TODO: remove the line below in the future
        # so that we can have multiple net.run() and it considers
        # current time of previous runs!
        self.current_t = 0.0
        self.current_it = 0
        for layer in self.layers:
            if layer.monitor:
                layer.monitor.init_records(time)

    def update_dt(self, obj):
        obj.dt = self.dt

    def check_post_spike(self, layer: BaseModel, l: int, synapse: Synapse, monitor: Monitor):

        # idx : index of neurons not in refractory period.
        idx = layer.check_refractory()
        # idx_spike is the index of the neurons not in refractory period and their membrane potential above threshold
        idx_spike = self.potential[l] >= layer.threshold
        idx_spike = idx * idx_spike
        synapse.on_post_w(idx_spike)
        # synapse.on_post_test(idx, self.current_t)
        
        if monitor:
            monitor.record_spike(self.current_t, idx_spike)
            if monitor.potential_rec.shape[0] == 1:
                if idx_spike.squeeze() == True:
                    monitor.potential_rec[0, self.current_it] = layer.eta_kernel(0.0)
            else:
                monitor.potential_rec[idx_spike.squeeze(), self.current_it] = layer.eta_kernel(0.0)
        layer.start_refractory(idx_spike, self.current_t)

    def get_potential(self, layer, l):
        spikes_t = self.input_train.spikes_t
        spikes_i = self.input_train.spikes_i
        new_potential = np.array([]) 
        for i in range(layer.N):
            
            start_idx = max(0, self.current_t - 7 * layer.tau_m, layer.last_spike_time[i].squeeze())
            start_slice = np.searchsorted(spikes_t, start_idx, side='left')
            end_slice = np.searchsorted(spikes_t, self.current_t, side='right')
            spikes_t_i = spikes_t[start_slice:end_slice]
            spikes_i_i = spikes_i[start_slice:end_slice]

            w_tmp = self.synapses[l].get_w_tmp(spikes_i_i, i)
            potential_i = layer.forward(spikes_t_i, w_tmp, self.current_t, self.current_it, i)
            new_potential = np.append(new_potential, potential_i)
        
        if layer.monitor:
            layer.monitor.record_potential(self.current_it + 1, new_potential)
        self.potential = np.array(new_potential)


class Network(NetworkBase):
    def __init__(self,):
        super().__init__()
    
    def run_one_step(self):
        spikes_t = self.input_train.spikes_t
        spikes_i = self.input_train.spikes_i  
        ## WARNING:
        ## TODO: There is a bug here when a deep network is defined.
        ## here istead of a usinng spike_t which is from the InputTrain
        ## we should instead use the spike train from previous layer not the input!
        for l, layer in enumerate(self.layers):
                        
            self.check_post_spike(layer=layer, l=l, synapse=self.synapses[l], monitor=layer.monitor)
            self.get_potential(layer, l)

            # STDP LTD Rule
            idx_right = np.searchsorted(spikes_t, self.current_t, side='right')
            idx_left = np.searchsorted(spikes_t, self.current_t, side='left')

            ## TODO: check if there's actually a bug in the code or input!
            ## TODO: checing if there is a pre-neuron that spikes more than once in one time step
            # if len(spikes_i[idx_left:idx_right]) != len(set(spikes_i[idx_left:idx_right])):
            #     print("There is a bug here!")
            #     print(idx_left, idx_right)
            #     print(spikes_i[idx_left:idx_right])
            #     print(spikes_t[idx_left:idx_right])
            #     print(len(spikes_i[idx_left:idx_right]), len(set(spikes_i[idx_left:idx_right])))

            self.synapses[l].on_pre_w(spikes_i[idx_left:idx_right])
            # self.synapses[l].on_pre_test(spikes_i[idx_left:idx_right], self.current_t)    

            idx = layer.check_refractory()
            idx_spike = self.potential[l] >= layer.threshold
            idx_spike = idx * idx_spike
            self.synapses[l].on_post_a(idx_spike)
            self.synapses[l].on_pre_a(spikes_i[idx_left:idx_right])
            # self.synapses[l].update_ti_tj_test(idx_pre=spikes_i[idx_left:idx_right], idx_post=idx_spike, current_t=self.current_t)

            self.synapses[l].update_a()

            #### save logs 
            # self.synapses[l].monitor.record_weight()
            try:
                self.synapses[l].w_log[:,self.current_it] = self.synapses[l].w[:,0]
            except:
                pass

    def run(self, time: float):
        """
        time (ms)
        """
        self.init_records(time)
        ## Check the behaviour of + self.dt on the last loop step
        for it in range(time):
            self.current_it = it
            self.current_t = it * self.dt
            self.current_t = np.round(self.current_t,3)
            self.run_one_step()

            # if it % 20000 == 0:
            #     print(f"{it = }")
