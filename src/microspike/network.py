from .model.base_model import BaseModel
from .synapse import Synapse
from .input_train import InputTrain
from .monitor import Monitor
import numpy as np
from typing import List, Union

DEFAULT_DT = 0.001
DEFAULT_PATTERNLENGTH = 0.050

class NetworkBase():
    def __init__(
            self,
            train_mode: bool = True,
            dt = DEFAULT_DT,
            patternlength = DEFAULT_PATTERNLENGTH
            ) -> None:
        self.synapse:  Synapse
        self.layer: BaseModel
        self.input_train: Union[InputTrain, None] = None
        self.dt: float = dt
        self.patternlength = patternlength
        self.current_t: float = 0.0 ## also add in reset()!
        self.train_mode: bool = train_mode
        
    def add_synapse(self, synapse: Synapse):
        self.update_dt(synapse)
        self.synapse = synapse
      
    def add_layer(self, layer: BaseModel):
        self.update_dt(layer)
        if layer.monitor:
            self.update_dt(layer.monitor)
        self.layer = layer

    def add_input_train(self, input_train: InputTrain):
        self.input_train = input_train

    def reset_records(self,time):
        # TODO: remove the line below in the future
        # so that we can have multiple net.run() and it considers
        # current time of previous runs!
        self.current_t = 0.0
        self.current_it = 0
        if self.layer.monitor:
            self.layer.monitor.reset_records(time)

    def update_dt(self, obj):
        obj.dt = self.dt

    def get_idx_post_spiking(self):
        """
        this function will return the index of post-neuron that can spike and has the most potential
        """
        idx_spike = self.layer.get_kwta()

        return idx_spike
    
    def get_idx_pre_spiking(self): 
        # STDP LTD Rule
        idx_right = np.searchsorted(self.input_train.spikes_t, self.current_t, side='right')
        idx_left = np.searchsorted(self.input_train.spikes_t, self.current_t, side='left')
        return self.input_train.spikes_i[idx_left:idx_right]



    def update_synapse(self,idx_post: np.array, idx_pre: np.array, inside_wrong_pattern:Union[bool, None]):

        self.synapse.update_synapse(idx_post, idx_pre, inside_wrong_pattern)
        
    def update_monitor(self, monitor: Monitor, idx_post: np.array):
        if monitor:
            monitor.record_spike(self.current_t, idx_post)
            if monitor.potential_rec.shape[0] == 1:
                if len(idx_post) > 0:
                    monitor.potential_rec[0, self.current_it] = self.layer.eta_kernel(0.0)
            else:
                monitor.potential_rec[idx_post.squeeze(), self.current_it] = self.layer.eta_kernel(0.0)
        self.layer.start_refractory(idx_post, self.current_t)

        ## TODO: check if there's actually a bug in the code or input!
        ## TODO: checing if there is a pre-neuron that spikes more than once in one time step
        # if len(spikes_i[idx_left:idx_right]) != len(set(spikes_i[idx_left:idx_right])):
        #     print("There is a bug here!")
        #     print(idx_left, idx_right)
        #     print(spikes_i[idx_left:idx_right])
        #     print(spikes_t[idx_left:idx_right])
        #     print(len(spikes_i[idx_left:idx_right]), len(set(spikes_i[idx_left:idx_right])))

    def get_potential(self):
        spikes_t = self.input_train.spikes_t
        spikes_i = self.input_train.spikes_i
        new_potential = np.array([]) 
        for i in range(self.layer.N):
            
            start_idx = max(0, self.current_t - 7 * self.layer.tau_m, self.layer.last_spike_time[i].squeeze())
            start_slice = np.searchsorted(spikes_t, start_idx, side='left')
            end_slice = np.searchsorted(spikes_t, self.current_t, side='right')
            spikes_t_i = spikes_t[start_slice:end_slice]
            spikes_i_i = spikes_i[start_slice:end_slice]

            w_tmp = self.synapse.get_w_tmp(spikes_i_i, i)
            potential_i = self.layer.forward(spikes_t_i, w_tmp, self.current_t, i)
            new_potential = np.append(new_potential, potential_i)
        
        if self.layer.monitor:
            self.layer.monitor.record_potential(self.current_it + 1, new_potential)
        
        self.layer.potential = new_potential.reshape(-1,1)

    
    def add_new_neurons(self, num_original_patterns):
        self.num_original_patterns = num_original_patterns
        self.train_mode = False
    
    def get_punishment_signal(self):
        position_idx = int(np.round(
                (self.current_t - self.dt) / self.patternlength,
                decimals=3))
        pattern_num = self.input_train.position_copypaste[position_idx].item()
        inside_wrong_pattern = True if (pattern_num >= 1 and pattern_num <= self.num_original_patterns) else False
        return inside_wrong_pattern



class Network(NetworkBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def run_one_step(self):
        if not self.train_mode:
            ## inference time, detecting if we're inside a new pattern
            inside_wrong_pattern = self.get_punishment_signal()
        else:
            ## Primary training time
            inside_wrong_pattern = None 


        idx_post_spiking = self.get_idx_post_spiking()
        idx_pre_spiking = self.get_idx_pre_spiking()
        self.update_synapse(idx_post= idx_post_spiking, idx_pre= idx_pre_spiking, inside_wrong_pattern=inside_wrong_pattern)
        self.update_monitor(monitor= self.layer.monitor, idx_post= idx_post_spiking)
        self.get_potential()


    def run(self, time: float):
        """
        time (ms)
        """
        time = int(time * (DEFAULT_DT / self.dt))
        self.reset_records(time)
        for it in range(time):
            self.current_it = it
            self.current_t = it * self.dt
            self.current_t = np.round(self.current_t,3)
            self.run_one_step()
