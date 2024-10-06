from .model.base_model import BaseModel
from .synapse import Synapse
from .input_train import InputTrain
from .monitor import Monitor
import torch
from typing import List, Optional

DEFAULT_DT = 0.001

class NetworkBase:
    """
    A base class for a spiking neural network simulation.

    Attributes:
        synapses (List[Synapse]): A list of synapses connecting different layers.
        layers (List[BaseModel]): A list of layers in the network.
        input_train (Optional[InputTrain]): Input train data for the network.
        dt (float): Time step for the simulation.
        current_t (float): The current time of the simulation.
        train_mode (bool): A flag indicating whether the network is in training mode.
    """
    def __init__(self, device: Optional[str]= None, train_mode: bool = True, dt: float = DEFAULT_DT,) -> None:
        self.device = device
        if not self.device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.synapses: List[Synapse] = []
        self.layers: List[BaseModel] = []
        self.input_train: Optional[InputTrain] = None
        self.dt: float = dt
        self.current_t: float = 0.0
        self.train_mode: bool = train_mode


    def add_synapse(self, synapse: Synapse) -> None:
        """Adds a synapse to the network and updates the time step."""
        self.update_dt(synapse)
        self.synapses.append(synapse)

    def add_layer(self, layer: BaseModel) -> None:
        """Adds a layer to the network and updates the time step."""
        self.update_dt(layer)
        if layer.monitor:
            self.update_dt(layer.monitor)
        self.layers.append(layer)

    def add_input_train(self, input_train: InputTrain) -> None:
        """Sets the input train data for the network."""
        self.input_train = input_train

    def reset_records(self, time: float) -> None:
        """Resets the monitoring records of all layers in the network."""
        self.current_t = 0.0
        self.current_it = 0
        for layer in self.layers:
            if layer.monitor:
                layer.monitor.reset_records(time)

    def update_dt(self, obj) -> None:
        """Updates the time step for an object (layer or synapse)."""
        obj.dt = self.dt

    def get_idx_post_spiking(self, layer: BaseModel) -> torch.Tensor:
        """Returns the indices of post-neurons that are most likely to spike."""
        idx_spike = layer.get_kwta()
        return idx_spike

    def get_idx_pre_spiking(self) -> torch.Tensor:
        """Returns the indices of pre-neurons that have spiked in the current time window."""
        idx_right = torch.searchsorted(self.input_train.spikes_t, torch.tensor([self.current_t]), side='right')
        idx_left = torch.searchsorted(self.input_train.spikes_t, torch.tensor([self.current_t]), side='left')
        return self.input_train.spikes_i[idx_left:idx_right]

    def update_synapse(self, layer: BaseModel, synapse: Synapse, idx_post: torch.Tensor, idx_pre: torch.Tensor) -> None:
        """Updates the synaptic weights based on the pre- and post-spiking activity."""
        synapse.update_synapse(idx_post, idx_pre)

    def update_monitor(self, layer: BaseModel, l: int, monitor: Monitor, idx_post: torch.Tensor) -> None:
        """Records spikes and updates the membrane potentials of neurons."""
        if monitor:
            monitor.record_spike(self.current_t, idx_post)
            if monitor.potential_rec.shape[0] == 1:
                if len(idx_post) > 0:
                    monitor.potential_rec[0, self.current_it] = layer.eta_kernel(0.0)
            else:
                monitor.potential_rec[idx_post.squeeze(), self.current_it] = layer.eta_kernel(0.0)
        layer.start_refractory(idx_post, self.current_t)

    def get_potential(self, layer: BaseModel, l: int) -> None:
        """Calculates and updates the membrane potential of neurons."""
        spikes_t = self.input_train.spikes_t
        spikes_i = self.input_train.spikes_i
        new_potential = torch.empty(0, device=self.device)
        
        for i in range(layer.N):
            start_idx = max(0, self.current_t - 7 * layer.tau_m, layer.last_spike_time[i].item())
            start_slice = torch.searchsorted(spikes_t, torch.tensor([start_idx]), side='left')
            end_slice = torch.searchsorted(spikes_t, torch.tensor([self.current_t]), side='right')
            spikes_t_i = spikes_t[start_slice:end_slice]
            spikes_i_i = spikes_i[start_slice:end_slice]

            w_tmp = self.synapses[l].get_w_tmp(spikes_i_i, i)
            potential_i = layer.forward(spikes_t_i, w_tmp, self.current_t, i)
            new_potential = torch.cat((new_potential, potential_i))

        if layer.monitor:
            layer.monitor.record_potential(self.current_it + 1, new_potential)
        
        layer.potential = new_potential.view(-1, 1)

class Network(NetworkBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_one_step(self) -> None:
        """Performs one simulation step for the network."""
        for l, layer in enumerate(self.layers):       
            idx_post_spiking = self.get_idx_post_spiking(layer)
            idx_pre_spiking = self.get_idx_pre_spiking()
            self.update_synapse(layer=layer, synapse=self.synapses[l], idx_post=idx_post_spiking, idx_pre=idx_pre_spiking)
            self.update_monitor(layer=layer, l=l, monitor=layer.monitor, idx_post=idx_post_spiking)
            self.get_potential(layer, l)

    def run(self, time: float) -> None:
        """
        Runs the network for a given period.

        Args:
            time (float): The duration in milliseconds for which the network should be simulated.
        """
        time_steps = int(time * (DEFAULT_DT / self.dt))
        self.reset_records(time_steps)
        for it in range(time_steps):
            self.current_it = it
            self.current_t = it * self.dt
            self.current_t = round(self.current_t, 3)
            self.run_one_step()

