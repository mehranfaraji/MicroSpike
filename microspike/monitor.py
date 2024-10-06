import torch
from .model.base_model import BaseModel
from typing import Optional, List

class Monitor:
    """
    Monitor class for tracking spikes and membrane potentials in a neural network layer.

    Attributes:
        device (str): Device to use ('cuda' if available, otherwise 'cpu').
        spikes_t (Tensor): Times at which spikes occurred.
        spikes_i (Tensor): Indices of neurons that spiked.
        potential_rec (Tensor): Recorded membrane potentials over time.
        N (int): Number of neurons in the monitored layer.
        dt (float): Simulation time step.
    """

    def __init__(self, layer: BaseModel, device: Optional[str] = None) -> None:
        """
        Initializes the Monitor for the given neural network layer.

        Args:
            layer (BaseModel): The neural network layer to be monitored.
            device (str, optional): Device to use for computations ('cuda' or 'cpu').
        """
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'

        layer.monitor = self

        # Spike times and indices, initially empty
        self.spikes_t = torch.tensor([], device=self.device, dtype=torch.float32)
        self.spikes_i = torch.tensor([], device=self.device, dtype=torch.int64)
        self.potential_rec = torch.empty(0, device=self.device)

        self.N = layer.N  # Number of neurons in the layer
        self.dt = layer.dt  # Simulation time step

    def reset_records(self, time: float) -> None:
        """
        Resets the potential recording array based on the simulation time.

        Args:
            time (float): Total simulation time.
        """
        T = int(time) + 1
        self.potential_rec = torch.zeros((self.N, T), device=self.device)

    def record_spike(self, current_t: float, idx: torch.Tensor) -> None:
        """
        Records the spike times and indices for neurons that spike.

        Args:
            current_t (float): The current time at which spikes are being recorded.
            idx (Tensor): Indices of neurons that spiked.
        """
        tmp_t = torch.tensor([current_t] * len(idx), device=self.device, dtype=torch.float32)
        self.spikes_t = torch.cat([self.spikes_t, tmp_t])
        self.spikes_i = torch.cat([self.spikes_i, idx.to(self.device)])

    def record_potential(self, current_it: int, potential: torch.Tensor) -> None:
        """
        Records the membrane potential at the current iteration for all neurons.

        Args:
            current_it (int): Current iteration in the simulation.
            potential (Tensor): Membrane potentials of neurons at the current iteration.
        """
        self.potential_rec[:, current_it] = potential.to(self.device)

    def get_latencies(self, position_copypaste: List[bool], patternlength: float) -> List[torch.Tensor]:
        """
        Computes the latencies of spikes relative to the start of each pattern.

        Args:
            position_copypaste (List[bool]): Boolean list indicating valid pattern occurrences.
            patternlength (float): Length of the repeating pattern in the simulation.

        Returns:
            List[Tensor]: Latencies for each neuron in milliseconds.
        """
        latencies = [[] for _ in range(self.N)]

        for spike_time, spike_index in zip(self.spikes_t, self.spikes_i):
            indx_position = int(spike_time.item() / patternlength)
            is_pattern = position_copypaste[indx_position]

            latency = torch.round(spike_time - indx_position * patternlength, decimals=3).item()
            latency *= is_pattern
            latencies[int(spike_index)].append(latency)

        latencies = [torch.tensor(lat_list, device=self.device) * 1000 for lat_list in latencies]

        return latencies
