import torch
from torch import Tensor
from .base_model import BaseModel
from typing import Optional

class SRMInhibitory(BaseModel):
    """
    A model for a Spike Response Model (SRM) with inhibitory neurons.

    Attributes:
        N (Tensor): Number of neurons.
        inhibitory_connection (Tensor): Indicates if inhibitory connections exist.
        threshold (Tensor): The firing threshold.
        reset (Tensor): The reset potential after a spike.
        refractory (Tensor): Number of time steps a neuron remains refractory after spiking.
        alpha (Tensor): Scaling factor for inhibitory effects.
        k (Tensor): Number of neurons in the k-Winner-Take-All (KWTA) mechanism.
        tau_m (Tensor): Membrane time constant.
        tau_s (Tensor): Synaptic time constant.
        K1 (Tensor): Positive pulse constant.
        K2 (Tensor): Negative spike afterpotential constant.
        window_time (Tensor): Time window for ignoring old spikes.
        dt (Tensor): Simulation time step.
    """

    def __init__(
            self, 
            N: int,
            inhibitory_connection: bool,
            threshold: float,
            reset: float,
            refractory: int,  # Number of time steps
            alpha: float,
            k: int,
            tau_m: float,
            tau_s: float,
            K1: float,
            K2: float,
            window_time: float,
            dt: float,
            device: Optional[str]=None,

        ) -> None:
        """
        Initializes the SRM model with inhibitory neurons.
        """
        assert N is not None, "Number of neurons must be provided."
        
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'

        self.N = torch.tensor(N, device=self.device)
        self.inhibitory_connection = torch.tensor(inhibitory_connection, device=self.device, dtype=torch.bool)
        self.threshold = torch.tensor(threshold, device=self.device, dtype=torch.float32)
        self.reset = torch.tensor(reset, device=self.device, dtype=torch.float32)
        self.refractory = torch.tensor(refractory, device=self.device, dtype=torch.int32)
        self.alpha = torch.tensor(alpha, device=self.device, dtype=torch.float32)
        self.k = torch.tensor(k, device=self.device, dtype=torch.int32)
        self.tau_m = torch.tensor(tau_m, device=self.device, dtype=torch.float32)
        self.tau_s = torch.tensor(tau_s, device=self.device, dtype=torch.float32)
        self.K1 = torch.tensor(K1, device=self.device, dtype=torch.float32)
        self.K2 = torch.tensor(K2, device=self.device, dtype=torch.float32)
        self.window_time = torch.tensor(window_time, device=self.device, dtype=torch.float32)
        self.dt = torch.tensor(dt, device=self.device, dtype=torch.float32)

        # Initialize internal states
        self.ref_counter = torch.zeros((self.N, 1), device=self.device, dtype=torch.int32)
        self.last_spike_time = -torch.ones((self.N, 1), device=self.device, dtype=torch.float32) * 10e10
        self.potential = torch.zeros((self.N, 1), device=self.device, dtype=torch.float32)
        self.inhibitory_weights = torch.ones((self.N, self.N), device=self.device, dtype=torch.float32)
        self.monitor = None

    
    def forward(self, spikes_t: Tensor, w_tmp: Tensor, current_t: Tensor, i: int) -> Tensor:
        """
        Computes the membrane potential for the i-th postsynaptic neuron.

        Args:
            spikes_t (Tensor): Spike times of the presynaptic neurons.
            w_tmp (Tensor): Synaptic weights between presynaptic and postsynaptic neurons.
            current_t (Tensor): Current time in the simulation.
            i (int): Index of the postsynaptic neuron.

        Returns:
            Tensor: The updated membrane potential for the neuron.
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
        
    def mu_kernel(self, current_t: Tensor, i: int) -> Tensor:
        """
        Computes the inhibitory kernel for the i-th neuron.

        Args:
            current_t (Tensor): Current time in the simulation.
            i (int): Index of the postsynaptic neuron.

        Returns:
            Tensor: The inhibitory effect for the neuron.
        """
        s = (current_t + self.dt - self.last_spike_time) * (self.last_spike_time > self.last_spike_time[i])
        s = s * self.inhibitory_weights[:, i:i+1]
        mu = -self.alpha * self.threshold * self.eps_kernel(s)
        mu = mu.sum()
        return mu
    
    def check_refractory(self) -> Tensor:
        """
        Checks which neurons are not in the refractory period.

        Returns:
            Tensor: A boolean mask of neurons not in the refractory period.
        """
        idx = self.ref_counter > 0
        self.ref_counter[idx] -= 1
        return ~idx

    def start_refractory(self, idx: Tensor, current_t: Tensor) -> None:
        """
        Starts the refractory period for the spiking neurons.

        Args:
            idx (Tensor): Indices of neurons that spiked.
            current_t (Tensor): Current time in the simulation.
        """
        self.ref_counter[idx] = self.refractory
        self.last_spike_time[idx] = current_t
    
    def get_kwta(self) -> Tensor:
        """
        Implementation of the K-winner-take-all algorithm to determine neurons that can spike.

        Returns:
            Tensor: Indices of neurons that are spiking.
        """
        if self.inhibitory_connection:
            idx_bool = self.check_refractory()
            idx_bool = idx_bool * (self.potential >= self.threshold)
            if idx_bool.sum() == 0:
                return torch.empty(0, dtype=torch.int32, device=self.device)
            idx = (idx_bool * self.potential).squeeze(-1).argsort()[-self.k:]
        else:
            idx_bool = self.check_refractory()
            idx_bool = idx_bool * (self.potential >= self.threshold)
            idx = torch.where(idx_bool)[0]

        return idx
        
    def reset_neuron(self) -> None:
        """
        Resets the neuron to its initial state.
        """
        self.last_spike_time = -torch.ones((self.N, 1), device=self.device, dtype=torch.float32) * float('inf')
    
    def compute_K(self) -> None:
        """
        Computes the scaling constant K for the epsilon kernel.
        """
        s_max = (self.tau_m * self.tau_s) / (self.tau_s - self.tau_m) * torch.log(self.tau_s / self.tau_m)
        max_val = torch.exp(-s_max / self.tau_m) - torch.exp(-s_max / self.tau_s)
        self.K = 1 / max_val.item()
            
    def eps_kernel(self, s: Tensor) -> Tensor:
        """
        Computes the epsilon kernel for synaptic interactions.

        Args:
            s (Tensor): Time differences between spikes and current time.

        Returns:
            Tensor: Epsilon kernel values.
        """
        if not hasattr(self, 'K'):
            self.compute_K()
        return self.K * (torch.exp(-s / self.tau_m) - torch.exp(-s / self.tau_s))

    def eta_kernel(self, s: Tensor) -> Tensor:
        """
        Computes the eta kernel for refractory effects.

        Args:
            s (Tensor): Time differences between spikes and current time.

        Returns:
            Tensor: Eta kernel values.
        """
        positive_pulse = self.K1 * torch.exp(-s / self.tau_m)
        negative_spike_afterpotential = self.K2 * (torch.exp(-s / self.tau_m) - torch.exp(-s / self.tau_s))
        return self.threshold * (positive_pulse - negative_spike_afterpotential)
    
    def add_new_neurons(self, num_new_neurons: int) -> None:
        """
        Adds new neurons to the existing model.

        Args:
            num_new_neurons (int): Number of new neurons to add.
        """
        self.N += num_new_neurons
        self.num_new_neurons = num_new_neurons
        self.reset_internal_params()
        self.get_kwta = self.get_kwta_new_neurons
        
    def reset_internal_params(self) -> None:
        """
        Resets internal parameters when new neurons are added.
        """
        self.ref_counter = torch.zeros((self.N, 1), device=self.device, dtype=torch.float32)
        self.last_spike_time = -torch.ones((self.N, 1), device=self.device, dtype=torch.float32) * 10e10
        self.potential = torch.zeros((self.N, 1), device=self.device, dtype=torch.float32)
        self.inhibitory_weights = torch.ones((self.N, self.N), device=self.device, dtype=torch.float32)
        self.inhibitory_weights[self.N - self.num_new_neurons:, :self.N - self.num_new_neurons] = 0

    def get_kwta_new_neurons(self) -> Tensor:
        """
        KWTA algorithm considering new neurons, prioritizing original ones for spiking.

        Returns:
            Tensor: Indices of neurons that are spiking.
        """
        if self.inhibitory_connection:
            idx_bool = self.check_refractory()
            idx_bool = idx_bool * (self.potential >= self.threshold)
            if idx_bool.sum() == 0:
                return torch.empty(0, dtype=torch.int32, device=self.device)
            if idx_bool[:self.N - self.num_new_neurons].sum() > 0:
                idx = (idx_bool[:self.N - self.num_new_neurons] * self.potential[:self.N - self.num_new_neurons]).squeeze(-1).argsort()[-self.k:]
                return idx
            else:
                idx = (idx_bool * self.potential).squeeze(-1).argsort()[-self.k:]
        else:
            idx_bool = self.check_refractory()
            idx_bool = idx_bool * (self.potential >= self.threshold)
            idx = torch.where(idx_bool)[0]

        return idx
