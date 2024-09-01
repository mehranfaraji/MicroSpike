import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
from IPython.display import display
from microspike import PatternGenerator, InputTrain, SRMInhibitory, Synapse, Monitor, Network
import pickle


def investigate_potential(monitor, dt, position_copypaste, save_path=None):
    """
    Visualize and interact with the membrane potential of neurons over time.

    This function creates an interactive plot of the membrane potential for a selected neuron,
    allowing the user to choose the neuron and time range to display. It also highlights
    different patterns in the background.

    Parameters:
    -----------
    monitor : object
        An object containing the recorded potential data (monitor.potential_rec).
    dt : float
        Time step of the simulation.
    position_copypaste : array-like
        Array indicating the pattern positions.
    save_path : str, optional
        Path to save the generated plot. If None, the plot is not saved.

    Returns:
    --------
    None
        Displays an interactive plot using IPython widgets.
    """
    num_neurons = monitor.potential_rec.shape[0]
    patternlength = 0.050
    
    # Generate a larger set of distinct colors using HSV color space
    num_patterns = len(np.unique(position_copypaste)) - 1  # Subtract 1 to exclude 0
    hue_values = np.linspace(0, 1, num_patterns, endpoint=False)
    colors = [plt.cm.hsv(h) for h in hue_values]

    def update_plot(neuron_number, start_time, end_time):
        start_time = float(start_time)
        end_time = float(end_time)
        time_axis = np.arange(start_time, end_time, dt)
        
        fig, ax = plt.subplots(figsize=(15, 5))
        fig.patch.set_facecolor('#F0F0F0')
        ax.set_facecolor('#FFFFFF')
        
        # Plot membrane potential
        ax.plot(time_axis, monitor.potential_rec[neuron_number-1, int(start_time/dt):int(end_time/dt)], 
                linewidth=2, color='#1E88E5')
        
        ax.axhline(0, linestyle='--', color='#424242', alpha=0.7, linewidth=1)
        
        # Highlight patterns
        pattern_idx = np.where(position_copypaste > 0)[0]
        pattern_time = pattern_idx * patternlength
        ind = np.where((pattern_time >= start_time) & (pattern_time <= end_time))
        
        for time, idx in zip(pattern_time[ind], pattern_idx[ind]):
            pattern_number = position_copypaste[idx]
            ax.axvspan(time, time + patternlength, facecolor=colors[pattern_number-1], alpha=0.3, 
                       label=f'Pattern {pattern_number}')
        
        # Set labels and title
        ax.set_xlabel("Time (s)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Membrane Potential (a.u.)", fontsize=12, fontweight='bold')
        ax.set_title(f"Membrane Potential for Neuron {neuron_number}", fontsize=14, fontweight='bold')
        
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Create legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(sorted(zip(labels, handles), key=lambda x: int(x[0].split()[-1])))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # Create widgets
    style = {'description_width': 'initial'}
    slider_layout = widgets.Layout(width='300px')
    layout = widgets.Layout(width='200px')
    
    neuron_slider = widgets.IntSlider(value=1, min=1, max=num_neurons, step=1, description='Neuron:',
                                      style=style, layout=slider_layout, continuous_update=False)
    start_time_input = widgets.FloatText(value=0.0, description='Start Time (s):', style=style, layout=layout)
    end_time_input = widgets.FloatText(value=1.0, description='End Time (s):', style=style, layout=layout)
    update_button = widgets.Button(description='Update Plot', button_style='info', layout=layout)
    
    output = widgets.Output()

    def on_button_click(b):
        with output:
            output.clear_output(wait=True)
            update_plot(neuron_slider.value, start_time_input.value, end_time_input.value)

    update_button.on_click(on_button_click)

    # Arrange and display widgets
    controls = widgets.HBox([neuron_slider, start_time_input, end_time_input, update_button])
    display(widgets.VBox([controls, output]))
    
    # Initial plot
    with output:
        update_plot(neuron_slider.value, start_time_input.value, end_time_input.value)

def w_uniform(M, N):
  """
  Creates a weight matrix of shape (M, N) with elements drawn from a uniform distribution (0, 1).

  Args:
      M: Number of presynaptic neurons (rows in the weight matrix).
      N: Number of postsynaptic neurons (columns in the weight matrix).

  Returns:
      A weight matrix of shape (M, N) with random values between 0 (inclusive) and 1 (exclusive).
  """
  weight = np.random.uniform(low=0.0, high=1.0, size=(M, N))
  return weight

def generate_data_train_model(P, N, M, time, w_max, w_min, A_pre, A_post, tau_pre, tau_post, approximate,
                              threshold, reset, refractory, alpha, k, tau_m, tau_s, K1, K2, window_time,
                              inhibitory_connection):

  generator = PatternGenerator(number_pattern=P, number_neurons=M,
                                  total_pattern_freq = 1/3,
                                  )
  times, indices, position_copypaste, patterns_info, timing_pattern = generator.generate()


  weight = w_uniform(M=M, N=N)

  input_train = InputTrain(times, indices)

  model = SRMInhibitory(N=N,
              inhibitory_connection=inhibitory_connection,
              threshold=threshold,
              reset= reset,
              refractory=refractory,
              alpha= alpha,
              k = k,
              tau_m=tau_m,
              tau_s=tau_s,
              K1=K1,
              K2=K2,
              window_time= window_time,
              )
  synapse = Synapse(w=weight,
                  w_max=w_max,
                  w_min=w_min,
                  A_pre=A_pre,
                  A_post=A_post,
                  tau_pre=tau_pre,
                  tau_post=tau_post,
                  approximate= approximate
                  )
  monitor = Monitor(model)
  net = Network(dt=0.001)

  net.add_input_train(input_train)
  net.add_layer(model)
  net.add_synapse(synapse)

  net.run(time= time)
  return input_train, model, synapse, monitor, times, indices, position_copypaste, patterns_info, timing_pattern


def save_all_data(input_train, model, synapse, monitor, times, indices, position_copypaste, patterns_info, timing_pattern, filename):
    data = {
        'input_train': input_train,
        'model': model,
        'synapse': synapse,
        'monitor': monitor,
        'times': times,
        'indices': indices,
        'position_copypaste': position_copypaste,
        'patterns_info': patterns_info,
        'timing_pattern': timing_pattern
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_all_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return (data['input_train'], data['model'], data['synapse'], data['monitor'],
            data['times'], data['indices'], data['position_copypaste'],
            data['patterns_info'], data['timing_pattern'])