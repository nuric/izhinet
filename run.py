"""Run random spiking neural networks."""
import argparse
from timeit import default_timer as timer
import numpy as np

from izhinet import izhinet, plot_spikes

# Disable scientific printing
np.set_printoptions(suppress=True, precision=3, linewidth=180)

# ---------------------------

# Arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-rt", "--runtime", type=int, default=500, help="Simulation runtime in milliseconds per input.")
parser.add_argument("-dt", "--deltat", type=float, default=0.1, help="Simulation delta time (dt), resolution.")
ARGS = parser.parse_args()

# ---------------------------

# Debug run with no connections
net = dict()
net['ntypes'] = np.array([True, True]) # (N,)
net['nrands'] = np.array([0, 0]) # (N,)
net['weights'] = np.zeros((2, 2)) # (N, N)
net['delays'] = np.zeros((2, 2), dtype=np.int32) # (N, N)

current = np.array([[36, 12]]) # (B, N)

tstart = timer()
# Simulate for 500 milliseconds with 0.1 delta t updates
spikes = izhinet(net, current, ARGS.runtime, ARGS.deltat)
print(f"Spike count: {spikes.sum(-1)}")
print(f"Simulation took {timer() - tstart} seconds.")
plot_spikes(spikes, show_plot=True)

# ---------------------------

# Random run with connections
neurons = 400
net['ntypes'] = np.random.rand(neurons) < 0.50 # (N,)
net['nrands'] = np.random.rand(neurons) # (N,)
net['weights'] = np.random.uniform(0, 1, size=(neurons, neurons)) # (N, N)
# net['weights'] = np.random.normal(5, 10, size=(neurons, neurons)) # (N, N)
net['weights'] = np.clip(net['weights'], 0, None) # (N, N)
# Inhibitory neurons have negative weights
net['weights'] = net['weights']*net['ntypes'][..., None] - np.logical_not(net['ntypes'][..., None])*net['weights']
net['delays'] = np.random.randint(11, size=(neurons, neurons)) # (N, N)

current = np.random.randint(2, size=(7, neurons), dtype=np.bool) # (B, N)
current = current*12 + (1-current)*5 # units mV

tstart = timer()
spikes = izhinet(net, current, ARGS.runtime, ARGS.deltat)
print(f"Simulation took {timer() - tstart} seconds.")
plot_spikes(spikes, show_plot=True)
