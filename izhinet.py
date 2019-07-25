"""Manual numpy based implementation of spiking neural network."""
import numpy as np
import matplotlib.pyplot as plt


def izhinet(params: dict, in_current: np.ndarray, runtime: int, deltat: float) -> np.ndarray:
  """Simulate Izhikevich Networks with given parameters."""
  # params['ntypes'] (N,) True for excitatory, False for inhibitory
  # params['nrands'] (N,)
  # params['weights'] (N, N)
  # params['delays'] (N, N)
  # in_current (B, N)
  ntypes = params['ntypes'] # (N,)
  nrands = params['nrands'] # (N,)
  # We will look back in time, so need to transpose these
  recv_weights = params['weights'].T # (N, N)
  recv_delays = params['delays'].T # (N, N)
  # ---------------------------
  # Setup variables
  bs = in_current.shape[0] # batch size B
  ns = ntypes.shape[0] # number of neurons N
  ns_range = np.arange(ns) # (N,)

  firings = np.zeros((bs, ns, runtime), dtype=np.bool) # (B, N, T)

  # https://www.izhikevich.org/publications/spikes.pdf
  # Neuron parameters as described in the paper
  a = ntypes*0.02 + (1-ntypes)*(0.02+0.08*nrands) # (N,)
  b = ntypes*0.2 + (1-ntypes)*(0.25-0.5*nrands) # (N,)
  nrsquared = nrands*nrands # (N,)
  c = ntypes*(-65+15*nrsquared) + (1-ntypes)*-65 # (N,)
  d = ntypes*(8-6*nrsquared) + (1-ntypes)*2 # (N,)
  a, b, c, d = [np.repeat(x[None], bs, axis=0) for x in (a, b, c, d)] # (B, N)

  # Runtime state of neurons, v is the membrane voltage
  v = np.ones((bs, ns), dtype=np.float32)*-65 # (B, N)
  u = v * b # (B, N)
  # ---------------------------
  for t in range(runtime): # milliseconds
    # Compute input current
    past = t-recv_delays # (N, N)
    # This is okay because nothing has fired at the current time yet
    past[past < 0] = t # reset negative values to current time
    # Look back in time for neurons firing
    past_fired = firings[:, ns_range[None, :], past] # (B, N, N)
    icurrent = (past_fired*recv_weights).sum(-1) # (B, N)
    icurrent += in_current # (B, N)
    # ---------------------------
    fired = firings[..., t] # (B, N)
    # Integrate using the Euler method
    for _ in range(int(1/deltat)): # delta t to update differential equations
      # To avoid overflows with large input currents,
      # keep updating only neurons that haven't fired this millisecond.
      notfired = np.logical_not(fired) # (B, N)
      nfv, nfu = v[notfired], u[notfired] # (NF,), (NF,)
      # https://www.izhikevich.org/publications/spikes.pdf
      v[notfired] += deltat*(0.04*nfv*nfv + 5*nfv + 140 - nfu + icurrent[notfired]) # (B, N)
      u[notfired] += deltat*(a[notfired]*(b[notfired]*nfv - nfu)) # (B, N)
      # Update firings
      fired[:] = np.logical_or(fired, v >= 30) # threshold potential in mV
    # ---------------------------
    # Reset for next millisecond
    v[fired] = c[fired] # (F,)
    u[fired] += d[fired] # (F,)
  return firings

def plot_spikes(firings: np.ndarray, batch_idx: int = 0, outf: str = None, show_plot: bool = False):
  """Plot spikes for given batch and firings tensor."""
  # firings (B, N, T)
  spikes = firings[batch_idx] # (N, T)
  rows, cols = np.nonzero(spikes)
  plt.plot(cols, rows, '.')
  plt.xlabel('Time (ms)')
  plt.xlim(0, firings.shape[-1])
  plt.ylabel('Neuron Index')
  if outf:
    plt.savefig(outf, bbox_inches='tight')
  if show_plot:
    plt.show()
