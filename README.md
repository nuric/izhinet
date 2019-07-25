# izhinet

This repository contains a pure numpy based implementation of [Izhikevich neurons](https://www.izhikevich.org/publications/spikes.pdf) as a [spiking neural network](https://en.wikipedia.org/wiki/Spiking_neural_network). It is designed for clarity and simplicity, for complex simulations with larger network sizes have a look at [Brian2](https://brian2.readthedocs.io/en/stable/).

![spiketrain](https://raw.githubusercontent.com/nuric/izhinet/master/spike_train.png)

## Getting Started
Everything required to run a simulation and plot the spike trains are provided in `izhinet.py` with example code in `run.py` showing how to run. To install the dependecies you can:

```bash
pip3 install --no-cache-dir --upgrade -r requirements.txt
```

If you are [Anaconda](https://www.anaconda.com/) or other Python environment, you'll to install `numpy` and `matplotlib` following their instructions. Once you have the dependencies setup:

```bash
python3 run.py -h
	usage: run.py [-h] [-rt RUNTIME] [-dt DELTAT]

	Run random spiking neural networks.

	optional arguments:
		-h, --help            show this help message and exit
		-rt RUNTIME, --runtime RUNTIME
													Simulation runtime in milliseconds per input.
		-dt DELTAT, --deltat DELTAT
													Simulation delta time (dt), resolution.
```

For details on the simulation, you can refer to the [paper](https://www.izhikevich.org/publications/spikes.pdf) and the code in `izhinet.py` for the actual implementation.

## Limitations & To-Dos

 - Currently only the spikes / firings are stored, the state variables `v` and `u` can also be tracked albeit with extra memory usage.
 - The input current is fixed for the duration of the run, a timed input - one that changes at certain invervals - can be implemented to provide more flexibility.
 - It seems `numpy` runs on a single core, moving to a multi-threaded numerical computation library might be worth exploring.

## Built With

 - [NumPy](https://numpy.org/)
 - [matplotlib](https://matplotlib.org/)
