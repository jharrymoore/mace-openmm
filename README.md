### MACE-openmm 

Code to provide interop functionality between [openmm](https://github.com/openmm/openmm) and the [MACE](https://github.com/ACEsuit/mace).

the `mace-md` entrypoint provides functionality for running molecular dynamics simulations with a hybrid Hamiltonian, in which the description of the system can be smoothly interpolated between the full MM and the mixed MM/ML description

### Currently supported running modes:
- vanilla MD: a specified ligand is described by the MACE potential
- Non-equilibrium switching: MM $\rightarrow$ MM/ML
- Equilibrium switching MM $\rightarrow$ MM/ML with replica exchange - supports parallelism over multiple GPUs


### Near-term developments
- Example scripts + notebooks


### References
- [Towards chemical accuracy for alchemical free energy calculations with hybrid physics-based machine learning / molecular mechanics potentials](https://doi.org/10.1101/2020.07.29.227959)


### Installation
- This package depends on both MACE and openmm.  We have found the provided environment to be installable from the config file on our hardware (cuda 11.4), _using mamba, not conda_

### Notes
- The code is currently distributed as follows:
	- Command line interface `mace-md` for running various types of simulation is provided when you install the `mace-openmm` package.  
	- The fork of openmm-ml specified in the environment.yml contains classes required to interface the mace model into openMM as a neural network potential, including creation of the TorchForce object and methods to create the mixed system 
	- Currently supported are vanilla MD, replica exchange equilibrium FEP and non-equilibrium switching.  Examples of the commands for each running mode, as well as accessing the functionality through the classes, is provided in the examples folder.


### Running the examples notebook
- On CSD3, follow the instructions [here](https://docs.hpc.cam.ac.uk/hpc/software-packages/jupyter.html?highlight=jupyter#running-jupyter-on-a-compute-node) to run a jupyter notebook instance on a compute node using the mlmm environment
