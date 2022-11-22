import sys
from typing import List
from ase.io import read
import torch
from mace.calculators.openmm import MacePotentialImplFactory

import sys
from openmm import Platform, System, LangevinMiddleIntegrator, State, Vec3
from openmmtools.integrators import AlchemicalNonequilibriumLangevinIntegrator
from openmm.app import Simulation, Topology, StateDataReporter
from openmm.unit import kelvin, picosecond, femtosecond
from openmmml import MLPotential
import numpy as np

import sys
from scipy.optimize import fmin
import scipy.stats


MLPotential.registerImplFactory("mace", MacePotentialImplFactory())
torch.set_default_dtype(torch.float32)


# TODO:
# 1) extract equally spaced frames from the equilibrium simulation
# 2) take the snapshot and apply the protocol from run_md_mm_ml_coupling
# 3) sample from the equilibiurm distribution with probability given by the non-equilibrium work values to approximate the equilibrium ensemble from runnign an expensive MM/ML trajectory

# Outstanding issue is that we can't run the relative alchemical perturbation without the OE licence.  For now we probably want to use the gromacs equilibrium simulations approach from Icolos to generate the end state trajectories, then read these in with parmed or something.


class NEQCorrection:
    """
    Takes an equilibrium trajectory and computes the work distribution for the forward and reverse NEQ switching protocol to the MM/ML description of the system
    """

    # This will take the converted mixed system created by the plugin.

    equilibrium_system: System
    topology: Topology
    platform: Platform
    # Hold a list of snapshots to run the NEQ protocol on

    def __init__(
        self,
        system: System,
        topology: Topology,
        platform: Platform,
        # List of positions for the system compatible with OpenMM
        snapshots: List[Vec3],
    ) -> None:
        self.equilibrium_system = system
        # self.mixed_system = potential.createMixedSystem(topology=topology, system=system, atoms=ml_atoms, )
        self.topology = topology
        self.platform = platform
        # TODO: Should these just be positions or the full states? extracting positions seems cleaner for now
        self.snapshots = snapshots

        # containers to hold the data as it is generated
        self.forward_dG_estimates = []
        self.reverse_dG_estimates = []

    # For now, let's assume the snapshots were taken whilst the simulation was progressing, unlike the gromacs approch since there is not trajconv alternative
    # def take_snapshots(self, n: int) -> List[State]:
    #     """Takes equally spaced snapshots from the equilibrium trajectory

    #     :param int n: number of snapshots to take
    #     :return List[State]: List of openMM snapshots sampled from the trajectory.  Implicity Boltzmann distributed
    #     """
    #     # TODO: Not clear how best to do this right now
    #     pass

    def apply_ml_correction(self):
        # Main function to compute the corrective dG for the provided equilibirum simulation

        # Generate the distribution of forward work values for each of the snapshots
        for snapshot in self.snapshots:
            self.forward_dG_estimates.append(
                self.compute_neq_dG(
                    system=self.equilibrium_system,
                    topology=self.topology,
                    platform=self.platform,
                    positions=snapshot,
                )
            )
        # Now sample the snapshots with probability $exp^{-w_{MM/ML}}$ for the reverse work distribution
        # sample the same numeber of snapshots with replacement to initiate the reverse transitions (after decorrelation of each frame)
        reverse_samples = np.random.choice(
            self.snapshots,
            size=len(self.snapshots),
            replace=True,
            p=np.exp(-self.forward_dG_estimates),
        )

        # set up a regular equilibium to run the decorrelative job, return the final snapshot to start the reverse NEQ transition
        decorrelated_snapshots = []
        for sample in reverse_samples:
            decorrelated_snapshots.append(
                self.decorrelate_snapshot(
                    self.system, self.topology, self.platform, positions=sample
                ).getPositions()
            )

        # Now run the reverse NEQ simulation
        for rev_snapshot in decorrelated_snapshots:
            self.reverse_dG_estimates.append(
                self.compute_neq_dG(
                    system=self.system,
                    topology=self.topology,
                    platform=self.platform,
                    positions=rev_snapshot,
                )
            )

        ##############################################################################

        # The following is borrowed from https://github.com/deGrootLab/pmx of the PMX functionality for NEQ calculations
        bar = BAR(
            self.forward_dG_estimates,
            self.reverse_dG_estimates,
            T=298.15,
            nboots=100,
            nblocks=1,
        )
        self.dG, self.dG_err = bar.dg, bar.err

        # now plot the histograms to disk

        plot_work_dist(self.forward_dG_estimates, self.reverse_dG_estimates)

    def decorrelate_snapshot(
        self,
        system: System,
        topology: Topology,
        platform: Platform,
        positions: List,
        steps: int = 10000,
    ) -> State:
        print(f"Decorrelating snapshot...")

        temperature = 298.15 * kelvin
        frictionCoeff = 1 / picosecond
        timeStep = 1 * femtosecond
        integrator = LangevinMiddleIntegrator(temperature, frictionCoeff, timeStep)

        simulation = Simulation(
            topology,
            system,
            integrator,
            platform=platform,
            platformProperties={"Precision": "Single"},
        )
        simulation.context.setPositions(positions)
        # This will use the mixed system, need to set the context's lambda parameter to 1 to ensure this is done with the ML potential
        simulation.context.setParameter("lambda_interpolate", 1)
        state = simulation.context.getState(
            getEnergy=True,
            getVelocities=True,
            getParameterDerivatives=True,
            getForces=True,
            getPositions=True,
        )
        simulation.minimizeEnergy()
        simulation.step(steps)
        state = simulation.context.getState(getPositions=True)
        return state

    def compute_neq_dG(
        self,
        system: System,
        topology: Topology,
        platform: Platform,
        positions: List,
        forward: bool = True,
    ) -> float:
        """Takes an openMM system, creates a simulation environment and propagates the forward NES trajector

        :param System system: the OpenMM system taken from a snapshot of the equilibrium simulation
        :param Topology topology: the topology corresponding to the MM system
        :param Platform platform: compute platform on which to perform the simulation
        :param List positions: openMM positions from the modeller
        :return float: the dimentionless work value computed from the forward switching trajectory
        """

        # TODO: this should be parametrised
        temperature = 298.15 * kelvin
        frictionCoeff = 1 / picosecond
        timeStep = 1 * femtosecond
        # in the paper they do a 10 ps switching time, leaving at 1ps for testing
        n_step_neq = 1000

        # ensure lambda is running in the correct direction for the reverse transition
        alchemical_functions = (
            {"lambda_interpolate": "lambda"}
            if forward
            else {"lambda_interpolate": "1 - lambda"}
        )
        integrator = AlchemicalNonequilibriumLangevinIntegrator(
            alchemical_functions=alchemical_functions,
            nsteps_neq=n_step_neq,
            temperature=temperature,
            collision_rate=frictionCoeff,
            timestep=timeStep,
        )

        simulation = Simulation(
            topology,
            system,
            integrator,
            platform=platform,
            platformProperties={"Precision": "Single"},
        )
        simulation.context.setPositions(positions)
        simulation.minimizeEnergy()

        simulation.context.setVelocitiesToTemperature(temperature)

        reporter = StateDataReporter(
            file=sys.stdout,
            reportInterval=1000,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
            totalSteps=n_step_neq,
            remainingTime=True,
        )
        simulation.reporters.append(reporter)

        simulation.step(n_step_neq)
        neq_work = integrator.get_protocol_work(dimensionless=True)
        print("work done during switch from mm to ml", neq_work)
        return neq_work


# Taken from the PMX repo to give us some initial plotting capability


def plot_work_dist(
    wf,
    wr,
    fname="Wdist.png",
    nbins=20,
    dG=None,
    dGerr=None,
    units="kJ/mol",
    dpi=300,
    statesProvided="AB",
):

    from matplotlib import pyplot as plt

    """Plots forward and reverse work distributions. Optionally, it adds the
    estimate of the free energy change and its uncertainty on the plot.

    Parameters
    ----------
    wf : list
        list of forward work values.
    wr : list
        list of reverse work values.
    fname : str, optional
        filename of the saved image. Default is 'Wdist.png'.
    nbins : int, optional
        number of bins to use for the histogram. Default is 20.
    dG : float, optional
        free energy estimate.
    dGerr : float, optional
        uncertainty of the free energy estimate.
    units : str, optional
        the units of dG and dGerr. Default is 'kJ/mol'.
    dpi : int
        resolution of the saved image file.
    statesProvided: str
        work values for two states or only one

    Returns
    -------
    None

    """

    def smooth(x, window_len=11, window="hanning"):

        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")
        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than " "window size.")
        if window_len < 3:
            return x
        if window not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
            raise ValueError(
                "Window is on of 'flat', 'hanning', 'hamming', "
                "'bartlett', 'blackman'"
            )
        s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
        # moving average
        if window == "flat":
            w = np.ones(window_len, "d")
        else:
            w = eval("np." + window + "(window_len)")
        y = np.convolve(w / w.sum(), s, mode="same")
        return y[window_len - 1 : -window_len + 1]

    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    x1 = 0
    x2 = 0
    if "A" in statesProvided:
        x1 = range(len(wf))
    if "B" in statesProvided:
        x2 = range(len(wr))
    if x1 > x2:
        x = x1
    else:
        x = x2
    if "A" in statesProvided:
        mf, devf, Af = data2gauss(wf)
    if "B" in statesProvided:
        mb, devb, Ab = data2gauss(wr)

    if "AB" in statesProvided:
        mini = min(wf + wr)
        maxi = max(wf + wr)
        sm1 = smooth(np.array(wf))
        sm2 = smooth(np.array(wr))
        plt.plot(x1, wf, "g-", linewidth=2, label="Forward (0->1)", alpha=0.3)
        plt.plot(x1, sm1, "g-", linewidth=3)
        plt.plot(x2, wr, "b-", linewidth=2, label="Backward (1->0)", alpha=0.3)
        plt.plot(x2, sm2, "b-", linewidth=3)
    elif "A" in statesProvided:
        maxi = max(wf)
        mini = min(wf)
        sm1 = smooth(np.array(wf))
        plt.plot(x1, wf, "g-", linewidth=2, label="Forward (0->1)", alpha=0.3)
        plt.plot(x1, sm1, "g-", linewidth=3)
    elif "B" in statesProvided:
        maxi = max(wr)
        mini = min(wr)
        sm2 = smooth(np.array(wr))
        plt.plot(x2, wr, "b-", linewidth=2, label="Backward (1->0)", alpha=0.3)
        plt.plot(x2, sm2, "b-", linewidth=3)

    plt.legend(shadow=True, fancybox=True, loc="upper center", prop={"size": 12})
    plt.ylabel(r"W [kJ/mol]", fontsize=20)
    plt.xlabel(r"# Snapshot", fontsize=20)
    plt.grid(lw=2)
    plt.xlim(0, x[-1] + 1)
    xl = plt.gca()
    for val in xl.spines.values():
        val.set_lw(2)
    plt.subplot(1, 2, 2)
    plt.hist(
        wf,
        bins=nbins,
        orientation="horizontal",
        facecolor="green",
        alpha=0.75,
        density=True,
    )
    plt.hist(
        wr,
        bins=nbins,
        orientation="horizontal",
        facecolor="blue",
        alpha=0.75,
        density=True,
    )

    x = np.arange(mini, maxi, 0.5)

    if "AB" in statesProvided:
        y1 = gauss_func(Af, mf, devf, x)
        y2 = gauss_func(Ab, mb, devb, x)
        plt.plot(y1, x, "g--", linewidth=2)
        plt.plot(y2, x, "b--", linewidth=2)
        size = max([max(y1), max(y2)])
    elif "A" in statesProvided:
        y1 = gauss_func(Af, mf, devf, x)
        plt.plot(y1, x, "g--", linewidth=2)
        size = max(y1)
    elif "B" in statesProvided:
        y2 = gauss_func(Ab, mb, devb, x)
        plt.plot(y2, x, "b--", linewidth=2)
        size = max(y2)

    res_x = [dG, dG]
    res_y = [0, size * 1.2]
    if dG is not None and dGerr is not None:
        plt.plot(
            res_y,
            res_x,
            "k--",
            linewidth=2,
            label=r"$\Delta$G = %.2f $\pm$ %.2f %s" % (dG, dGerr, units),
        )
        plt.legend(shadow=True, fancybox=True, loc="upper center", prop={"size": 12})
    elif dG is not None and dGerr is None:
        plt.plot(
            res_y, res_x, "k--", linewidth=2, label=r"$\Delta$G = %.2f %s" % (dG, units)
        )
        plt.legend(shadow=True, fancybox=True, loc="upper center", prop={"size": 12})
    else:
        plt.plot(res_y, res_x, "k--", linewidth=2)

    plt.xticks([])
    plt.yticks([])
    xl = plt.gca()
    for val in xl.spines.values():
        val.set_lw(2)
    plt.subplots_adjust(wspace=0.0, hspace=0.1)
    plt.savefig(fname, dpi=dpi)


def gauss_func(A, mean, dev, x):
    """Given the parameters of a Gaussian and a range of the x-values, returns
    the y-values of the Gaussian function"""
    x = np.array(x)
    y = A * np.exp(-(((x - mean) ** 2.0) / (2.0 * (dev**2.0))))
    return y


def data2gauss(data):
    """Takes a one dimensional array and fits a Gaussian.

    Returns
    -------
    float
        mean of the distribution.
    float
        standard deviation of the distribution.
    float
        height of the curve's peak.
    """
    m = np.average(data)
    dev = np.std(data)
    A = 1.0 / (dev * np.sqrt(2 * np.pi))
    return m, dev, A


kb = 0.00831447215  # kJ/(K*mol)


class BAR(object):
    """Bennett acceptance ratio (BAR).

    Description...

    Parameters
    ----------

    Examples
    --------
    """

    def __init__(self, wf, wr, T, nboots=0, nblocks=1):
        self.wf = np.array(wf)
        self.wr = np.array(wr)
        self.T = float(T)
        self.nboots = nboots
        self.nblocks = nblocks

        self.nf = len(wf)
        self.nr = len(wr)
        self.beta = 1.0 / (kb * self.T)
        self.M = kb * self.T * np.log(float(self.nf) / float(self.nr))

        # Calculate all BAR properties available
        self.dg = self.calc_dg(self.wf, self.wr, self.T)
        self.err = self.calc_err(self.dg, self.wf, self.wr, self.T)
        if nboots > 0:
            self.err_boot = self.calc_err_boot(self.wf, self.wr, nboots, self.T)
        self.conv = self.calc_conv(self.dg, self.wf, self.wr, self.T)
        if nboots > 0:
            self.conv_err_boot = self.calc_conv_err_boot(
                self.dg, self.wf, self.wr, nboots, self.T
            )
        if nblocks > 1:
            self.err_blocks = self.calc_err_blocks(self.wf, self.wr, nblocks, self.T)

    @staticmethod
    def calc_dg(wf, wr, T):
        """Estimates and returns the free energy difference.

        Parameters
        ----------
        wf : array_like
            array of forward work values.
        wr : array_like
            array of reverse work values.
        T : float
            temperature

        Returns
        ----------
        dg : float
            the BAR free energy estimate.
        """

        nf = float(len(wf))
        nr = float(len(wr))
        beta = 1.0 / (kb * T)
        M = kb * T * np.log(nf / nr)

        def func(x, wf, wr):
            sf = 0
            for v in wf:
                sf += 1.0 / (1 + np.exp(beta * (M + v - x)))

            sr = 0
            for v in wr:
                sr += 1.0 / (1 + np.exp(-beta * (M + v - x)))

            r = sf - sr
            return r**2

        avA = np.average(wf)
        avB = np.average(wr)
        x0 = (avA + avB) / 2.0
        dg = fmin(func, x0=x0, args=(wf, wr), disp=0)

        return float(dg)

    @staticmethod
    def calc_err(dg, wf, wr, T):
        """Calculates the analytical error estimate.

        Parameters
        ----------
        dg : float
            the BAR free energy estimate
        wf : array_like
            array of forward work values.
        wr : array_like
            array of reverse work values.
        T : float
            temperature
        """

        nf = float(len(wf))
        nr = float(len(wr))
        beta = 1.0 / (kb * T)
        M = kb * T * np.log(nf / nr)

        err = 0
        for v in wf:
            err += 1.0 / (2 + 2 * np.cosh(beta * (M + v - dg)))
        for v in wr:
            err += 1.0 / (2 + 2 * np.cosh(beta * (M + v - dg)))
        N = nf + nr
        err /= float(N)
        tot = 1 / (beta**2 * N) * (1.0 / err - (N / nf + N / nr))

        err = float(np.sqrt(tot))
        return err

    @staticmethod
    def calc_err_boot(wf, wr, nboots, T):
        """Calculates the error by bootstrapping.

        Parameters
        ----------
        wf : array_like
            array of forward work values.
        wr : array_like
            array of reverse work values.
        T : float
            temperature
        nboots: int
            number of bootstrap samples.

        """

        nf = len(wf)
        nr = len(wr)
        dg_boots = []
        for k in range(nboots):
            sys.stdout.write(
                "\r  Bootstrap (Std Err): iteration %s/%s" % (k + 1, nboots)
            )
            sys.stdout.flush()

            bootA = np.random.choice(wf, size=nf, replace=True)
            bootB = np.random.choice(wr, size=nr, replace=True)
            dg_boot = BAR.calc_dg(bootA, bootB, T)
            dg_boots.append(dg_boot)

        sys.stdout.write("\n")
        err_boot = np.std(dg_boots)

        return err_boot

    @staticmethod
    def calc_err_blocks(wf, wr, nblocks, T):
        """Calculates the standard error based on a number of blocks the
        work values are divided into. It is useful when you run independent
        equilibrium simulations, so that you can then use their respective
        work values to compute the standard error based on the repeats.

        Parameters
        ----------
        wf : array_like
            array of forward work values.
        wr : array_like
            array of reverse work values.
        T : float
            temperature
        nblocks: int
            number of blocks to divide the data into. This can be for
            instance the number of independent equilibrium simulations
            you ran.
        """

        dg_blocks = []
        # loosely split the arrays
        wf_split = np.array_split(wf, nblocks)
        wr_split = np.array_split(wr, nblocks)

        # calculate all dg
        for wf_block, wr_block in zip(wf_split, wr_split):
            dg_block = BAR.calc_dg(wf_block, wr_block, T)
            dg_blocks.append(dg_block)

        # get std err
        err_blocks = scipy.stats.sem(dg_blocks, ddof=1)

        return err_blocks

    @staticmethod
    def calc_conv(dg, wf, wr, T):
        """Evaluates BAR convergence as described in Hahn & Then, Phys Rev E
        (2010), 81, 041117. Returns a value between -1 and 1: the closer this
        value to zero the better the BAR convergence.

        Parameters
        ----------
        dg : float
            the BAR free energy estimate
        wf : array_like
            array of forward work values.
        wr : array_like
            array of reverse work values.
        T : float
            temperature

        """

        wf = np.array(wf)
        wr = np.array(wr)

        beta = 1.0 / (kb * T)
        nf = len(wf)
        nr = len(wr)
        N = float(nf + nr)

        ratio_alpha = float(nf) / N
        ratio_beta = float(nr) / N
        bf = 1.0 / (ratio_beta + ratio_alpha * np.exp(beta * (wf - dg)))
        tf = 1.0 / (ratio_alpha + ratio_beta * np.exp(beta * (-wr + dg)))
        Ua = (np.mean(tf) + np.mean(bf)) / 2.0
        Ua2 = ratio_alpha * np.mean(np.power(tf, 2)) + ratio_beta * np.mean(
            np.power(bf, 2)
        )
        conv = (Ua - Ua2) / Ua
        return conv

    @staticmethod
    def calc_conv_err_boot(dg, wf, wr, nboots, T):
        nf = len(wf)
        nr = len(wr)
        conv_boots = []
        for k in range(nboots):
            sys.stdout.write(
                "\r  Bootstrap (Conv): " "iteration %s/%s" % (k + 1, nboots)
            )
            sys.stdout.flush()

            bootA = np.random.choice(wf, size=nf, replace=True)
            bootB = np.random.choice(wr, size=nr, replace=True)
            conv_boot = BAR.calc_conv(dg, bootA, bootB, T)
            conv_boots.append(conv_boot)

        sys.stdout.write("\n")
        err = np.std(conv_boots)
        return err
