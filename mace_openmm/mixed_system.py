import sys
from ase.io import read
import torch
import time
import numpy as np
import sys
from ase import Atoms
from openmm.openmm import System
from typing import List, Tuple, Optional
from openmm import LangevinMiddleIntegrator
from openmmtools.integrators import AlchemicalNonequilibriumLangevinIntegrator
from openmm.app import (
    Simulation,
    StateDataReporter,
    PDBReporter,
    ForceField,
    PDBFile,
    HBonds,
    Modeller,
    PME,
)
from openmm.unit import nanometer, nanometers, molar, angstrom
from openmm.unit import kelvin, picosecond, femtosecond, kilojoule_per_mole, picoseconds, femtoseconds
from openff.toolkit.topology import Molecule

from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from mace.calculators.openmm import MacePotentialImplFactory
from openmmml import MLPotential

from openmmtools.openmm_torch.repex import (
    MixedSystemConstructor,
    RepexConstructor,
    get_atoms_from_resname,
)
from tempfile import mkstemp
import os
import logging


def get_xyz_from_mol(mol):

    xyz = np.zeros((mol.GetNumAtoms(), 3))
    conf = mol.GetConformer()
    for i in range(conf.GetNumAtoms()):
        position = conf.GetAtomPosition(i)
        xyz[i, 0] = position.x
        xyz[i, 1] = position.y
        xyz[i, 2] = position.z
    return xyz


MLPotential.registerImplFactory("mace", MacePotentialImplFactory())
# platform = Platform.getPlatformByName("CUDA")
# platform.setPropertyDefaultValue("DeterministicForces", "true")

logger = logging.getLogger("INFO")
SM_FF = "openff_unconstrained-1.0.0.offxml"


class MixedSystem:
    def __init__(
        self,
        file: str,
        ml_mol: str,
        model_path: str,
        forcefields: List[str],
        resname: str,
        padding: float,
        ionicStrength: float,
        nonbondedCutoff: float,
        potential: str,
        temperature: float,
        repex_storage_path: str,
        dtype: torch.dtype,
        neighbour_list: str,
        friction_coeff: float = 1.0,
        timestep: float = 1,
        pure_ml_system: bool = False
    ) -> None:

        self.forcefields = forcefields
        self.padding = padding
        self.ionicStrength = ionicStrength
        self.nonbondedCutoff = nonbondedCutoff
        self.resname = resname
        self.potential = potential
        self.temperature = temperature
        self.friction_coeff = friction_coeff / picosecond
        self.timestep = timestep * femtosecond
        self.repex_storage_path = repex_storage_path
        self.dtype = dtype
        self.neighbour_list = neighbour_list
        self.openmm_precision = "Double" if dtype == torch.float64 else "Mixed"
        logger.debug(f"OpenMM will use {self.openmm_precision} precision")

        self.mixed_system, self.modeller = self.create_mixed_system(
            file=file, ml_mol=ml_mol, model_path=model_path, pure_ml_system=pure_ml_system
        )

    def initialize_mm_forcefield(self, molecule: Optional[Molecule] = None) -> ForceField:

        forcefield = ForceField(*self.forcefields)
        if molecule is not None:
            # Ensure we use unconstrained force field
            smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule, forcefield=SM_FF)
            forcefield.registerTemplateGenerator(smirnoff.generator)
        return forcefield

    def initialize_ase_atoms(self, ml_mol: str) -> Tuple[Atoms, Molecule]:
        # ml_mol can be a path to a file, or a smiles string
        if os.path.isfile(ml_mol):
            molecule = Molecule.from_file(ml_mol)
        else:
            molecule = Molecule.from_smiles(ml_mol)

        _, tmpfile = mkstemp(suffix="xyz")
        molecule._to_xyz_file(tmpfile)
        atoms = read(tmpfile)
        os.remove(tmpfile)
        return atoms, molecule


    def create_mixed_system(
        self,
        file: str,
        model_path: str,
        ml_mol: str,
        pure_ml_system: bool = False,
    ) -> Tuple[System, Modeller]:
        """Creates the mixed system from a purely mm system

        :param str file: input pdb file
        :param str smiles: smiles of the small molecule, only required when passed as part of the complex
        :param str model_path: path to the mace model
        :return Tuple[System, Modeller]: return mixed system and the modeller for topology + position access by downstream methods
        """
        # initialize the ase atoms for MACE
        
        atoms, molecule = self.initialize_ase_atoms(ml_mol)

        # Handle a complex, passed as a pdb file
        if file.endswith(".pdb"):
            input_file = PDBFile(file)
            topology= input_file.getTopology()

            # if pure_ml_system specified, we just need to parse the input file
            # if not pure_ml_system:
            modeller = Modeller(input_file.topology, input_file.positions)
            print(f"Initialized topology with {len(input_file.positions)} positions")

        # Handle a ligand, passed as an sdf, override the Molecule initialized from smiles
        elif file.endswith(".sdf"):
            molecule = Molecule.from_file(file, allow_undefined_stereo=True)
            input_file = molecule
            topology = molecule.to_topology().to_openmm()
            # Hold positions in nanometers
            positions = get_xyz_from_mol(molecule.to_rdkit()) / 10

            print(f"Initialized topology with {positions.shape} positions")

            modeller = Modeller(molecule.to_topology().to_openmm(), positions)
        if pure_ml_system:
            # we have the input_file, create the system directly from the mace potential
            # modeller = None
            # atoms.set_cell([50,50,50])
            ml_potential = MLPotential("mace")
            ml_system = ml_potential.createSystem(
                topology, atoms_obj=atoms, filename=model_path, dtype=self.dtype
            )
            return ml_system, modeller
        
        # Handle the mixed systems with a classical forcefield
        else:
            forcefield = self.initialize_mm_forcefield(molecule=molecule)
            modeller.addSolvent(
                forcefield,
                padding=self.padding * nanometers,
                ionicStrength=self.ionicStrength * molar,
                neutralize=True,
            )

            omm_box_vecs = modeller.topology.getPeriodicBoxVectors()

            atoms.set_cell(
                [
                    omm_box_vecs[0][0].value_in_unit(angstrom),
                    omm_box_vecs[1][1].value_in_unit(angstrom),
                    omm_box_vecs[2][2].value_in_unit(angstrom),
                ]
            )

            mm_system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=PME,
                nonbondedCutoff=self.nonbondedCutoff * nanometer,
                constraints= None,
            )

            mixed_system = MixedSystemConstructor(
                system=mm_system,
                topology=modeller.topology,
                nnpify_resname=self.resname,
                nnp_potential=self.potential,
                atoms_obj=atoms,
                filename=model_path,
                dtype=self.dtype,
                nl=self.neighbour_list,
            ).mixed_system

        return mixed_system, modeller

    def run_mixed_md(self, steps: int, interval: int, output_file: str) -> float:
        """Runs plain MD on the mixed system, writes a pdb trajectory

        :param int steps: number of steps to run the simulation for
        :param int interval: reportInterval attached to reporters
        """
        integrator = LangevinMiddleIntegrator(
            self.temperature, self.friction_coeff, self.timestep
        )

        simulation = Simulation(
            self.modeller.topology,
            self.mixed_system,
            integrator,
            platformProperties={"Precision": self.openmm_precision},
        )
        simulation.context.setPositions(self.modeller.getPositions())

        logging.info("Minimising energy")
        simulation.minimizeEnergy()

        reporter = StateDataReporter(
            file=sys.stdout,
            reportInterval=interval,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
        )
        simulation.reporters.append(reporter)
        simulation.reporters.append(
            PDBReporter(
                file=output_file,
                reportInterval=interval,
            )
        )

        simulation.step(steps)
        state = simulation.context.getState(getEnergy=True)
        energy_2 = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        return energy_2

    def run_replex_equilibrium_fep(self, replicas: int, restart: bool, steps: int) -> None:
        del os.environ["SLURM_PROCID"]
        sampler = RepexConstructor(
            mixed_system=self.mixed_system,
            initial_positions=self.modeller.getPositions(),
            # repex_storage_file="./out_complex.nc",
            temperature=self.temperature * kelvin,
            n_states=replicas,
            restart=restart,
            mcmc_moves_kwargs={
                "timestep": 1.0*femtoseconds,
                "collision_rate": 1.0 / picoseconds,
                "n_steps": 1000,
                "reassign_velocities": True
            },
            replica_exchange_sampler_kwargs={
                "number_of_iterations": steps,
                "online_analysis_interval": 10,
                "online_analysis_minimum_iterations": 10
            },
            storage_kwargs={
                "storage": self.repex_storage_path,
                "checkpoint_interval": 100,
                "analysis_particle_indices": get_atoms_from_resname(
                    topology=self.modeller.topology, resname=self.resname
                ),
            },
        ).sampler
        if not restart:
            logging.info("Minimizing system...")
            t1 = time.time()
            sampler.minimize()

            logging.info(f"Minimised system  in {time.time() - t1} seconds")

        sampler.run()

    def run_neq_switching(self, steps: int, interval: int) -> float:
        """Compute the protocol work performed by switching from the MM description to the MM/ML through lambda_interpolate

        :param int steps: number of steps in non-equilibrium switching simulation
        :param int interval: reporterInterval
        :return float: protocol work from the integrator
        """
        alchemical_functions = {"lambda_interpolate": "lambda"}
        integrator = AlchemicalNonequilibriumLangevinIntegrator(
            alchemical_functions=alchemical_functions,
            nsteps_neq=steps,
            temperature=self.temperature,
            collision_rate=self.friction_coeff,
            timestep=self.timestep,
        )

        simulation = Simulation(
            self.modeller.topology,
            self.mixed_system,
            integrator,
            platformProperties={"Precision": "Mixed"},
        )
        simulation.context.setPositions(self.modeller.getPositions())

        logging.info("Minimising energy")
        simulation.minimizeEnergy()

        reporter = StateDataReporter(
            file=sys.stdout,
            reportInterval=interval,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
            totalSteps=steps,
            remainingTime=True,
        )
        simulation.reporters.append(reporter)
        # Append the snapshots to the pdb file
        simulation.reporters.append(
            PDBReporter("output_frames.pdb", steps / 80, enforcePeriodicBox=True)
        )
        # We need to take the final state
        simulation.step(steps)
        protocol_work = (integrator.get_protocol_work(dimensionless=True),)
        return protocol_work
