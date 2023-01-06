import sys
from ase.io import read
import torch
import time
import numpy as np
import sys
from ase import Atoms
from openmm.openmm import System
from typing import List, Tuple, Optional
from openmm import LangevinMiddleIntegrator, Vec3, MonteCarloBarostat
from openmmtools.integrators import AlchemicalNonequilibriumLangevinIntegrator
from openmm.app import (
    Simulation,
    StateDataReporter,
    PDBReporter,
    ForceField,
    PDBFile,
    HBonds,
    AllBonds,
    Modeller,

    PME,
)
from openmm.unit import nanometer, nanometers, molar, angstrom
from openmm.unit import (
    kelvin,
    picosecond,
    femtosecond,
    kilojoule_per_mole,
    picoseconds,
    femtoseconds,
    bar
)
from openff.toolkit.topology import Molecule

from openmmforcefields.generators import SMIRNOFFTemplateGenerator

# from mace.calculators.openmm import MacePotentialImplFactory
from openmmml.models.mace_potential import MacePotentialImplFactory
from openmmml.models.anipotential import ANIPotentialImplFactory
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
MLPotential.registerImplFactory("ani2x", ANIPotentialImplFactory())


# platform = Platform.getPlatformByName("CUDA")
# platform.setPropertyDefaultValue("DeterministicForces", "true")

logger = logging.getLogger("INFO")


class MixedSystem:
    forcefields: List[str]
    padding: float
    ionicStrength: float
    nonbondedCutoff: float
    resname: str
    potential: str
    temperature: float
    friction_coeff: float
    timestep: float
    dtype: torch.dtype
    output_dir: str
    neighbour_list: str
    openmm_precision: str
    SM_FF: str
    modeller: Modeller
    mixed_system: System
    decoupled_system: Optional[System]

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
        dtype: torch.dtype,
        neighbour_list: str,
        output_dir: str,
        system_type: str,
        friction_coeff: float = 1.0,
        timestep: float = 1.0,
        smff: str = "1.0",
        pressure: Optional[float] = None
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
        self.dtype = dtype
        self.output_dir = output_dir
        self.neighbour_list = neighbour_list
        self.openmm_precision = "Double" if dtype == torch.float64 else "Mixed"
        logger.debug(f"OpenMM will use {self.openmm_precision} precision")

        if smff == "1.0":
            self.SM_FF = "openff_unconstrained-1.0.0.offxml"
            logger.info("Using openff-1.0 unconstrained forcefield")
        elif smff == "2.0":
            self.SM_FF = "openff_unconstrained-2.0.0.offxml"
            logger.info("Using openff-2.0 unconstrained forcefield")
        else:
            raise ValueError(f"Small molecule forcefield {smff} not recognised")

        os.makedirs(self.output_dir, exist_ok=True)

        self.create_mixed_system(
            file=file, ml_mol=ml_mol, model_path=model_path, system_type=system_type, pressure=pressure
        )

    def initialize_mm_forcefield(
        self, molecule: Optional[Molecule] = None
    ) -> ForceField:

        forcefield = ForceField(*self.forcefields)
        if molecule is not None:
            # Ensure we use unconstrained force field
            smirnoff = SMIRNOFFTemplateGenerator(
                molecules=molecule, forcefield=self.SM_FF
            )
            forcefield.registerTemplateGenerator(smirnoff.generator)
        return forcefield

    def initialize_ase_atoms(self, ml_mol: str) -> Tuple[Atoms, Molecule]:
        """Generate the ase atoms object from the   

        :param str ml_mol: file path or smiles  
        :return Tuple[Atoms, Molecule]: ase Atoms object and initialised openFF molecule
        """
        # ml_mol can be a path to a file, or a smiles string
        if os.path.isfile(ml_mol):
            molecule = Molecule.from_file(ml_mol, allow_undefined_stereo=True)
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
        system_type: str,
        pressure: Optional[float]
    ) -> None:
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
            topology = input_file.getTopology()

            # if pure_ml_system specified, we just need to parse the input file
            # if not pure_ml_system:
            self.modeller = Modeller(input_file.topology, input_file.positions)
            print(f"Initialized topology with {len(input_file.positions)} positions")

        # Handle a ligand, passed as an sdf, override the Molecule initialized from smiles
        elif file.endswith(".sdf"):
            molecule = Molecule.from_file(file)
            input_file = molecule
            topology = molecule.to_topology().to_openmm()
            # Hold positions in nanometers
            positions = get_xyz_from_mol(molecule.to_rdkit()) / 10

            print(f"Initialized topology with {positions.shape} positions")

            self.modeller = Modeller(topology, positions)

        if system_type == "pure":
            # we have the input_file, create the system directly from the mace potential
            # TODO: add a function to compute periodic box vectors to enforce a minimum padding distance to each box wall
            atoms.set_cell([50, 50, 50])
            topology.setPeriodicBoxVectors([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]])
            ml_potential = MLPotential("mace")
            self.mixed_system = ml_potential.createSystem(
                topology, atoms_obj=atoms, filename=model_path, dtype=self.dtype
            )

        # Handle the mixed systems with a classical forcefield
        elif system_type in ["hybrid", "decoupled"]:
            forcefield = self.initialize_mm_forcefield(molecule=molecule)
            self.modeller.addSolvent(
                forcefield,
                padding=self.padding * nanometers,
                ionicStrength=self.ionicStrength * molar,
                neutralize=True,
            )

            omm_box_vecs = self.modeller.topology.getPeriodicBoxVectors()
            # print(omm_box_vecs)
            atoms.set_cell(
                [
                    omm_box_vecs[0][0].value_in_unit(angstrom),
                    omm_box_vecs[1][1].value_in_unit(angstrom),
                    omm_box_vecs[2][2].value_in_unit(angstrom),
                ]
            )

            system = forcefield.createSystem(
                self.modeller.topology,
                nonbondedMethod=PME,
                nonbondedCutoff=self.nonbondedCutoff * nanometer,
                constraints=None,
            )
            if pressure is not None:
                logger.info(f"Pressure will be maintained at {pressure} bar with MC barostat")
                system.addForce(MonteCarloBarostat(pressure*bar, self.temperature*kelvin))


            # write the final prepared system to disk
            with open(os.path.join(self.output_dir, "prepared_system.pdb"), "w") as f:
                PDBFile.writeFile(
                    self.modeller.topology, self.modeller.getPositions(), file=f
                )
            if system_type == "hybrid":
                logger.debug("Creating hybrid system")
                self.mixed_system = MixedSystemConstructor(
                    system=system,
                    topology=self.modeller.topology,
                    nnpify_resname=self.resname,
                    nnp_potential=self.potential,
                    atoms_obj=atoms,
                    filename=model_path,
                    dtype=self.dtype,
                    nl=self.neighbour_list,
                ).mixed_system

            # optionally, add the alchemical customCVForce for the nonbonded interactions to run ABFE edges
            else:
                logger.debug("Creating decoupled system")
                self.mixed_system = MixedSystemConstructor(
                    system=system,
                    topology=self.modeller.topology,
                    nnpify_resname=self.resname,
                    nnp_potential=self.potential,
                    atoms_obj=atoms,
                    filename=model_path,
                    dtype=self.dtype,
                    nl=self.neighbour_list,
                ).decoupled_system
            
        else:
            raise ValueError(f"system type {system_type} not recognised - aborting!")
    # def run_abfe

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
        logging.info("Minimising energy...")
        simulation.minimizeEnergy()
        minimised_state = simulation.context.getState(
            getPositions=True, getVelocities=True, getForces=True
        )
        with open(os.path.join(self.output_dir, f"minimised_system.pdb"), "w") as f:
            PDBFile.writeFile(
                self.modeller.topology, minimised_state.getPositions(), file=f
            )
            # if something goes wrong minimising the hybrid system, fall back to minimising with the full MM hamiltonian

        # for step in range(1000):
        #     print(f"minimisation step {step}")
        #     simulation.minimizeEnergy(maxIterations=1)
        #     minimised_state = simulation.context.getState(getPositions=True, getVelocities=True, getForces=True)

        #         # write the minimised structure
        #     with open(os.path.join(self.output_dir, f"minimised_system_{step}.pdb"), 'w') as f:
        #         PDBFile.writeFile(self.modeller.topology, minimised_state.getPositions(), file=f)

        # dump velocities and forces to xml
        # with open(os.path.join(self.output_dir, "minimised_system.xml"), 'w') as f:
        #     f.write(XmlSerializer.serialize(minimised_state))

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
                file=os.path.join(self.output_dir, output_file),
                reportInterval=interval,
                enforcePeriodicBox=False,
            )
        )

        simulation.step(steps)

    def run_replex_equilibrium_fep(
        self, replicas: int, restart: bool, steps: int
    ) -> None:

        repex_file_exists = os.path.isfile(os.path.join(self.output_dir, "repex.nc"))
        # even if restart has been set, disable if the checkpoint file was not found, enforce minimising the system
        if not repex_file_exists:
            restart = False
        sampler = RepexConstructor(
            mixed_system=self.mixed_system,
            initial_positions=self.modeller.getPositions(),
            # repex_storage_file="./out_complex.nc",
            temperature=self.temperature * kelvin,
            n_states=replicas,
            restart=restart,
            mcmc_moves_kwargs={
                "timestep": 1.0 * femtoseconds,
                "collision_rate": 1.0 / picoseconds,
                "n_steps": 1000,
                "reassign_velocities": True,
            },
            replica_exchange_sampler_kwargs={
                "number_of_iterations": steps,
                "online_analysis_interval": 10,
                "online_analysis_minimum_iterations": 10,
            },
            storage_kwargs={
                "storage": os.path.join(self.output_dir, "repex.nc"),
                "checkpoint_interval": 100,
                "analysis_particle_indices": get_atoms_from_resname(
                    topology=self.modeller.topology, resname=self.resname
                ),
            },
        ).sampler
        # do not minimsie if we are hot-starting the simulation from a checkpoint
        if not restart:
            logging.info("Minimizing system...")
            t1 = time.time()
            sampler.minimize()

            logging.info(f"Minimised system  in {time.time() - t1} seconds")
            # we want to write out the positions after the minimisation - possibly something weird is going wrong here and it's ending up in a weird conformation

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
            platformProperties={"Precision": "Double"},
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
            PDBReporter(
                os.path.join(self.output_dir, "output_frames.pdb"),
                steps / 80,
                enforcePeriodicBox=True,
            )
        )
        # We need to take the final state
        simulation.step(steps)
        protocol_work = (integrator.get_protocol_work(dimensionless=True),)
        return protocol_work
