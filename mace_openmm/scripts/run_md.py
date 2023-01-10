import os
import mpiplus
from argparse import ArgumentParser
from mace_openmm.mixed_system import MixedSystem
from mace import tools
import logging
import torch


def main():
    parser = ArgumentParser()

    parser.add_argument("--file", "-f", type=str)
    parser.add_argument(
        "--ml_mol",
        type=str,
        help="either smiles string or file path for the small molecule to be described by MACE",
        default=None,
    )
    parser.add_argument(
        "--run_type", choices=["md", "repex", "neq"], type=str, default="md"
    )
    parser.add_argument("--steps", "-s", type=int, default=10000)
    parser.add_argument("--padding", "-p", default=1.2, type=float)
    parser.add_argument("--nonbondedCutoff", "-c", default=1.0, type=float)
    parser.add_argument("--ionicStrength", "-i", default=0.15, type=float)
    parser.add_argument("--potential", default="mace", type=str)
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--pressure", type=float, default=None)
    parser.add_argument("--replicas", type=int, default=10)
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default="output.pdb",
        help="output file for the pdb reporter",
    )
    parser.add_argument("--log_level", default=logging.INFO, type=int)
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument(
        "--output_dir",
        help="directory where all output will be written",
        default="./junk",
    )
    parser.add_argument(
        "--neighbour_list", default="torch_nl", choices=["torch_nl", "torch_nl_n2"]
    )

    # optionally specify box vectors for periodic systems
    parser.add_argument("--box", type=float, nargs="+", action="append")

    parser.add_argument("--log_dir", default="./logs")

    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--forcefields",
        type=list,
        default=[
            "amber/protein.ff14SB.xml",
            "amber/tip3p_standard.xml",
            "amber14/DNA.OL15.xml",
        ],
    )
    parser.add_argument(
        "--smff",
        help="which version of the openff small molecule forcefield to use",
        default="1.0",
        type=str,
        choices=["1.0", "2.0"],
    )
    parser.add_argument(
        "--interval", help="steps between saved frames", type=int, default=100
    )
    parser.add_argument(
        "--resname",
        "-r",
        help="name of the ligand residue in pdb file",
        default="UNK",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        "-m",
        help="path to the mace model",
        default="tests/test_openmm/MACE_SPICE_larger.model",
    )
    parser.add_argument(
        "--system_type",
        type=str,
        choices=["pure", "hybrid", "decoupled"],
        default="pure",
    )
    args = parser.parse_args()

    if args.dtype == "float32":
        logging.warning(
            "Running with single precision - this can lead to numerical stability issues"
        )
        torch.set_default_dtype(torch.float32)
        dtype = torch.float32
    elif args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
        dtype = torch.float64
    tools.setup_logger(level=args.log_level, directory=args.log_dir)

    # we don't need to specify the file twice if dealing with just the ligand
    if args.file.endswith(".sdf") and args.ml_mol is None:
        args.ml_mol = args.file

    # TODO: refactor this
    # user has specified a directory containing sdf files, parallelise over MPI ranks
    if os.path.isdir(args.file) and all(
        [f.endswith(".sdf") for f in os.listdir(args.file)]
    ):
        if args.run_type != "md":
            raise ValueError(
                f"When multiple moleucles are specified, only MD can be specified as the running mode, not {args.run_type}"
            )

        def _initialize_mixed_system(sdf_file):
            with open(sdf_file, "r") as f:
                # crudely extract resname as first line of sdf file
                lines = f.readlines()
                resname = lines[0].strip()
                print(f"Got resname {resname} ")

            return MixedSystem(
                file=sdf_file,
                ml_mol=sdf_file,
                model_path=args.model_path,
                forcefields=args.forcefields,
                resname=resname,
                ionicStrength=args.ionicStrength,
                nonbondedCutoff=args.nonbondedCutoff,
                potential=args.potential,
                padding=args.padding,
                temperature=args.temperature,
                dtype=dtype,
                output_dir=os.path.join(args.output_dir, resname),
                neighbour_list=args.neighbour_list,
                system_type=args.system_type,
                smff=args.smff,
                pressure=args.pressure,
                boxvecs=args.box,
            )

        ml_mols = [os.path.join(args.file, f) for f in os.listdir(args.file)]

        # TODO: Why does system setup with MPI cause the downstream MPI processes to misbehave?

        # mixed_systems, _ = mpiplus.distribute(_initialize_mixed_system, ml_mols, send_results_to=0, sync_nodes=True)
        mixed_systems = [_initialize_mixed_system(sdf_file) for sdf_file in ml_mols]
        print("mixed systems: ", mixed_systems)

        def _run_mixed_md(system_idx: int):
            return mixed_systems[system_idx].run_mixed_md(
                args.steps, args.interval, args.output_file
            )

        # now distribute execution of the MD jobs between the MPI ranks

        print("Running MD on parallel MPI ranks")
        mpiplus.distribute(_run_mixed_md, range(len(mixed_systems)))
        # print(results)

    else:

        mixed_system = MixedSystem(
            file=args.file,
            ml_mol=args.ml_mol,
            model_path=args.model_path,
            forcefields=args.forcefields,
            resname=args.resname,
            ionicStrength=args.ionicStrength,
            nonbondedCutoff=args.nonbondedCutoff,
            potential=args.potential,
            padding=args.padding,
            temperature=args.temperature,
            dtype=dtype,
            output_dir=args.output_dir,
            neighbour_list=args.neighbour_list,
            system_type=args.system_type,
            smff=args.smff,
            pressure=args.pressure,
            boxvecs=args.box,
        )
        if args.run_type == "md":
            mixed_system.run_mixed_md(args.steps, args.interval, args.output_file)
        elif args.run_type == "repex":
            mixed_system.run_replex_equilibrium_fep(
                args.replicas, args.restart, args.steps
            )
        elif args.run_type == "neq":
            mixed_system.run_neq_switching(args.steps, args.interval)
        else:
            raise ValueError(f"run_type {args.run_type} was not recognised")


if __name__ == "__main__":
    main()
