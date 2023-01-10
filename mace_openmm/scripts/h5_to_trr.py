# Postprocessing scripts to handle hdf5 files written out from MD for further analysis
# 
import mdtraj
import MDAnalysis
from argparse  import ArgumentParser

def hdf5_to_trr(filename: str, output_file: str):
    traj = mdtraj.load_hdf5(filename)
    print(traj)
    traj.save_trr(output_file)


if __name__ == "__main__":
    parser = ArgumentParser
    parser.add_argument("-f", "--filename", help="input hdf5 file to analyse")
    parser.add_argument("-o", "--output", help="file to write converted trajectory to")
    args = parser.parse_args()
    hdf5_to_trr(filename=args.f, output_file=args.o)