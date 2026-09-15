import os
import argparse
from datetime import datetime
from glob import glob
from tqdm.contrib.concurrent import process_map
import torch
import numpy as np
import pdbfixer
import openmm
from openmm import unit
from openmm.app import PDBFile, Modeller, ForceField, Simulation
from openmm import LangevinIntegrator
from openmm import Platform

import nbfold
from nbfold.models.AbResNet import load_model
from nbfold.models.ModelEnsemble import ModelEnsemble
from nbfold.build_fv.build_cen_fa import build_initial_fv
from nbfold.util.pdb import renumber_pdb
import pyrosetta

pyrosetta.init("-mute all")


def prog_print(text):
    print("*" * 50)
    print(text)
    print("*" * 50)


def refine_fv_openmm(in_pdb_file, out_pdb_file, stiffness=10.0, tolerance=2.39, use_gpu=True, gpu_id=3):
    ENERGY = unit.kilocalories_per_mole
    LENGTH = unit.angstroms

    tolerance = tolerance * ENERGY
    stiffness = stiffness * ENERGY / (LENGTH ** 2)

    fixer = pdbfixer.PDBFixer(in_pdb_file)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    forcefield = ForceField("amber14/protein.ff14SB.xml")
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(forcefield)

    system = forcefield.createSystem(modeller.topology)

    force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", stiffness)
    for p in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(p)
    for atom in modeller.topology.atoms():
        if atom.name in ["N", "CA", "C", "CB"]:
            force.addParticle(atom.index, modeller.positions[atom.index])
    system.addForce(force)

    integrator = LangevinIntegrator(0, 0.01, 1.0)

    platform_name = "CUDA" if use_gpu else "CPU"
    platform = Platform.getPlatformByName(platform_name)
    if use_gpu:
        platform.setPropertyDefaultValue("DeviceIndex", str(gpu_id))

    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(tolerance)

    with open(out_pdb_file, "w") as f:
        PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(getPositions=True).getPositions(),
            f,
            keepIds=True,
        )


def refine_fv_(args):
    in_pdb_file, out_pdb_file, use_gpu = args
    refine_fv_openmm(in_pdb_file, out_pdb_file, use_gpu=use_gpu)
    return 0  # Dummy score


def build_structure(model,
                    fasta_file,
                    out_dir,
                    target="pred",
                    num_decoys=5,
                    num_procs=1,
                    single_chain=False,
                    device=None,
                    use_gpu=False):
    decoy_dir = os.path.join(out_dir, "decoys")
    os.makedirs(decoy_dir, exist_ok=True)

    prog_print("Creating MDS structure")
    mds_pdb_file = os.path.join(decoy_dir, f"{target}.mds.pdb")
    build_initial_fv(fasta_file,
                     mds_pdb_file,
                     model,
                     single_chain=single_chain,
                     device=device)

    filename = os.path.splitext(os.path.basename(fasta_file))[0]
    prog_print("Creating decoy structures")

    decoy_pdb_pattern = os.path.join(decoy_dir, f"{target}.nbfold.{{}}.pdb")
    refine_args = [(mds_pdb_file, decoy_pdb_pattern.format(i), use_gpu) for i in range(num_decoys)]

    # If using GPU, avoid multiprocessing due to CUDA init issues
    if use_gpu:
        print("Running OpenMM refinement sequentially on GPU to avoid CUDA initialization errors.")
        for args in refine_args:
            refine_fv_(args)
    else:
        process_map(refine_fv_, refine_args, max_workers=num_procs)

    # Pick best decoy (dummy pick first since we return 0 scores)
    best_decoy_pdb = decoy_pdb_pattern.format(0)
    out_pdb = os.path.join("nano", f"{filename}.pdb")
    os.system(f"cp {best_decoy_pdb} {out_pdb}")

    return out_pdb


def _get_args():
    project_path = os.path.abspath(os.path.join(nbfold.__file__, "../.."))
    now = str(datetime.now().strftime('%y-%m-%d_%H:%M:%S'))

    parser = argparse.ArgumentParser(description="Antibody Fv structure prediction using OpenMM refinement")
    parser.add_argument("--pred_dir",
                        type=str,
                        default=os.path.join(project_path, f"pred_{now}"),
                        help="Directory to save results.")
    parser.add_argument("--model_dir",
                        type=str,
                        default="trained_models/ensemble",
                        help="Directory with trained .pt model files.")
    parser.add_argument("--target", type=str, default="pred", help="Target name for prediction.")
    parser.add_argument("--decoys", type=int, default=5, help="Number of decoy structures to generate.")
    parser.add_argument("--num_procs", type=int, default=5, help="Parallel processes for refinement on CPU.")
    parser.add_argument("--renumber", default=False, action="store_true", help="Renumber final PDB with AbNum.")
    parser.add_argument("--single_chain", default=False, action="store_true", help="Use single chain input.")
    parser.add_argument("--native_pdb", type=str, default=None, help="Optional native PDB for RMSD comparison.")
    parser.add_argument("--use_gpu", default=False, action="store_true", help="Use GPU for OpenMM refinement and model prediction.")

    return parser.parse_args()


def _cli():
    args = _get_args()

    pred_dir = args.pred_dir
    model_dir = args.model_dir
    target = args.target
    decoys = args.decoys
    num_procs = args.num_procs
    renumber = args.renumber
    single_chain = args.single_chain
    native_pdb = args.native_pdb
    use_gpu = args.use_gpu

    # Select PyTorch device with GPU #3 if available and requested
    device_type = 'cuda:6' if torch.cuda.is_available() and use_gpu else 'cpu'
    device = torch.device(device_type)

    model_files = glob(os.path.join(model_dir, "*.pt"))
    if len(model_files) == 0:
        exit(f"No model files found at: {model_dir}")

    model = ModelEnsemble(model_files=model_files,
                          load_model=load_model,
                          eval_mode=True,
                          device=device)

    os.makedirs("nano", exist_ok=True)

    fasta_files = glob("data/nano/*.fasta")
    print("Found FASTA files:", fasta_files)

    for fasta_file in fasta_files:
        prog_print(f"Processing {fasta_file}")
        pred_pdb = build_structure(model,
                                   fasta_file,
                                   pred_dir,
                                   target=target,
                                   num_decoys=decoys,
                                   num_procs=num_procs,
                                   single_chain=single_chain,
                                   device=device,
                                   use_gpu=use_gpu)

        if renumber:
            renumber_pdb(pred_pdb, pred_pdb)

    if native_pdb is not None and os.path.exists(native_pdb):
        from nbfold.metrics.rosetta_ab import get_ab_metrics

        pose = pyrosetta.pose_from_pdb(pred_pdb)
        native_pose = pyrosetta.pose_from_pdb(native_pdb)

        metrics = get_ab_metrics(pose, native_pose)
        print("Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name.ljust(12)}{round(metric_value, 3)}")


if __name__ == '__main__':
    _cli()
