"""Relax the surfaces in parallel on HPC"""
from argparse import ArgumentParser
from asyncio import as_completed
from pathlib import Path
import logging
import sys

import parsl
from parsl import python_app
from parsl import Config, HighThroughputExecutor
from parsl.providers import PBSProProvider


@python_app
def relax_surface(surface: Path, max_steps: int) -> Path:
    """Execute surface relaxation

    Args:
        surface: Path to the surface structure
        max_steps: Maximum number of outputs steps
    Returns:
        ``surface``
    """
    from ase.constraints import FixAtoms
    from ase.optimize import FIRE
    from ase.io import read
    from co2rr.cp2k import make_calculator
    from uuid import uuid4
    import numpy as np

    traj_path = surface.parent / 'relax.traj'
    start = 0
    if traj_path.exists():
        traj = read(traj_path, ':')
        start = len(traj)
        atoms = traj[-1]
    else:
        atoms = read(surface)
        atoms.pbc = [True, True, False]  # No periodicity in the Z direction

    # Skip if we've already hit the budget
    if start > max_steps:
        return surface

    # Fix the atoms in the middle third center of the structure
    top_z = np.max(atoms.positions[:, 2])
    bot_z = np.min(atoms.positions[:, 2])
    depth = top_z - bot_z
    mask = np.logical_and(atoms.positions[:, 2] > bot_z + depth / 3,
                          atoms.positions[:, 2] < top_z - depth / 3)
    atoms.set_constraint(FixAtoms(mask=mask))

    # Run the optimization
    run_dir = Path('run') / str(uuid4()).split("-")[0]
    run_dir.mkdir()
    with make_calculator(atoms, cutoff=600, max_scf=500, uks=True) as calc:
        # Delete the old run
        for f in ['cp2k.out']:
            (run_dir / f).write_text("")  # Clear it

        # Set up the calculator
        calc.directory = str(run_dir)
        atoms.calc = calc

        # Run the relaxation
        opt = FIRE(atoms,
                   logfile=str(surface.parent / 'relax.log'),
                   trajectory=str(traj_path))
        opt.run(fmax=0.1, steps=max_steps - start)

    # Save the relaxed structure
    atoms.write(surface.parent.joinpath('relaxed.extxyz'),
                columns=['symbols', 'positions', 'move_mask'])
    return surface


if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser()
    parser.add_argument('initial', nargs='+', help='Directory holding initial structures to relax', type=Path)
    parser.add_argument('--max-steps', default=256, help='Maximum number of relaxation steps', type=int)
    args = parser.parse_args()

    # Make the logger
    logger = logging.getLogger('main')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Find the structures to relax
    all_surfaces = sum([list(Path(i).rglob('unrelaxed.extxyz')) for i in args.initial], [])
    logger.info(f'Found {len(all_surfaces)} surfaces')
    unrelaxed_surfaces = set(
        surface for surface in all_surfaces
        if not surface.parent.joinpath('relaxed.extxyz').is_file()
    )
    logger.info(f'Found {len(unrelaxed_surfaces)} we have not relaxed yet')

    # Configure Parsl
    config = Config(
        executors=[
            HighThroughputExecutor(
                max_workers_per_node=1,
                provider=PBSProProvider(
                    account='co2rr_vfp',
                    min_blocks=0,
                    max_blocks=4,
                    nodes_per_block=2,
                    walltime='72:00:00',
                    worker_init='''
# Load the conda environment
source activate /lcrc/project/co2rr_vfp/perovskites-for-co2rr/env
which python

# Load environment
module load gcc mpich
module list

nnodes=`cat $PBS_NODEFILE | sort | uniq | wc -l`
ranks_per_node=64
total_ranks=$(($ranks_per_node * $nnodes))
threads_per_rank=$((128 / ranks_per_node))

echo Running $total_ranks ranks across $nnodes nodes with $threads_per_rank threads per rank

export OMP_NUM_THREADS=$threads_per_rank
export ASE_CP2K_COMMAND="mpiexec -n $total_ranks -ppn $ranks_per_node --bind-to core:$threads_per_rank /lcrc/project/Athena/cp2k-mpich/exe/local/cp2k_shell.psmp"''')
            )
        ]
    )
    parsl.load(config)
    logger.info('Loaded parsl configuration')

    # Loop over the surfaces to relax
    futures = []
    for surface in unrelaxed_surfaces:
        logger.info(f'Starting to relax {surface.parent}')
        out_path = surface.parent.joinpath('relaxed.extxyz')
        if out_path.exists():
            continue

        futures.append(relax_surface(surface, args.max_steps))
    logger.info(f'Submitted {len(futures)} surface calculations')

    # Wait as they complete
    for i, future in enumerate(as_completed(futures)):
        if (exc := future.exception()) is not None:
            logger.warning(f'Surface relaxation failed. Exception: {exc}')
        else:
            logger.info(f'Completed {future.result()}. Remaining: {len(futures) - i - 1}')
