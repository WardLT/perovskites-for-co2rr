"""Relax the atomic degrees of freedom"""
from argparse import ArgumentParser
from asyncio import as_completed
from pathlib import Path
import logging
import sys
from uuid import uuid4

import ase
from ase.db import connect
import parsl
from parsl import python_app
from parsl import Config, HighThroughputExecutor
from parsl.providers import PBSProProvider
from parsl.launchers import SimpleLauncher


def run_optimization(atoms: ase.Atoms, run_dir: Path) -> ase.Atoms:
    """Perform an optimization in a directory"""
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS
    from ase.io import read
    from pathlib import Path
    import logging

    from co2rr.cp2k import make_calculator

    logger = logging.getLogger('opt')

    # Start with last step in the trajectory, if available
    logger.info(f'Running {name} from {path} in {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    cur_step = 0
    if (run_dir / 'relax.traj').exists():
        atoms = read(run_dir / 'relax.traj', -1)
        cur_step = len(read(run_dir / 'relax.traj', ':'))
        logger.info('Read last step from ongoing trajectory')
    else:
        atoms.rattle()  # Break symmetry
        logger.info('Starting from cubic structure')

    # Run relaxation
    cp2k_dir = Path('run') / str(uuid4()).split("-")[0]
    cp2k_dir.mkdir()
    with make_calculator(atoms, cutoff=600, max_scf=500) as calc:
        # Delete the old run
        for f in ['cp2k.out']:
            Path(f'run/{f}').write_text("")  # Clear it

        # Start by computing the stresses acting on the cell
        calc.directory = cp2k_dir
        atoms.calc = calc
        stresses = atoms.get_stress()
        logger.info(f'{name} - Initial volume: {atoms.get_volume() / len(atoms):.2f}. Stress: {stresses[:3].sum() / 3:.2f}')

        # Run the optimization
        opt_atoms = FrechetCellFilter(atoms)  # Allow ASE to optimize the lattice parameter
        opt = BFGS(opt_atoms,
                   trajectory=str(run_dir / 'relax.traj'),
                   logfile=str(run_dir / 'relax.log'))
        opt.run(fmax=0.1, steps=args.max_steps - cur_step)
        stresses = atoms.get_stress()
        logger.info(f'{name} - Final volume: {atoms.get_volume() / len(atoms):.2f}. Stress: {stresses[:3].sum() / 3:.2f}')

        # Copy the wfn file
        (run_dir / 'cp2k-RESTART.wfn').rename(traj_dir / f'pbe-plus-u.wfn')

    return atoms


_problem_names = [
    # Structures which transform on relaxation
    'EuOsO3', 'TmCoO3',

    # Electronic structure fails to converge
    "Ce(Tm7Al1)O3", "Ce(Al7Tm1)O3",

    # Failing for unknown reasons
    'LaTlO3'
]

if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser(description='Relax the volume of the supercell, holding atomic positions and cubic symmetry constant')
    parser.add_argument('--supercell-size', default=2, type=int, help='Number of repeats of the supercell')
    parser.add_argument('--max-steps', default=128, type=int, help='Maximum number of optimization steps')
    args = parser.parse_args()

    # Make the logger
    logger = logging.getLogger('main')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Configure parsl
    config = Config(
        executors=[
            HighThroughputExecutor(
                max_workers_per_node=1,
                provider=PBSProProvider(
                    launcher=SimpleLauncher(),
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
cd $PBS_O_WORKDIR

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

    # Loop over structures in the 'volume-relax.db' database
    futures = []
    for path in ['volume-relax.db', 'initial-dilute-mixing.db']:
        with connect(path) as db_in:
            for row in db_in.select(supercell=args.supercell_size):
                # Skip problematic structures
                name = row.key_value_pairs['name']
                if name in _problem_names:
                    continue

                # Check if it's already done
                with connect('atoms-relax.db') as db_out:
                    if db_out.count(name=name, supercell=args.supercell_size) > 0:
                        continue

                traj_dir = Path('atoms-relax') / name / f'{args.supercell_size}-cells'
                future = run_optimization(row.toatoms(), traj_dir)
                future.kv = row.key_value_pairs.copy()
                future.kv['name'] = name

    for future in as_completed(futures):
        if future.exception():
            logger.error(f'Failed due to: {future.exception()}')
        else:
            # Write to the output
            atoms = future.result()
            with connect('atoms-relax.db') as db_out:
                db_out.write(atoms, **future.kv)

    parsl.dfk().cleanup()
