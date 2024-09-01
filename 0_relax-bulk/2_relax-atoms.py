"""Relax the atomic degrees of freedom"""
from argparse import ArgumentParser
from pathlib import Path
import logging
import sys

from ase.db import connect
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from ase.io import read

from co2rr.cp2k import make_calculator


_problem_names = [
    # Structures which transform on relaxation
    'EuOsO3', 'TmCoO3'
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

    # Loop over structures in the 'volume-relax.db' database
    for path in ['initial-dilute-mixing.db', 'volume-relax.db']:
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
    
                # Start with last step in the trajectory, if available
                traj_dir = Path('atoms-relax') / name / f'{args.supercell_size}-cells'
                logger.info(f'Running {name} from {path} in {traj_dir}')
                traj_dir.mkdir(parents=True, exist_ok=True)
                if (traj_dir / 'relax.traj').exists():
                    atoms = read(traj_dir / 'relax.traj', -1)
                    logger.info('Read last step from ongoing trajectory')
                else:
                    atoms = row.toatoms()
                    logger.info('Starting from cubic structure')
    
                # Run relaxation
                with make_calculator(atoms, cutoff=600, max_scf=500) as calc:
                    # Delete the old run
                    for f in ['cp2k.out']:
                        Path(f'run/{f}').write_text("")  # Clear it
    
                    # Start by computing the stresses acting on the cell
                    calc.directory = 'run'
                    atoms.calc = calc
                    stresses = atoms.get_stress()
                    logger.info(f'{name} - Initial volume: {atoms.get_volume() / len(atoms):.2f}. Stress: {stresses[:3].sum() / 3:.2f}')
    
                    # Run the optimization
                    opt_atoms = FrechetCellFilter(atoms)  # Allow ASE to optimize the lattice parameter
                    opt = BFGS(opt_atoms,
                               trajectory=str(traj_dir / 'relax.traj'),
                               logfile=str(traj_dir / 'relax.log'))
                    opt.run(fmax=0.1, steps=args.max_steps)
                    stresses = atoms.get_stress()
                    logger.info(f'{name} - Final volume: {atoms.get_volume() / len(atoms):.2f}. Stress: {stresses[:3].sum() / 3:.2f}')
    
    
                # Write to the output
                with connect('atoms-relax.db') as db_out:
                    new_keys = row.key_value_pairs.copy()
                    new_keys['name'] = name
                    db_out.write(atoms, **new_keys)
