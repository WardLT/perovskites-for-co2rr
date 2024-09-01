"""Relax the atomic degrees of freedom"""
from argparse import ArgumentParser
from tarfile import TarFile
from pathlib import Path
import logging
import sys

from ase.db import connect

from co2rr.cp2k import make_calculator


_problem_names = [
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

    # Loop over structures in the 'atoms-relax.db' database
    with connect('atoms-relax.db') as db_in:
        for row in db_in.select(supercell=args.supercell_size):
            # Skip problematic structures
            name = row.key_value_pairs['name']
            if name in _problem_names:
                continue

            # Check if it's already done
            traj_dir = Path('atoms-relax') / name / f'{args.supercell_size}-cells'
            pdos_path = traj_dir / 'pdos.pbe-plus-u.tar.gz'
            if pdos_path.exists():
                continue

            # Run pdos calculation
            atoms = row.toatoms()
            run_dir = Path('run')
            with make_calculator(atoms, cutoff=600, max_scf=500, compute_pdos=True) as calc:
                # Delete the old run
                for f in ['cp2k.out']:
                    Path(run_dir / f).write_text("")  # Clear it
                for f in run_dir.glob('*.pdos'):
                    f.unlink()
                calc.directory = run_dir

                # Start by computing the stresses acting on the cell
                logger.info(f'Computing the PDOS for {name}')
                atoms.calc = calc
                atoms.get_potential_energy()

            # Move the pdos output file to the data director
            pdos_count = 0
            with TarFile.gzopen(pdos_path, 'w') as tar:
                for f in Path('run').glob("*.pdos"):
                    pdos_count += 1
                    tar.add(f, arcname=f.name)
                    f.unlink()
            if pdos_count == 0:
                pdos_path.unlink()
                raise ValueError('No PDOS files were written (or found)')
            logger.info(f'Wrote {pdos_path} PDOS files to {pdos_path}')

            # Store the wfn file
            (run_dir / 'cp2k-RESTART.wfn').rename(traj_dir / 'pbs-plus-u.wfn')
            logger.info('Stored the wfn file as a backup')
