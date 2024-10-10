"""Relax the atomic degrees of freedom"""
from argparse import ArgumentParser
from tarfile import TarFile
from shutil import copyfileobj
from pathlib import Path
import logging
import gzip
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
    parser.add_argument('--xc-name', default="pbe-plus-u", help='Which XC functional to use')
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
            pdos_path = traj_dir / f'pdos.{args.xc_name}.tar.gz'
            if pdos_path.exists():
                continue

            # Find the WFN file for a restart
            wfn_path = traj_dir / 'pbe-plus-u.wfn'
            if not wfn_path.exists():
                wfn_path = None  # Skip if not available

            # Run pdos calculation
            atoms = row.toatoms()
            run_dir = Path('run')
            with make_calculator(atoms, xc_name=args.xc_name,
                                 cutoff=600 if args.xc_name == 'pbe-plus-u' else 400,
                                 max_scf=500,
                                 compute_pdos=True,
                                 wfn_guess=wfn_path) as calc:
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
            logger.info(f'Wrote {pdos_count} PDOS files to {pdos_path}')

            # Copy the charge information
            chg_file = next(run_dir.glob('density-ELECTRON_DENSITY-*.cube'))
            with gzip.open(traj_dir / f'{args.xc_name}.cube.gz', 'wb') as fo:
                with chg_file.open('rb') as fi:
                    copyfileobj(fi, fo)
            chg_file.unlink()
            logger.info(f'Moved the cube file from {chg_file}')

            # Copy the mulliken charges
            mlk_file = run_dir / 'mulliken.charges'
            mlk_file.rename(traj_dir / f'{args.xc_name}.mulliken.charges')

            # Store the wfn file
            (run_dir / 'cp2k-RESTART.wfn').rename(traj_dir / f'{args.xc_name}.wfn')
            logger.info('Stored the wfn file as a backup')
