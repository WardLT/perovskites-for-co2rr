"""Relax the surfaces in parallel on HPC"""
from argparse import ArgumentParser
from pathlib import Path
import logging
import json
import sys

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read
from ase.optimize import BFGS

from co2rr.cp2k import make_calculator

if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser()
    parser.add_argument('initial', nargs='+', help='Directory holding initial structures to relax', type=Path)
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

    # Loop over the surfaces to relax
    #  TODO (wardlt): Use Parsl to run loop on HPC
    for surface in unrelaxed_surfaces:
        logger.info(f'Starting to relax {surface.parent}')

        # Load the file and write as an extended XYZ file
        traj_path = surface.parent / 'relax.traj'
        if traj_path.exists():
            atoms = read(traj_path, -1)
            logger.info(f'Read the last structure from {traj_path}')
        else:
            atoms = read(surface)
            atoms.pbc = [True, True, False]  # No periodicity in the Z direction
            logger.info('Starting a new run')

        # Redo the fixes
        indices = np.argsort(atoms.positions[:, 2])[:len(atoms) // 2]
        atoms.set_constraint(FixAtoms(indices=indices))

        # Load the charge
        charge = json.loads((surface.parent / 'info.json').read_text())['charge']

        # Run the optimization
        with make_calculator(atoms, cutoff=600, max_scf=500, charge=0) as calc:
            # Delete the old run
            for f in ['cp2k.out']:
                Path(f'run/{f}').write_text("")  # Clear it

            # Set up the calculator
            calc.directory = 'run'
            atoms.calc = calc

            # Run the relaxation
            opt = BFGS(atoms,
                       logfile=str(surface.parent / 'relax.log'),
                       trajectory=str(traj_path))
            opt.run(fmax=0.1)

        # Save the relaxed structure
        out_path = surface.parent.joinpath('relaxed.extxyz')
        atoms.write(out_path, columns=['symbols', 'positions', 'move_mask'])
        logger.info(f'Done with {surface.parent}. Saved result to {out_path}')
