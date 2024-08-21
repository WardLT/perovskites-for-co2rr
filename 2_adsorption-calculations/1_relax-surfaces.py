"""Relax the surfaces in parallel on HPC"""
from argparse import ArgumentParser
from pathlib import Path
import logging
import json
import sys
from uuid import uuid4

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read
from ase.optimize import FIRE

from co2rr.cp2k import make_calculator

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

    # Loop over the surfaces to relax
    #  TODO (wardlt): Use Parsl to run loop on HPC
    for surface in unrelaxed_surfaces:
        logger.info(f'Starting to relax {surface.parent}')

        # Load the file and write as an extended XYZ file
        traj_path = surface.parent / 'relax.traj'
        start = 0
        if traj_path.exists():
            traj = read(traj_path, ':')
            start = len(traj)
            atoms = traj[-1]
            logger.info(f'Read the last structure out out {start} from {traj_path}')
        else:
            atoms = read(surface)
            atoms.pbc = [True, True, False]  # No periodicity in the Z direction
            logger.info('Starting a new run')

        # Skip if we've already hit the budget
        if start > args.max_steps:
            logger.info(f'Already hit the step budget of {args.max_steps}')
            continue

        # Fix all except the CO2 atoms
        atoms.set_constraint(FixAtoms(indices=range(len(atoms) - 2)))

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
            opt.run(fmax=0.1, steps=args.max_steps - start)

        # Save the relaxed structure
        out_path = surface.parent.joinpath('relaxed.extxyz')
        atoms.write(out_path, columns=['symbols', 'positions', 'move_mask'])
        logger.info(f'Done with {surface.parent}. Saved result to {out_path}')
