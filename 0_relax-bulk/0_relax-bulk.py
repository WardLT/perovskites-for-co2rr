"""Relax the surfaces in parallel on HPC"""
from argparse import ArgumentParser
from pathlib import Path
import logging
import sys

import numpy as np
from ase.constraints import FixAtoms
from ase.db import connect
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS

from co2rr.cp2k import make_calculator
from co2rr.emery import load_emery_dataset, generate_initial_structure

if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser(description='Relax the volume of the supercell, holding atomic positions and cubic symmetry constant')
    parser.add_argument('--supercell-size', default=1, type=int, help='Number of repeats of the supercell')
    args = parser.parse_args()

    # Make the logger
    logger = logging.getLogger('main')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Find the structures to relax
    emery = load_emery_dataset()
    for _, row in emery.iterrows():
        # Check if it's already done
        name = row['Chemical formula']
        with connect('cp2k-relax.db') as db:
            if db.count(name=name, supercell=args.supercell_size) > 0:
                continue

        # Run, if needed
        atoms = generate_initial_structure(row)
        atoms *= [args.supercell_size] * 3
        with make_calculator(atoms, cutoff=600, max_scf=500) as calc:
            # Delete the old run
            for f in ['cp2k.out', 'relax.log']:
                Path(f'run/{f}').write_text("")  # Clear it

            # Start by computing the stresses acting on the cell
            calc.directory = 'run'
            atoms.calc = calc
            stresses = atoms.get_stress()
            logger.info(f'{name} - Initial volume: {atoms.get_volume() / len(atoms):.2f}. Stress: {stresses[:3].sum() / 3:.2f}')

            # Run the optimization
            atoms.set_constraint(FixAtoms(mask=[True] * len(atoms)))
            opt_atoms = FrechetCellFilter(atoms, hydrostatic_strain=True)  # Allow ASE to optimize the lattice parameter
            opt = BFGS(opt_atoms, trajectory='run/relax.traj', logfile='run/relax.log')
            opt.run(fmax=0.1)
            stresses = atoms.get_stress()
            logger.info(f'{name} - Final volume: {atoms.get_volume() / len(atoms):.2f}. Stress: {stresses[:3].sum() / 3:.2f}')

            atoms.set_constraint()

        # Write to the output
        with connect('cp2k-relax.db') as db:
            try:
                a_val, b_val = int(row['Valence A']), int(row['Valence B'])
            except ValueError:
                a_val = b_val = -1
            db.write(atoms, name=name, a=row['A'], b=row['B'], a_val=a_val, b_val=b_val, supercell=args.supercell_size)
