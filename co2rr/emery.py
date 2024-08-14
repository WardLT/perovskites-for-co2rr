"""Tools for ready from the Perovskite data of Emery et al."""

from ase import Atoms, data
import pandas as pd
import numpy as np


def load_emery_dataset() -> pd.DataFrame:
    """Load the Emery dataset with the actinides and unstable compounds filtered out"""

    init_data = pd.read_csv('../datasets/emory-abo3-2017.csv')
    init_data['max_Z'] = np.max([
        init_data['A'].apply(data.atomic_numbers.__getitem__).tolist(),
        init_data['B'].apply(data.atomic_numbers.__getitem__).tolist(),
    ], axis=0)
    init_data['Stability [eV/atom]'] = pd.to_numeric(init_data['Stability [eV/atom]'], errors='coerce')
    return init_data.query('`Stability [eV/atom]` < 0.1 and max_Z < 89')


def generate_initial_structure(row: pd.Series) -> Atoms:
    """Generate an initial structure given the data from the data

    Args:
        row: Description of a perovskite
    Returns:
        Example atomic structure
    """

    # Determine the lattice parameter
    lat_param = np.power(float(row["Volume per atom [A^3/atom]"]) * 5, 1. / 3)

    # Make the structure
    atoms = Atoms(
        symbols=[row['B'], row['A'], 'O', 'O', 'O'],
        scaled_positions=[
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.],
            [0.5, 0., 0.5],
            [0., 0.5, 0.5]
        ],
        cell=[lat_param] * 3,
        pbc=True
    )

    # Set magnetic momments in OQMD's style https://oqmd.org/documentation/vasp
    magmoms = np.zeros((len(atoms),))
    for i, z in enumerate(atoms.get_atomic_numbers()):
        if 21 <= z <= 30:
            magmoms[i] = 5
        elif 57 <= z <= 71:
            magmoms[i] = 7
    atoms.set_initial_magnetic_moments(magmoms)
    return atoms
