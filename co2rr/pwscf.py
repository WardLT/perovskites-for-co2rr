"""Utilities for running PWSCF computations"""
from subprocess import run
from pathlib import Path
import json
import re

import numpy as np
from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile

import pandas as pd

pwscf_bin_dir = Path.home() / 'Software' / 'q-e-qe-7.3' / 'bin'
_name_re = re.compile(r'pdos_atm#(?P<num>\d+)\((?P<elem>[A-Z][a-z]?)\)_wfc#\d\((?P<shell>[a-z])\)')

# Load in the PPs
file_dir = Path(__file__).parent / '..' / 'files'
sssp_dir = file_dir / 'SSSP_efficiency'
pp_list = json.loads((sssp_dir / 'SSSP_1.3.0_PBE_efficiency.json').read_text())
max_ecutwfc = 90 * 1.2
max_ecutrho = max_ecutwfc * 12


def setup_calculator(atoms: Atoms, use_max: bool = True, postfix: str = 'relax') -> tuple[Espresso, Path]:
    """Configure GPAW to run with the target structure

    Args:
        atoms: Atoms to be evaluated
        use_max: Whether to use the maximum cutoff energy
        postfix: Name of the run directory
    Returns:
        GPAW calculator
    """

    espresso_profile = EspressoProfile(
        str((file_dir / "qe-run.sh").absolute()),
        pseudo_dir=str(sssp_dir.resolve())
    )

    # Lookup the U-values
    # TODO (wardlt) use these
    u_ref = {
        'V': 3.1,
        'Cr': 3.5,
        'Mn': 3.8,
        'Fe': 4.0,
        'Co': 3.3,
        'Ni': 6.4,
        'Cu': 4.0,
        'Th': 4.0,
        'U': 4.0,
        'Np': 4.0,
        'Pu': 4.0
    }
    u_values = {}
    for symbol in set(atoms.symbols):
        if symbol in u_ref:
            u_values[symbol] = f':d,{u_ref[symbol]}'

    # Get the pseudopotentials
    pps = dict(
        (s, pp_list[s]['filename']) for s in set(atoms.symbols)
    )

    # Determine the cutoff energy
    if use_max:
        ecutrho, ecutwfc = max_ecutrho, max_ecutwfc
    else:
        ecutrho = max(
            pp_list[s]['cutoff_rho'] for s in set(atoms.symbols)
        )
        ecutwfc = max(
            pp_list[s]['cutoff_wfc'] for s in set(atoms.symbols)
        )

    run_dir = Path(f'../qe-run/{atoms.get_chemical_formula()}-{postfix}')
    return Espresso(
        profile=espresso_profile,
        directory=run_dir,
        pseudopotentials=pps,
        kspacing=0.1,
        input_data={
            'system': {
                'ecutrho': ecutrho, 'ecutwfc': ecutwfc,
                'occupations': 'smearing', 'degauss': 0.02,
            },
            'control': {'tstress': True, 'tprnfor': True},
            'electrons': {'mixing_beta': 0.5}
        }
    ), run_dir


def extract_projected_dos(run_path: Path, prefix: str = 'pwscf') -> pd.DataFrame:
    """Extract the projected density of states from an PWSCF computation

    Assumes that projwfc.x is on the path.

    Args:
        run_path: Path to the completed PWSCF computation
    Returns:
        Dataframe containing the density of states, split by atom and orbital type (s/p/d/f)
    """

    # Write the POS data to a subdirectory
    out_dir = run_path / 'projdos'
    out_dir.mkdir(exist_ok=True)
    input_file = f'''&projwfc
    prefix = '{prefix}'
    outdir = '{run_path.absolute()}'
/
'''
    (out_dir / 'projwfc.inp').write_text(input_file)
    error_file = out_dir / 'projwfc.err'
    with open(out_dir / 'projwfc.out', 'w') as fo, error_file.open('w') as fe:
        proc = run(
            [pwscf_bin_dir / 'projwfc.x', '-in', 'projwfc.inp'],
            stdout=fo,
            stderr=fe,
            cwd=out_dir
        )
    if proc.returncode != 0:
        raise ValueError(f'Wave function projection failed: {error_file.read_text()}')

    return parse_projected_dos(out_dir)


def parse_projected_dos(out_dir: Path) -> pd.DataFrame:
    """Parse the projected DOS files from projwfc.x

    Args:
        out_dir: Path ot the output files
    Returns:
        Dataframe containing the density of states, split by atom and orbital type (s/p/d/f)
    """

    # Read each output file
    p_dos = []
    for file in out_dir.glob('*pdos_atm*'):
        # Determine whether we have spin-polarized or not
        with file.open() as fp:
            header = fp.readline()
            spin = header.count('ldos') == 2

        # Read the data
        data = pd.read_csv(file, usecols=[0, 1, 2] if spin else [0, 1],
                           names=['energy'] + (['pdos_up', 'pdos_down'] if spin else ['pdos']),
                           sep='\s+', header=None, skiprows=1)

        # Add in the metadata from file name
        match = _name_re.search(file.name)
        if match is None:
            raise ValueError(f'Name match failure with: {file.name}')
        atom_id, elem, shell = match.groups()
        data['atom'] = int(atom_id)
        data['elem'] = elem
        data['shell'] = shell

        p_dos.append(data)

    return pd.concat(p_dos)
