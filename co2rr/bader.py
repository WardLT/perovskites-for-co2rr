"""Compute partial charges from charge density"""
import gzip
from tempfile import TemporaryDirectory
from shutil import copyfileobj
from subprocess import run
from pathlib import Path
import sys

import pandas as pd

_file_dir = Path(__file__).parent / 'files'
_atomic_density_folder_path = Path(sys.prefix) / "share" / "chargemol" / "atomic_densities"


def compute_partial_charges(cube_path: Path) -> list[float]:
    """Compute partial charges with DDEC

    Args:
        cube_path: Path to a cube file
    Returns:
        Charges for each atom
    """

    # Make a copy of the input file
    with TemporaryDirectory('chgmol') as tmpdir:
        tmpdir = Path(tmpdir)

        # my local CP2k is not renaming the output automatically, so an extra renaming step is added here
        with (tmpdir / 'valence_density.cube').open('wb') as fo, gzip.open(cube_path, 'rb') as fi:
            copyfileobj(fi, fo)

        # chargemol uses all cores for OpenMP parallelism if no OMP_NUM_THREADS is set
        stderr_path = tmpdir / 'bader.stderr'
        with open(tmpdir / 'bader.stdout', 'w') as fo, open(stderr_path, 'w') as fe:
            proc = run(["bader", 'valence_density.cube'], cwd=tmpdir, stdout=fo, stderr=fe)
        if proc.returncode != 0:
            raise ValueError(f'bader failed in {tmpdir}.\nSTDERR\n: {stderr_path.read_text()}')

        # Copy the ACF file
        run_type = cube_path.name.split(".")[0]
        out_path = cube_path.parent / f'{run_type}.ACF.dat'
        (tmpdir / 'ACF.dat').rename(out_path)
        return pd.read_csv(out_path, sep='\s+', skipfooter=4, skiprows=[1], engine='python')['CHARGE']
