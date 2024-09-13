"""Utilities for CP2K"""
from shutil import copyfile
from pathlib import Path

from ase.calculators.cp2k import CP2K
from ase import data, units, Atoms

pp_val = {
    'Li': 3, 'Be': 4,
    'C': 4, 'O': 6,
    'Na': 9, 'Mg': 10,
    'K': 9, 'Ca': 10,  # 4s
    'Al': 3, 'Si': 4,  # 3p
    'Sc': 11, 'Ti': 12, 'V': 13, 'Cr': 14, 'Mn': 15, 'Fe': 16, 'Co': 17, 'Ni': 18, 'Cu': 11, 'Zn': 12,  # 3d TM
    'Ga': 13, 'Ge': 4, 'As': 5,  # 4p
    'Rb': 9, 'Sr': 10,  # 5s
    'Y': 11, 'Zr': 12, 'Nb': 13, 'Mo': 14, 'Tc': 15, 'Ru': 16, 'Rh': 17, 'Pd': 18, 'Ag': 11, 'Cd': 12,  # 4d TM
    'In': 13, 'Sn': 4, 'Sb': 5, 'Te': 6,  # 5p
    'Cs': 9, 'Ba': 10,  # 6s
    'La': 11, 'Hf': 12, 'Ta': 13, 'W': 14, 'Re': 15, 'Os': 16, 'Ir': 17, 'Pt': 18, 'Au': 11, 'Hg': 12,  # 5d TM
    'Ce': 12, 'Pr': 13, 'Nd': 14, 'Pm': 15, 'Sm': 16, 'Eu': 17, 'Gd': 18,
    'Tb': 29, 'Dy': 30, 'Ho': 31, 'Er': 32, 'Tm': 33, 'Yb': 34, 'Lu': 35,  # Lanthanides
    'Tl': 13, 'Pb': 4, 'Bi': 5  # 6p
}
_basis_level = {
    'Mg': 'TZV2Pd',
    'Na': 'TZV2Pd'
}


def make_kind_sections(elems: list[str]) -> tuple[str, bool]:
    """Make the &KIND section of a CP2K input file

    Args:
        elems: List of elements in a file
    Returns:
        - &KIND sections ready to paste into input file
        - Whether we need UKS
    """

    output = []
    unpaired = 0
    total_e = 0
    for elem in set(elems):
        # Get the valance and basis set
        z = data.atomic_numbers[elem]
        val = pp_val[elem]
        pot = f'GTH-PBE-q{val}'
        if 57 <= z <= 71:
            pot += '\n    POTENTIAL_FILE_NAME LnPP1_POTENTIALS'
            basis = 'DZV-MOLOPT-SR-GTH' if z > 57 else 'DZV-MOLOPT-GTH'
        elif z in [14, 6, 8, 1]:
            basis = 'TZV2P-MOLOPT-GTH'
        else:
            level = _basis_level.get(elem, 'TZV2P')
            basis = f'{level}-MOLOPT-SR-GTH-q{val}'

        # Assign DFT+U Values
        _oqmd_u_values = {
            "V": 3.1,
            "Cr": 3.5,
            "Mn": 3.8,
            "Fe": 4.0,
            "Co": 3.3,
            "Ni": 6.4,
            "Cu": 4.0,
            "Th": 4.0,
            "U": 4.0,
            "Np": 4.0,
            "Pu": 4.0,
        }
        u_value = _oqmd_u_values.get(elem, None)

        # Get the magnetization
        is_ln = 57 <= z <= 69
        if 21 <= z <= 30:
            magmom = min(5, z - 20 if z < 26 else 30 - z)
        elif 57 <= z <= 67:  # Er,Tm,Lu,Yb have full f or d shells. Skip magnetization
            magmom = min(5, z - 57 if z < 64 else 71 - z)
        else:
            magmom = 0
        unpaired += elems.count(elem) * magmom
        total_e += elems.count(elem) * val

        subsys = f"""&KIND {elem}
    BASIS_SET {basis}
    POTENTIAL {pot}
    MAGNETIZATION {magmom}
"""

        #        if u_value is not None:
        #            subsys += f"""\n&DFT_PLUS_U
        #    L 2
        #    U_MINUS_J [eV] {u_value}
        # &END"""

        subsys += "\n&END KIND"""
        output.append(subsys)
    return "\n".join(output), unpaired != 0 or total_e % 2 == 1


def make_calculator(
        atoms: Atoms,
        cutoff: int = 500,
        max_scf: int = 64,
        charge: int = 0,
        uks: bool = False,
        outer_scf: int = 5,
        xc_name: str = 'pbe-plus-u',
        command: str | None = None,
        compute_pdos: bool = False,
        run_dir: Path = Path('run'),
        wfn_guess: Path | None = None
) -> CP2K:
    """
    Make a calculator ready for a certain set of atoms

    Adapted from https://pubs.acs.org/doi/10.1021/acs.jpclett.2c03788#notes5

    Args:
        atoms: Atoms to be evaluated
        cutoff: Cutoff in Ry
        max_scf: Maximum number of SCF steps
        charge: Total charge on the system
        uks: Perform a spin-unrestricted calculation, regardless whether magnetic or not
        outer_scf: Number of outer SCF steps
        command: CP2K command
        compute_pdos: Whether to compute the PDOS, which also adds MOS and changes the optimizer.
          Also prints the charge information
        wfn_guess: Path to a wfn backup file to use as the starting point
    Returns:
        A CP2K calculator
    """

    kind, uks_my = make_kind_sections(atoms.get_chemical_symbols())

    # Determine what kind of periodicity is required
    psolver = ''
    if not all(atoms.pbc):
        # Change from the default psolver
        periodic = ''.join(d if y else '' for d, y in zip('xyz', atoms.pbc))
        psolver = f"""&POISSON
    PERIODIC {periodic}
    POISSON_SOLVER MT
    &END POISSON
"""

    # Decide the SCF algorithm
    if compute_pdos:
        outputs = f"""&PRINT
     &PDOS
        # print all projected DOS available:
        NLUMO -1
        # split the density by quantum number:
        COMPONENTS
        FILENAME {run_dir}/run
     &END
      &E_DENSITY_CUBE ON
        FILENAME {run_dir}/density
        STRIDE 1 1 1
      &END E_DENSITY_CUBE
      &MULLIKEN ON
        FILENAME =run/mulliken.charges
      &END MULLIKEN
&END PRINT"""
    else:
        outputs = ""
    scf = """
&OT
    ALGORITHM IRAC
    MINIMIZER CG
    # NDIIS 8
    PRECONDITIONER FULL_SINGLE_INVERSE
&END OT"""

    if wfn_guess is not None:
        copyfile(wfn_guess, run_dir / 'cp2k-RESTART.wfn')
        scf += "\nSCF_GUESS RESTART"

    # Choose the XC functional
    if xc_name == 'pbe-plus-u':
        xc_block = """&XC_FUNCTIONAL
    &PBE
    &END PBE
&END XC_FUNCTIONAL
&VDW_POTENTIAL
    DISPERSION_FUNCTIONAL NON_LOCAL
    &NON_LOCAL
      TYPE RVV10
      KERNEL_FILE_NAME rVV10_kernel_table.dat
      PARAMETERS 9.3 9.3E-003
    &END NON_LOCAL
&END VDW_POTENTIAL"""
    elif xc_name == 'hse':
        xc_block = """&XC_FUNCTIONAL
    &PBE
    SCALE_X 0.0
    SCALE_C 1.0
    &END PBE
    &XWPBE
        SCALE_X -0.25
        SCALE_X0 1.0
        OMEGA 0.11
    &END XWPBE
&END XC_FUNCTIONAL
&HF
    &SCREENING
        EPS_SCHWARZ 1.0E-6
        SCREEN_ON_INITIAL_P FALSE
    &END SCREENING
    &INTERACTION_POTENTIAL
    POTENTIAL_TYPE SHORTRANGE
    OMEGA 0.11
    &END INTERACTION_POTENTIAL
    &MEMORY
    MAX_MEMORY 2400
    EPS_STORAGE_SCALING 0.1
    &END MEMORY
    FRACTION 0.25
&END HF
"""
    else:
        raise NotImplementedError()

    return CP2K(
        inp=f"""
&FORCE_EVAL
&DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    BASIS_SET_FILE_NAME BASIS_MOLOPT_UCL
    BASIS_SET_FILE_NAME BASIS_MOLOPT_LnPP1
    UKS {"T" if uks_my or uks else "F"}
    CHARGE {charge}
    &QS
        METHOD GPW
    &END QS
    &SCF
        EPS_SCF 1.0E-5
        IGNORE_CONVERGENCE_FAILURE
        &OUTER_SCF
            MAX_SCF {outer_scf}
        &END OUTER_SCF
        {scf}
    &END SCF
    {psolver}
    &XC
        {xc_block}
    &END XC
    &MGRID
        NGRIDS 5
        REL_CUTOFF 50
    &END MGRID
    {outputs}
&END DFT
&SUBSYS
    {kind}
&END SUBSYS
&END FORCE_EVAL
""",
        xc=None,
        cutoff=cutoff * units.Ry,
        max_scf=max_scf // outer_scf,
        basis_set_file=None,
        basis_set=None,
        pseudo_potential=None,
        potential_file=None,
        stress_tensor=all(atoms.pbc),
        set_pos_file=True,
        command=command,
        directory=run_dir
    )
