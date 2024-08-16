"""Utilities for CP2K"""

from ase.calculators.cp2k import CP2K
from ase import data, units, Atoms

_pp_val = {
    'O': 6,
    'K': 9,  # 4s
    'Fe': 16,
    'As': 5,  # 4p
    'Nb': 13, 'Mo': 14, 'Ag': 11,  # 4d TM
    'Rb': 9,  # 5s
    'Ba': 10,  # 6s
    'Hf': 12,  # 5f
    'Eu': 17, 'Yb': 34,  # Lanthanides
    'Ta': 13,  # 5d TM
    'Bi': 5
}
_basis_level = {
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
        val = _pp_val[elem]
        pot = f'GTH-PBE-q{val}'
        if 57 <= z <= 71:
            pot += '\n    POTENTIAL_FILE_NAME LnPP1_POTENTIALS'
            basis = 'DZV-MOLOPT-SR-GTH'
        elif z in [6, 8, 1]:
            basis = 'TZV2P-MOLOPT-GTH'
        else:
            level = _basis_level.get(elem, 'TZV2P')
            basis = f'{level}-MOLOPT-SR-GTH-q{val}'

        # Assign DFT+U Values
        # _oqmd_u_values = {
        #     "V": 3.1,
        #     "Cr": 3.5,
        #     "Mn": 3.8,
        #     "Fe": 4.0,
        #     "Co": 3.3,
        #     "Ni": 6.4,
        #     "Cu": 4.0,
        #     "Th": 4.0,
        #     "U": 4.0,
        #     "Np": 4.0,
        #     "Pu": 4.0,
        # }
        # u_value = _oqmd_u_values.get(elem, None)

        # Get the magnetization
        if 21 <= z <= 30:
            magmom = 5
        elif 57 <= z <= 69:  # Lu,Yb have full f shells. Skip magnetization
            magmom = 7
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
    return "\n".join(output), unpaired == 0 or total_e % 2 == 1


def make_calculator(atoms: Atoms, cutoff: int = 500, max_scf: int = 64, charge: int = 0, uks: bool = False, outer_scf: int = 5) -> CP2K:
    """
    Make a calculator ready for a certain set of atoms

    Adapted from https://pubs.acs.org/doi/10.1021/acs.jpclett.2c03788#notes5

    Args:
        atoms: Atoms to be evaluated
        cutoff: Cutoff in Ry
        max_scf: Maximum number of SCF steps
        charge: Total charge on the system
        uks: Perform a spin-unrestricted calculation, regardless of magnetic or not
        outer_scf: Number of outer SCF steps
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
        &OT
            ALGORITHM IRAC
            MINIMIZER CG
            # NDIIS 8
            PRECONDITIONER FULL_SINGLE_INVERSE
        &END OT
    &END SCF
    {psolver}
    &XC
        &XC_FUNCTIONAL
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
        &END VDW_POTENTIAL
    &END XC
    &MGRID
        NGRIDS 5
        REL_CUTOFF 50
    &END MGRID
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
        command='/home/lward/Software/cp2k-2024.1/exe/local/cp2k_shell.ssmp'
    )
