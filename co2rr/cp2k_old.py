"""Utilities for CP2K"""

from ase.calculators.cp2k import CP2K
from ase import data, units, Atoms

_pp_val = {
    'O': 6,
    'K': 9,  # 4s
    'Fe': 16,
    'As': 5,  # 4p
    'Nb': 13, 'Ag': 11,  # 4d TM
    'Rb': 9,  # 5s
    'Ba': 10,  # 6s
    'Eu': 17, 'Yb': 34,  # Lathanides
    'Ta': 13,  # 5d TM
    'Bi': 5
}
_basis_level = {
}


def make_kind_sections(elems: list[str]) -> tuple[str, int]:
    """Make the &KIND section of a CP2K input file

    Args:
        elems: List of elements in a file
    Returns:
        - &KIND sections ready to paste into input file
        - Required multiplicity
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
        else:
            level = _basis_level.get(elem, 'DZVP')
            basis = f'{level}-MOLOPT-SR-GTH-q{val}'

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
&END KIND"""
        output.append(subsys)
    return "\n".join(output), 1 if unpaired == 0 else unpaired + 1


def make_calculator(atoms: Atoms, cutoff: int = 500, max_scf: int = 64, add_mos: int = None, charge: int = 0) -> CP2K:
    """Make a calculator ready for a certain set of atoms

    Args:
        atoms: Atoms to be evaluated
        cutoff: Cutoff in Ry
        max_scf: Maximum number of SCF steps
        add_mos: Number of MOs to add
        charge: Total charge on the system
    Returns:
        A CP2K calculator
    """

    kind, mult = make_kind_sections(atoms.get_chemical_symbols())
    if add_mos is None:
        add_mos = len(atoms) * 4

    # Determine what kind of periodicity is required
    psolver = ''
    if not all(atoms.pbc):
        # Change from the default psolver
        periodic = ''.join(d if y else '' for d, y in zip('xyz', atoms.pbc))
        psolver = f"""&POISSON
            PERIODIC {periodic}
            POISSON_SOLVER WAVELET
        &END POISSON"""

    mult += charge
    return CP2K(
        inp=f"""
&FORCE_EVAL
    &DFT
        BASIS_SET_FILE_NAME BASIS_MOLOPT
        BASIS_SET_FILE_NAME BASIS_MOLOPT_LnPP1
        MULTIPLICITY {mult}
        UKS {"T" if mult > 1 else "F"}
        CHARGE {charge}
        &QS
            METHOD GAPW
        &END QS
        &SCF
            IGNORE_CONVERGENCE_FAILURE
            ADDED_MOS {add_mos}
            &SMEAR ON
                METHOD FERMI_DIRAC
                ELEC_TEMP [K] 300
            &END SMEAR
            &MIXING
                METHOD BROYDEN_MIXING
                ALPHA 0.2
                BETA 1.5
                NMIXING 8
            &END MIXING
        &END SCF
        {psolver}
        &XC
            &XC_FUNCTIONAL
                &PBE
                &END PBE
            &END XC_FUNCTIONAL
        &END XC
        &MGRID
            NGRIDS 5
            REL_CUTOFF 50
        &END MGRID
    &END DFT
    &SUBSYS
        {kind}
    &END SUBSYS
&END FORCE_EVAL""",
        xc=None,
        cutoff=cutoff * units.Ry,
        max_scf=max_scf,
        basis_set_file=None,
        basis_set=None,
        pseudo_potential=None,
        potential_file=None,
        stress_tensor=all(atoms.pbc),
        command='/home/lward/Software/cp2k-2024.1/exe/local/cp2k_shell.ssmp'
    )
