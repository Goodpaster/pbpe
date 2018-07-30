import numpy as np
cimport numpy as np

FTYPE = np.float
ctypedef np.float_t FTYPE_t
CTYPE = np.complex
ctypedef np.complex_t CTYPE_t

def concatenate_mols(mA, mB, ghost=False):
    '''Takes two pySCF mol objects, and concatenates them.
    If the option "ghost" is given, it keeps the ghost atoms
    from each mol, otherwise only the "real" atoms are copied.'''

    from pyscf import gto
    import re

    mC = gto.Mole()

    atmC = []
    ghstatm = {}
    nghost = 0

    # copy all atoms from mole A
    for i in range(mA.natm):
        if 'ghost' in mA.atom_symbol(i).lower():
            if ghost:
                nghost += 1
                atmC.append([mA.atom_symbol(i), mA.atom_coord(i)])
        else:
            atmC.append([mA.atom_symbol(i), mA.atom_coord(i)])
    mC.basis = mA.basis.copy()

    # copy all atoms from mole B
    for i in range(mB.natm):
        if 'ghost' in mB.atom_symbol(i).lower():
            if ghost:
                nghost += 1
                oghost = int(mB.atom_symbol(i).split(':')[1])
                newsym = 'GHOST:{0}'.format(nghost)
                atmC.append([newsym, mB.atom_coord(i)])
                mC.basis.update({newsym.lower(): mB.basis['ghost:{0}'.format(oghost)]})
        else:
            atmC.append([mB.atom_symbol(i), mB.atom_coord(i)])
            mC.basis.update({mB.atom_symbol(i): mB.basis[mB.atom_symbol(i)]})

    mC.atom = atmC
    mC.verbose = mA.verbose
    mC.charge = mA.charge + mB.charge
    mC.unit = 'bohr' # atom_coord is always stored in bohr (?)
    mC.build(dump_input=False)

    return mC

def concatenate_cells(mA, mB, ghost=False):
    '''Takes two pySCF mol objects, and concatenates them.
    If the option "ghost" is given, it keeps the ghost atoms
    from each mol, otherwise only the "real" atoms are copied.'''

    from pyscf.pbc import gto
    import re

    mC = gto.Cell()

    atmC = []
    ghstatm = {}
    b2a = 0.52917720859 # bohr to angstrom
    conversion = 1.
    if mA.unit in ('b', 'bohr'): conversion = b2a

    atmlabel = {'ghost': 0}
    # copy all atoms from mole A
    for i in range(mA.natm):
        if 'ghost' in mA.atom_symbol(i).lower():
            if ghost:
                atmC.append([mA.atom_symbol(i), mA.atom_coord(i)*b2a])
                atmlabel['ghost'] += 1
        else:
            atm = mA.atom_symbol(i).split(':')[0]
            if atm in atmlabel.keys():
                atmlabel[atm] += 1
            else:
                atmlabel.update({atm: 1})
            atmC.append([mA.atom_symbol(i), mA.atom_coord(i)*b2a])
    mC.basis = mA.basis.copy()

    # copy all atoms from mole B
    for i in range(mB.natm):
        if 'ghost' in mB.atom_symbol(i).lower():
            if ghost:
                atmlabel['ghost'] += 1
                oldsym = mB.atom_symbol(i)
                newsym = 'ghost:{0}'.format(atmlabel['ghost'])
                atmC.append([newsym, mB.atom_coord(i)*b2a])
                mC.basis.update({newsym.lower(): mB.basis[oldsym]})
        else:
            atm = mB.atom_symbol(i).split(':')[0]
            if atm in atmlabel.keys():
                atmlabel[atm] += 1
            else:
                atmlabel.update({atm: 1})
            oldsym = mB.atom_symbol(i)
            newsym = atm+":{0}".format(atmlabel[atm])
            atmC.append([newsym, mB.atom_coord(i)*b2a])
            mC.basis.update({newsym: mB.basis[oldsym]})

    mC.atom       = atmC
    mC.verbose    = mA.verbose
    mC.charge     = mA.charge + mB.charge
    mC.a          = mA.a * conversion
    mC.mesh       = mA.mesh
    mC.dimension  = mA.dimension
    mC.unit       = 'a'
    mC.ecp        = mA.ecp
    mC.max_memory = mA.max_memory
    mC.build(dump_input=False)

    return mC

def gen_grids(mol, a_vec, level=1):
    '''Generate a grid within the unit cell.'''

    from pyscf import dft

    # generate grid
    grids = dft.gen_grid.Grids(mol)
    grids.level = level
    grids.build()

    # make sure that grids are within unit cell
    # cubic cells only
    for i in range(3):

        a = np.where(grids.coords.transpose()[i] >= a_vec[i][i])
        grids.coords = np.delete(grids.coords, a, axis=0)
        grids.weights = np.delete(grids.weights, a, axis=0)
#        grids.weights[a] = 0.
#        grids.coords[a] -= inp.a_vec[i]
        a = np.where(grids.coords.transpose()[i] < 0.)
        grids.coords = np.delete(grids.coords, a, axis=0)
        grids.weights = np.delete(grids.weights, a, axis=0)
#        grids.coords[a] += inp.a_vec[i]
#        grids.weights[a] = 0.

    # return grids
    return grids

def g_2_mol(M, nA, nG):
    return M.reshape(nG,nA,nA).swapaxes(0,1).reshape(nA, nG*nA)
