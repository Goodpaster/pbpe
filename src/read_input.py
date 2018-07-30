def read_input(filename, build=True):
    '''Reads a formatted input file.'''

    from input_reader import InputReader
    import sys
    from pyscf import gto, dft, pbc
    from pyscf.pbc import gto as pbcgto, dft as pbcdft, df as pbcdf, scf as pbcscf
    import numpy as np
    from .mole import concatenate_cells
    import re
    from .simple_timer import timer
    from .integrals import get_sub2sup
    import h5py

    # initialize reader for a pySCF input
    reader = InputReader(comment=['!', '#', '::', '//'],
             case=False, ignoreunknown=True)

    # add finite subsystem block
    subsystem = reader.add_block_key('subsystem', required=True, repeat=True)
    subsystem.add_regex_line('atom',
        '\s*([A-Za-z.]+)\s+(\-?\d+\.?\d*)\s+(\-?\d+.?\d*)\s+(\-?\d+.?\d*)', repeat=True)
    subsystem.add_line_key('basis', type=str, default=None)     # unique subsystem basis

    # add embedding block
    embed = reader.add_block_key('embed')
    embed.add_line_key('cycles', type=int, default=200)     # max embedding cycles
    embed.add_line_key('conv', type=float, default=1e-6)    # embed energy convergence
    embed.add_line_key('method', type=str)                  # embedding method
    embed.add_boolean_key('freezeb')                        # optimize only subsystem A
    embed.add_line_key('subcycles', type=int, default=10)   # number of subsys diagonalizations
    operator = embed.add_mutually_exclusive_group(dest='operator', required=True)
    operator.add_line_key('mu', type=float, default=1e6)    # manby operator by mu
    operator.add_boolean_key('manby', action=1e6)           # manby-miller operator
    operator.add_boolean_key('huzinaga', action='huzinaga') # huzinaga operator

    # lattice vectors
    lattice = reader.add_block_key('lattice', required=True)
    lattice.add_regex_line('vector', '\s*(\d+.?\d*)\s+(\d+.?\d*)\s+(\d+.?\d*)',
        repeat=True)

    # k-points
    kgroup = reader.add_mutually_exclusive_group(dest='kgroup', required=True)
    kgroup.add_line_key('kpoints', type=[int, int, int])
    kscaled = kgroup.add_block_key('kgrid')
    kscaled.add_regex_line('kpoints', '\s*(\-?\d+\.?\d*)\s+(\-?\d+\.?\d*)\s+(\-?\d+\.?\d*)',
                            repeat=True)

    # grid/mesh points
    mesh = reader.add_mutually_exclusive_group(dest='mgroup', required=True)
    mesh.add_line_key('gspacing', type=float)
    mesh.add_line_key('gs', type=[int, int, int])
    mesh.add_line_key('mesh', type=[int, int, int])

    # basis block key
    basis = reader.add_block_key('basis')
    basis.add_regex_line('atom', '\s*([A-Za-z]+)\s+([A-Za-z0-9]+)', repeat=True)

    # dimensions
    reader.add_line_key('dimension', type=(0,1,2,3), required=True)
    reader.add_line_key('bspace', type=float, default=2.0)

    # general line keys
    reader.add_line_key('unit', type=('angstrom','a','bohr','b'), default='a')
    reader.add_line_key('method', default='lda,vwn')     # high level method
    reader.add_line_key('verbose', type=int, default=3)  # pySCF verbose level
    reader.add_line_key('maxiter', type=int, default=50) # max SCF iteration
    reader.add_line_key('supcycles', type=int, default=None) # SCF cycles for sup. pDFT
    reader.add_line_key('grid', type=int, default=1)     # Becke grid level
    reader.add_line_key('pseudo', type=str)              # pseudo potential
    reader.add_line_key('memory', type=(int, float))     # max memory in MB

    # line keys with good defaults (shouldn't need to change these)
    reader.add_line_key('auxbasis', type=str, case=True, default='weigend')
    reader.add_line_key('fit', type=('df', 'mdf', 'pwdf', 'fftdf'), default='df')
    reader.add_line_key('exxdiv', type=('vcut_sph', 'ewald', 'vcut_ws'),
                        default='ewald')

    # line keys to help with convergence
    reader.add_line_key('conv', type=float, default=1e-8)
    reader.add_line_key('diis', type=int, default=1)       # DIIS start cycle (0 to turn off)
    reader.add_line_key('smear', type=float, default=None) # Fermi smearing
    reader.add_line_key('mix', type=float, default=None)   # Fock matrix mixing
    reader.add_line_key('shift', type=float, default=None) # orbital energy shift
    reader.add_line_key('dmix', type=float, default=None)  # Density matrix mixing

    # add boolean keys
    reader.add_boolean_key('finite')     # do finite embedding
    reader.add_boolean_key('ldob')       # update B
    reader.add_boolean_key('fractional') # fractional input coordinates
    reader.add_boolean_key('huzfermi')   # shifts energy to set fermi to zero
    reader.add_boolean_key('fcidump')    # creates an .fcidump file for sup calc.

    # electric core potential (ECP)
    ecp = reader.add_block_key('ecp')
    ecp.add_regex_line('atom',
        '\s*([A-Za-z]+)\s+([A-Za-z0-9]+)', repeat=True)

    # read the input file
    inp = reader.read_input(filename)
    inp.filename = filename

    # start the timer
    inp.timer  = timer()

    # sanity checks
    lattice = []
    for r in inp.lattice.vector:
        lattice.append(np.array([r.group(1), r.group(2), r.group(3)], dtype=float))
    inp.lattice = np.array(lattice)
    if len(inp.lattice) < inp.dimension:
        sys.exit("Must provide as many LATTICE constants as DIMENSIONS!")
    if len(inp.subsystem) > 2:
        sys.exit("Only ONE or TWO subsystems can be used right now!")
    if inp.basis is None:
        if ((inp.subsystem[0].basis is not None and inp.subsystem[1].basis is None)
            or (inp.subsystem[1].basis is not None and inp.subsystem[0].basis is None)):
            sys.exit("Both subsystem basis must be specified!")

    # initialize pySCF cell objects and set some defaults
    inp.nsub = len(inp.subsystem)
    inp.cell = [None for c in range(inp.nsub)]
    inp.fcell = [None for c in range(inp.nsub)]
    if inp.auxbasis.lower() == 'none': inp.auxbasis = None
    if inp.embed is None: inp.embed = class_embed()
    if inp.embed.method is None: inp.embed.method = inp.method
    if inp.embed.method in ('ccsd', 'ccsd(t)'): inp.embed.method = 'hf'
    if inp.supcycles is None: inp.supcycles = inp.maxiter
    inp.embed.dconv = np.sqrt(inp.embed.conv) # density matrix convergence
    inp.dconv = np.sqrt(inp.conv)             # density matrix convergence

    # first, get all coordinates to figure out the size of the unit cell
    # in the non-periodic directions
    allcoords = []
    for c in range(inp.nsub):

        coords = []
        # get all atom coordinates
        for r in inp.subsystem[c].atom:
            coord = np.array([r.group(2), r.group(3), r.group(4)], dtype=float)
            coord = f2r(coord, inp.lattice, inp.dimension, inp.fractional)
            coords.append(coord)
        coords = np.array(coords, dtype=float)
        allcoords.append(coords)

    # get unit cell dimensions in non-periodic directions
    if len(inp.lattice) < 3:
        for d in range(1,3):
            if inp.dimension <= d:
                bmin = 1e6
                bmax = -1e6
                for i in range(len(allcoords)):
                    bmin = min(bmin, allcoords[i].transpose()[d].min())
                    bmax = max(bmax, allcoords[i].transpose()[d].max())
                bmax = bmax - bmin + 2.0 * inp.bspace
                bmin = inp.bspace - bmin
                for i in range(len(allcoords)):
                    allcoords[i].transpose()[d] += bmin
                tempvec = np.zeros((3))
                tempvec[d] = bmax
                inp.lattice = np.append(inp.lattice, [tempvec], axis=0)

    # create 3x3 lattice, print to screen
    conversion = 1.
    if inp.unit in ('bohr','b'): conversion = 0.52917720859
    b2a = 0.52917720859
    a2b = 1.88972613289
    print ('Lattice      Coordinate / Bohr               Coordinate / Angstrom')
    for i in range(3):
        la = inp.lattice * conversion
        lb = a2b * la
        print ('A{0:1d}     {1:9.6f} {2:9.6f} {3:9.6f}     {4:9.6f} '
               '{5:9.6f} {6:9.6f}'.format(i+1, lb[i][0],
               lb[i][1], lb[i][2], la[i][0], la[i][1], la[i][2]))
    print ('')

    # mesh parameter
    if isinstance(inp.mgroup, int) or isinstance(inp.mgroup, float):
        inp.mesh = np.array(np.round(inp.lattice.diagonal() / inp.mgroup, 0), dtype=int)
    else:
        inp.mesh = inp.mgroup

    # electric core potential
    ecp = None
    if inp.ecp is not None:
        ecp = {}
        for r in inp.ecp.atom:
            ecp.update({r.group(1): r.group(2)})

    # initialize HDF5 file using h5py
    if inp.filename[-4:] == '.inp':
        h5pyname = inp.filename[:-4]+'.hdf5'
    else:
        h5pyname = inp.filename+'.hdf5'
    inp.h5py = h5py.File(h5pyname)

    # basis block into basis dict
    basis = {}
    for r in inp.basis.atom:
        basis.update({r.group(1): r.group(2)})

    # create atom coords and basis
    bastype = []
    for c in range(inp.nsub):

        # initialize cells
        cell = pbcgto.Cell()    # cell with ghost atoms
        fcell = pbcgto.Cell()   # cell with ghosts replaced by real atoms

        # collect atoms
        cell.atom = []
        atmlabel = {'ghost': []}
        fatlabel = {}
        for i, r in enumerate(inp.subsystem[c].atom):
            coord = allcoords[c][i]
            if 'ghost.' in r.group(1).lower() or 'gh.' in r.group(1).lower():
                bastype.append(r.group(1).split('.')[1])
                atmstr = ['ghost:{0}'.format(len(bastype)), coord]
                cell.atom.append(atmstr)
                atmstr = ['{0}:{1}'.format(r.group(1).split('.')[1], len(bastype)), coord]
                fcell.atom.append(atmstr)

            else:
                bastype.append(r.group(1))
                atmstr = ['{0}:{1}'.format(r.group(1), len(bastype)), coord]
                cell.atom.append(atmstr)
                fcell.atom.append(atmstr)

        # build dict of basis for each atom
        cell.basis = {}
        fcell.basis = {}

        for iat in range(len(cell.atom)):
            at_len = int(cell.atom[iat][0].split(':')[1]) - 1
            at_typ = bastype[at_len]

            bas_typ = 'sto3g'
            if inp.subsystem[c].basis is not None:
                bas_typ = inp.subsystem[c].basis
            elif at_typ in basis.keys():
                bas_typ = basis[at_typ]

            cell.basis.update({cell.atom[iat][0]: pbcgto.basis.load(bas_typ, at_typ)})
            fcell.basis.update({fcell.atom[iat][0]: pbcgto.basis.load(bas_typ, at_typ)})

        # use local values first, then global values, if they exist
        if inp.memory is not None: cell.max_memory = inp.memory
        cell.unit = inp.unit
        fcell.unit = inp.unit
        cell.verbose = inp.verbose
        fcell.verbose = inp.verbose

        # set lattice vectors
        cell.a = inp.lattice
        fcell.a = inp.lattice

        #pseudo potentials
        if inp.pseudo is not None:
            cell.pseudo = inp.pseudo
            fcell.pseudo = inp.pseudo

        # electric core potential
        if ecp is not None:
            cell.ecp = ecp
            fcell.ecp = ecp

        # cell grid points
        cell.mesh = inp.mesh
        fcell.mesh = inp.mesh

        # cell dimension
        cell.dimension = inp.dimension
        fcell.dimension = inp.dimension

        # build cell object
        cell.build(dump_input=False)
        inp.cell[c] = cell
        fcell.build(dump_input=False)
        inp.fcell[c] = fcell

    # generate concatenated cell object (without ghost)
    if inp.nsub > 1:
        ccell = concatenate_cells(inp.cell[0], inp.cell[1], ghost=False)
        for i in range(2, inp.nsub):
            ccell = concatenate_cells(ccell, inp.cell[i], ghost=False)
        inp.ccell = ccell
    else:
        inp.ccell = ccell = inp.cell[0]

    if inp.pseudo is not None:
        inp.ccell.pseudo = inp.pseudo
        inp.ccell.build()

    # print subsystem coordinates
    for i in range(inp.nsub):
        print ('Subsystem {0}:'.format(i+1))
        print_coords(inp.cell[i])
    print ('Supermolecular system:')
    print_coords(inp.ccell)

    # make kpts
    if inp.kgroup.__class__ is tuple:
        inp.kpts = inp.ccell.make_kpts(inp.kgroup)
        inp.nkpts = len(inp.kpts)
    else:
        kabs = np.array([r.group(0).split() for r in inp.kgroup.kpoints], dtype=float)
        inp.kpts = inp.ccell.get_abs_kpts(kabs)
        inp.nkpts = len(inp.kpts)
#    inp.lgamma = inp.nkpts == 1
    inp.lgamma = False

    # print k-points
    print ('')
    print ('K-points (inv bohr):')
    print (repr(inp.kpts))

    # make periodic SCF objects
    if inp.lgamma:
        SCF = pbcdft.RKS
    else:
        SCF = pbcdft.KRKS
    ldft = True
    if inp.embed.method.lower() in ('hf', 'hartree-fock'):
        if inp.lgamma:
            SCF = pbcscf.RHF
        else:
            SCF = pbcscf.KRHF
        ldft = False

    inp.sSCF = SCF(inp.ccell, inp.kpts)
    inp.sSCF.kpts = inp.kpts
    if ldft: inp.sSCF.xc = inp.embed.method
    inp.sSCF.init_guess = 'atom'
    inp.sSCF.grids = pbcdft.gen_grid.BeckeGrids(inp.ccell)
    inp.sSCF.grids.level = inp.grid
    inp.sSCF.grids.build()

    inp.cSCF = [None for i in range(inp.nsub)]
    for i in range(inp.nsub):
        inp.cSCF[i] = SCF(inp.cell[i], inp.kpts)
        inp.cSCF[i].kpts = inp.kpts
        inp.cSCF[i].max_cycle = inp.embed.subcycles
        if ldft: inp.cSCF[i].xc = inp.embed.method
        inp.cSCF[i].init_guess = 'atom'
        inp.cSCF[i].grids = inp.sSCF.grids

    # account for density fitting
    if inp.fit == 'fftdf':
        inp.DF = pbcdf.FFTDF
    elif inp.fit == 'mdf':
        inp.DF = pbcdf.MDF
    elif inp.fit == 'pwdf':
        inp.DF = pbcdf.PWDF
    else:
        inp.DF = pbcdf.DF

    # do we need only density fitting for J?
    if build:
        inp.timer.start('density fitting')
        try:
            j_only = abs(inp.sSCF._numint.hybrid_coeff(inp.sSCF.xc)) < 1e-3
        except AttributeError:
            j_only = False

        # create density fitting object
        inp.sSCF.with_df = inp.DF(inp.ccell)
        inp.sSCF.with_df.kpts = inp.kpts
        inp.sSCF.with_df.auxbasis = inp.auxbasis
        if inp.embed.method.lower() in ('hf', 'hartree-fock'):
            inp.sSCF.exxdiv = inp.exxdiv
        inp.sSCF.with_df.build(j_only=j_only)

        for i in range(inp.nsub):
            inp.cSCF[i].with_df = inp.DF(inp.fcell[i])
            inp.cSCF[i].with_df.kpts = inp.kpts
            inp.cSCF[i].with_df.auxbasis = inp.auxbasis
            if inp.embed.method.lower() in ('hf', 'hartree-fock'):
                inp.cSCF[i].exxdiv = inp.exxdiv
            if inp.embed.cycles > -1 and i == 0:
                inp.cSCF[i].with_df.build(j_only=j_only)
        inp.timer.end('density fitting')

    # initialize other values
    inp.nao    = np.array([inp.cell[i].nao_nr() for i in range(inp.nsub)])
    inp.DIIS   = [None for i in range(inp.nsub+1)]
    inp.FOCK   = None
    inp.Fock   = [None for i in range(inp.nsub)]
    inp.Energy = [None for i in range(inp.nsub)]
    inp.Dmat   = [None for i in range(inp.nsub)]
    if 'fermi' in inp.h5py:
        inp.Fermi = inp.h5py['fermi']
    else:
        inp.Fermi = np.zeros((inp.nsub))
        inp.Fermi = inp.h5py.create_dataset('fermi', data=inp.Fermi)
    get_sub2sup(inp)

    # return
    return inp


class class_embed():
    '''A class to hold the embed options.'''

    def __init__(self):
        self.cycles     = -1
        self.conv       = 1e-8
        self.method     = None
        self.freezeb    = False
        self.subcycles  = 1
        self.operator   = 'huzinaga'


def f2r(coord, lattice, dimension, fractional):
    '''Takes a fractional atom coordinate, and returns the
    real atom coordinate based on the lattice parameters and
    the system dimension.
    NB: 0D assumes that no coord is fractional
        1D assumes only x coord is fractional
        2D assumes x and y coords are fractional
        3D assumes x, y, z coords are fractional
    '''
    import numpy as np
    if fractional:
        nc = np.zeros((3))
        if dimension >= 1:
            nc += coord[0] * lattice[0]
        if dimension >= 2:
            nc += coord[1] * lattice[1]
        if dimension == 3:
            nc += coord[2] * lattice[2]
    else:
        nc = coord.copy()
    return nc


def print_coords(cell):

    b2a = 0.52917720859

    print ('Atom         Coordinate / Bohr               Coordinate / Angstrom')
    for i in range(cell.natm):
        cb = cell.atom_coord(i)
        ca = cb * b2a
        sym = cell.atom_symbol(i).split(':')[0]
        print ('{0:<5}  {1:9.6f} {2:9.6f} {3:9.6f}     {4:9.6f} '
               '{5:9.6f} {6:9.6f}'.format(sym, cb[0],
               cb[1], cb[2], ca[0], ca[1], ca[2]))
    print ('')
