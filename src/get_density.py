#! /usr/bin/env python
from __future__ import print_function, division
import numpy as np
import pickle

def main(filename, x=None, y=None, z=None, sep=0.01):

    import pyscf
    from matplotlib import pyplot as plt
    import os
    from matplotlib.colors import LogNorm
    from matplotlib import cm

    # get input
    inp        = read_input(filename)
    inp.nao    = np.array([inp.cell[i].nao_nr() for i in range(inp.nsub)])
    nkpts = float(inp.nkpts)
    spacing = 0.1
    A2B = 1.8897261328856432

    # read densities from file
    print ('Reading densities from file: {0}'.format(inp.save+'.*'))
    inp.Dmat = pickle.load(open(inp.save+'.dmat', 'r'))

    # get coords of the atoms
    acharges = inp.ccell.atom_charges()
    acoords = inp.ccell.atom_coords() / A2B

    # get the becke grid points
    coords = inp.sSCF.grids.coords
    weights = inp.sSCF.grids.weights

    # get density of A
    print ('Calculating density of subsystem A')
    rhoA = get_density(inp.cell[0], coords, inp.kpts, inp.Dmat[0])
    print ('Nelec A ', (rhoA*weights).sum())

    # get density of B
    print ('Calculating density of subsystem B')
    rhoB = get_density(inp.cell[1], coords, inp.kpts, inp.Dmat[1])
    print ('Nelec B ', (rhoB*weights).sum())

    if os.path.isfile(inp.save+'.sup'):

        # get supermolecular density
        print ('Calculating density of supermolecular system')
        inp.Dsup = pickle.load(open(inp.save+'.sup', 'r'))
        rhoS = get_density(inp.ccell, coords, inp.kpts, inp.Dsup)
        print ('Nelec S ', (rhoS*weights).sum())

        # print absolute density difference
        rhoabs = np.abs(rhoA + rhoB - rhoS)
        print ('Absolute density difference ', (rhoabs*weights).sum())

    if os.path.isfile(inp.save+'.dmA'):
        dmA = pickle.load(open(inp.save+'.dmA', 'r'))
        mol = inp.cell[0].to_mol()
        AO = pyscf.dft.numint.eval_ao(mol, coords)
        rhoFA = np.einsum('ca,cb,ab->c', AO, AO.conjugate(), dmA)

        plt.plot(x, rhoFA, 'm-', lw=2, label='finite A')

    if os.path.isfile(inp.save+'.dmB'):
        dmB = pickle.load(open(inp.save+'.dmB', 'r'))
        AO = pyscf.pbc.dft.numint.eval_ao_kpts(inp.cell[1], coords, kpts=inp.kpts)
        AO = np.array(AO)
        rhoPB = np.einsum('kca,kcb,kab->c', AO, AO.conjugate(), dmB).real / nkpts

        plt.plot(x, rhoPB, 'c-', lw=2, label='periodic B')
        plt.plot(x, rhoFA + rhoPB, 'y--', lw=2, label='sum')
 
def read_input(filename):
    '''Reads a formatted input file.'''

    from input_reader import InputReader
    import sys
    from pyscf import gto, dft, pbc
    from pyscf.pbc import gto as pbcgto, dft as pbcdft, df as pbcdf, scf as pbcscf
    import numpy as np
    from .mole import concatenate_cells

    # initialize reader for a pySCF input
    reader = InputReader(comment=['!', '#', '::', '//'],
             case=False, ignoreunknown=True)

    # add finite subsystem block
    subsystem = reader.add_block_key('subsystem', required=True, repeat=True)
    subsystem.add_regex_line('atom',
        '\s*([A-Za-z.]+)\s+(\-?\d+\.?\d*)\s+(\-?\d+.?\d*)\s+(\-?\d+.?\d*)', repeat=True)
#    subsystem.add_line_key('charge', type=int, default=0)
    subsystem.add_line_key('spin', type=int)
    subsystem.add_line_key('basis')
    subsystem.add_line_key('unit', type=('angstrom','a','bohr','b'))

    # lattice vectors in periodic block
    lattice = reader.add_block_key('lattice', required=True, repeat=False)
    lattice.add_regex_line('a', '\s*(\-?\d+\.?\d*)\s+(\-?\d+\.?\d*)\s+(\-?\d+\.?\d*)',
                           repeat=True)

    # add embedding block
    embed = reader.add_block_key('embed', required=True)
    embed.add_line_key('cycles', type=int, default=10)          # max freeze-and-thaw cycles
    embed.add_line_key('conv', type=float, default=1e-4)        # f&t conv tolerance
    embed.add_line_key('method', type=str)                      # embedding method
    embed.add_boolean_key('localize')                           # whether to localize orbitals
    embed.add_boolean_key('freezeb')                            # optimize only subsystem A
    embed.add_line_key('subcycles', type=int, default=1)        # number of subsys diagonalizations
    operator = embed.add_mutually_exclusive_group(dest='operator', required=True)
    operator.add_line_key('mu', type=float, default=1e6)        # manby operator by mu
    operator.add_boolean_key('manby', action=1e6)               # manby operator
    operator.add_boolean_key('huzinaga', action='huzinaga')     # huzinaga operator
    operator.add_boolean_key('thomasfermi', action='tf')        # thomas-fermi KEDF
    operator.add_boolean_key('tf', action='tf')                 # thomas-fermi KEDF
    operator.add_boolean_key('hm', action='hm')                 # modified huzinaga

    # add more complex line keys
    kgroup = reader.add_mutually_exclusive_group(dest='kgroup', required=True)
    kgroup.add_line_key('kpoints', type=[int, int, int])
    kscaled = kgroup.add_block_key('kgrid')
    kscaled.add_regex_line('kpoints', '\s*(\-?\d+\.?\d*)\s+(\-?\d+\.?\d*)\s+(\-?\d+\.?\d*)',
                           repeat=True)
#    reader.add_line_key('kpoints', type=[int, int, int], required=True)
    reader.add_line_key('gs', type=[int, int, int], required=True)
    reader.add_line_key('dimension', type=(0,1,2,3), required=True)

    # add simple line keys
    reader.add_line_key('basis', default='sto-3g')
    reader.add_line_key('unit', type=('angstrom','a','bohr','b'), default='a')
    reader.add_line_key('method', default='lda,vwn')
    reader.add_line_key('verbose', type=int, default=0)
    reader.add_line_key('conv', type=float, default=1e-6)
    reader.add_line_key('memory', type=(int, float))
    reader.add_line_key('fit', type=('df', 'mdf', 'pwdf', 'fftdf'), default='mdf')
    reader.add_line_key('diis', type=int, default=1)            # DIIS cycle (0 to turn off)
    reader.add_line_key('smear', type=float, default=None)      # Fermi smearing
    reader.add_line_key('maxiter', type=int, default=50)        # max iteration for supermolecular calcs
    reader.add_line_key('grid', type=int, default=2)            # Becke grid level
    reader.add_line_key('save', type=str)                       # save dmats
    reader.add_line_key('read', type=str)                       # read dmats

    # add boolean keys
    reader.add_boolean_key('finite')                            # do finite embedding

    # read the input file
    inp = reader.read_input(filename)
    inp.filename = filename

    # unit conversion
    inp.conversion = 1.
    if inp.unit in ('angstrom', 'a'): inp.conversion = 1.8897261328856432

    # sanity checks
#    if not (inp.subsystem.charge==0 and inp.periodic.charge==0):
#        sys.exit("Periodic cell / system is charged!")
    if len(inp.lattice.a) != 3: sys.exit("Must provide THREE lattice vectors!")
    if len(inp.subsystem) != 2: sys.exit("Only TWO subsystems can be used right now!")

    # some defaults
    if inp.embed.method is None: inp.embed.method = inp.method
    if inp.embed.method in ('ccsd', 'ccsd(t)'): inp.embed.method = 'hf'

    # initialize pySCF cell objects
    inp.nsub = len(inp.subsystem)
    inp.cell = [None for c in range(inp.nsub)]
    for c in range(inp.nsub):
        cell = pbcgto.Cell()

        # collect atoms
        cell.atom = []
        ghbasis = []
        for r in inp.subsystem[c].atom:
            if 'ghost.' in r.group(1).lower() or 'gh.' in r.group(1).lower():
                ghbasis.append(r.group(1).split('.')[1])
                rgrp1 = 'ghost:{0}'.format(len(ghbasis))
                cell.atom.append([rgrp1, (float(r.group(2)), float(r.group(3)), float(r.group(4)))])
            else:
                cell.atom.append([r.group(1), (float(r.group(2)), float(r.group(3)), float(r.group(4)))])

        # build dict of basis for each atom
        cell.basis = {}
        nghost = 0
        subbas = [inp.subsystem[c].basis if inp.subsystem[c].basis else inp.basis][0]
        for i in range(len(cell.atom)):
            if 'ghost' in cell.atom[i][0]:
                cell.basis.update({cell.atom[i][0]: pbcgto.basis.load(subbas, ghbasis[nghost])})
                nghost += 1
            else:
                cell.basis.update({cell.atom[i][0]: pbcgto.basis.load(subbas, cell.atom[i][0])})

        # use local values first, then global values, if they exist
        if inp.memory is not None: cell.max_memory = inp.memory
        if inp.subsystem[c].unit is not None:
            cell.unit = inp.subsystem[c].unit
        elif inp.unit is not None:
            cell.unit = inp.unit
        if inp.subsystem[c].spin is not None: cell.spin = inp.subsystem[c].spin
        cell.verbose = inp.verbose

        # lattice vectors
        a = inp.lattice.a
        a = np.array([[a[i].group(1), a[i].group(2), a[i].group(3)]
                      for i in range(len(a))], dtype=float)
        cell.a = inp.conversion * a

        # cell grid points
        cell.mesh = np.array(inp.mesh, dtype=int)

        # cell dimension
        cell.dimension = inp.dimension

        # build cell object
        cell.build(dump_input=False)
        inp.cell[c] = cell

    # generate concatenated cell object (without ghost)
    ccell = concatenate_cells(inp.cell[0], inp.cell[1], ghost=False)
    for i in range(2, inp.nsub):
        ccell = concatenate_mols(ccell, inp.cell[i], ghost=False)
    inp.ccell = ccell

    # make kpts
    if inp.kgroup.__class__ is tuple:
        inp.kpts = inp.ccell.make_kpts(inp.kgroup)
        inp.nkpts = len(inp.kpts)
    else:
        kabs = np.array([r.group(0).split() for r in inp.kgroup.kpoints], dtype=float)
        inp.kpts = inp.ccell.get_abs_kpts(kabs)
        inp.nkpts = len(inp.kpts)

    # make periodic SCF objects
    SCF = pbcdft.KRKS
    ldft = True
    if inp.embed.method.lower() in ('hf', 'hartree-fock'):
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

    inp.sSCF.with_df = inp.DF(inp.ccell)
    inp.sSCF.with_df.kpts = inp.kpts

    for i in range(inp.nsub):
        inp.cSCF[i].with_df = inp.DF(inp.cell[i])
        inp.cSCF[i].with_df.kpts = inp.kpts

    # return
    return inp

def get_density(cell, coords, kpts, dmat):
    import pyscf
    AO = pyscf.pbc.dft.numint.eval_ao_kpts(cell, coords, kpts=kpts)
    AO = np.array(AO)
    nk = AO.shape[0]
    nA = AO.shape[1]
    nc = len(coords)
    rho = np.zeros((nc))
    for k in range(nk):
        aok = AO[k]
        aoc = aok.conjugate().transpose()
        dm = dmat[k]
        temp = np.dot(dm, aoc)
        temp = temp.transpose()
        rho += np.multiply(temp, aok).real.sum(-1)
    return rho / float(nk)

def plot_coords(plt, coords, charges, size):
    for i in range(len(coords)):
        x = coords[i]
        if charges[i] > 1.:
            ms = float(size) * charges[i] / 5.
        else:
            ms = 0.
        plt.plot(x[0],x[1],'ko',markersize=ms, markerfacecolor='none')
    return plt

if __name__ == '__main__':
    '''Main Program.'''

    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from textwrap import dedent
    import sys 

    # parse input files
    parser = ArgumentParser()#description=dedent(main.__doc__),
#                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('input_files', nargs='*', default=sys.stdin,
                        help='The input files to submit.')
    parser.add_argument('-x', type=float)
    parser.add_argument('-y', type=float)
    parser.add_argument('-z', type=float)
    args = parser.parse_args()

#    if args.x is None and args.y is None and args.z is None:
#        print ("At least one of x, y or z must be none!")
#        sys.exit()

    for filename in args.input_files:
        main(filename)
