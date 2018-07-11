#! /usr/bin/env python
from __future__ import print_function, division

import numpy as np
import pickle
from read_input import read_input

def main(filename, x=None, y=None, z=None, sep=0.01):

    import pyscf
    from matplotlib import pyplot as plt
    import os
    from matplotlib.colors import LogNorm
    import matplotlib.colors as colors
    from matplotlib import cm
    from .intor import intor, k2real, real2k
    import chem

    # get input
    inp     = read_input(filename, build=False)
    inp.nao = np.array([inp.cell[i].nao_nr() for i in range(inp.nsub)])
    nao     = inp.sSCF.cell.nao_nr()

    # get kpoints
    inp.band = read_band(inp.filename)
    kpts1 = np.copy(inp.kpts)
    print ('Initial kpts: ', repr(kpts1))
    B = inp.sSCF.cell.reciprocal_vectors()
    kpts2 = np.dot(inp.band.kpts, B)
    print ('Final kpts: ' , repr(kpts2))

    pi2 = np.pi * 2.

    # get real-space lattice vectors
    Ls = inp.sSCF.cell.get_lattice_Ls()
    print ('Ls: ', repr(Ls))

    # get overlap matrix in real-space lattice vectors
    Sm_T = np.zeros((len(Ls), nao, nao))
    mA = inp.sSCF.cell.to_mol()
    for i in range(len(Ls)):
        mB = mA.copy()
        atm = []
        for j in range(mA.natm):
            coord = mA.atom_coord(j) * 0.52917720859 - Ls[i] * 0.52917720859
            atm.append([mA.atom_symbol(j), coord])
        mB.atom = atm
        mB.unit = 'A'
        mB.build(dump_input=False)

        Sm_T[i] = intor(mA, mB, 'cint1e_ovlp_sph')

    # read fock matrices from file
    if inp.read and os.path.isfile(inp.read+'.fock'):
        print ('Reading supermolecular Fock matrix from file: '+inp.read+'.fock')
        FS = pickle.load(open(inp.read+'.fock', 'r'))

        print ('Reading supermolecular Smat matrix from file: '+inp.read+'.smat')
        Smat = pickle.load(open(inp.read+'.smat', 'r'))

        E_k1 = diagonalize(FS, Smat)

        x = np.arange(len(E_k1)) / len(E_k1)
        E_k1 = E_k1.transpose()
        for i in range(len(E_k1)):
            plt.plot(x,E_k1[i],'b-')

        # transform into real space
        FS_T = k2real(FS, kpts1, Ls)
        Sm_T2 = k2real(Smat, kpts1, Ls)
        print ('SMAT DIFF: ', np.abs(Sm_T - Sm_T2).max())

        # transform back into k-space
        FS_k = real2k(FS_T, kpts2, Ls)
        Sm_k = real2k(Sm_T, kpts2, Ls)

#        print (repr(FS.reshape(len(kpts1), nao, nao)[0]))
#        print (FS_k[0])
        print (np.abs(FS.reshape(len(kpts1), nao, nao)[0] - FS_k[0]).max())

        E_k2 = diagonalize(FS_k, Sm_k, kpts2)

        x = np.arange(len(E_k2)) / len(E_k2)
        E_k2 = E_k2.transpose()
        for i in range(len(E_k2)):
            plt.plot(x,E_k2[i],'r-')

        plt.show()

def diagonalize(Fock, Smat, kpts=None):
    '''Diagonalize k-points fock matrix and returns matrix of E vectors.'''

    import scipy as sp

    nk = Fock.shape[0]
    na = Fock.shape[1]

    E = np.zeros((nk, na))
    for k in range(nk):
        if kpts is not None: print (kpts[k])
        E[k], C = sp.linalg.eigh(Fock[k], Smat[k])

    return E

def read_band(filename):
    '''Reads a formatted input file.'''

    from input_reader import InputReader
    import sys
    from pyscf import gto, dft, pbc
    from pyscf.pbc import gto as pbcgto, dft as pbcdft, df as pbcdf, scf as pbcscf
    import numpy as np
    from mole import concatenate_cells

    # initialize reader for a pySCF input
    reader = InputReader(comment=['!', '#', '::', '//'],
             case=False, ignoreunknown=True)

    # define "band" block
    rband = reader.add_block_key('band', required=True)
    rband.add_line_key('npoints', type=int, required=True)
    rband.add_regex_line('points',
        '\s*(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)', repeat=True)

    # read the input file
    inp = reader.read_input(filename)
    inp.nskpts = len(inp.band.points)

    # get each special k-point
    skpts = np.zeros((inp.nskpts,3))
    for i in range(inp.nskpts):
        skpts[i][0] = inp.band.points[i].group(1)
        skpts[i][1] = inp.band.points[i].group(2)
        skpts[i][2] = inp.band.points[i].group(3)

    # get distance between spcial kpoints
    dkpts = np.zeros((inp.nskpts-1))
    for i in range(inp.nskpts-1):
        temp = skpts[i+1] - skpts[i]
        dkpts[i] = np.sqrt(np.dot(temp,temp))

    # kpoints spacing
    kspace = dkpts.sum() / float(inp.band.npoints)

    # get kpoint coordinates
    x = np.array([], dtype=float)
    for i in range(inp.nskpts-1):
        vec  = skpts[i+1] - skpts[i]
        lvec = np.sqrt(np.dot(vec,vec))
        temp = np.arange(0, lvec, kspace) / lvec
        temp = np.outer(temp, vec) + skpts[i]
        x = np.append(x, temp.flatten())
    x = np.array(x).flatten()
    lx = len(x)
    x = x.reshape(int(lx/3.),3)
    if not (x[-1] == skpts[-1]).all():
        x = np.append(x, [skpts[-1]], axis=0)

    # replace all 1's with zeros
    ix = np.where(x == 1.)
    x[ix] = 0.

    inp.kpts = np.copy(x)

    return inp

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
    args = parser.parse_args()

    for filename in args.input_files:
        main(filename)
