#! /usr/bin/env python
from __future__ import print_function, division
import numpy as np
import pickle

def main(filename, x=None, y=None, z=None, sep=0.01):
    '''Calculates the number of displaced electrons,
    and plots the density difference contour maps
    for periodic-in-periodic and cluster-in-periodic
    embedding.
    Requires the *filename*.hdf5 file to be saved.'''

    import pyscf
    from .read_input import read_input
    from matplotlib import pyplot as plt
    import os
    from matplotlib.colors import LogNorm
    import matplotlib.colors as colors
    from matplotlib import cm
    import h5py

    # whether to plot contour maps
    lcontour = True

    # get input
    inp        = read_input(filename, build=False)
    inp.nao    = np.array([inp.cell[i].nao_nr() for i in range(inp.nsub)])
    nkpts = float(inp.nkpts)

    # open HDF5 file
    if inp.filename[-4:] == '.inp':
        h5pyname = inp.filename[:-4]+'.hdf5'
    else:
        h5pyname = inp.filename+'.hdf5'
    inp.h5py = h5py.File(h5pyname)

    # read densityies from file
    inp.Dmat = [None for i in range(inp.nsub)]
    for i in range(inp.nsub):
        inp.Dmat[i] = inp.h5py['{0}/dmat'.format(i)][...]

    # generate the grid points
    A2B = 1.8897261328856432
    a = inp.ccell.lattice_vectors().diagonal() / A2B
    spacingx = 0.1
    spacingy = 0.1
    spacingz = 0.1
    x = np.arange(0, a[0]+1e-6, spacingx)
    y = np.arange(0, a[1]+1e-6, spacingy)
    z = np.arange(0, a[2]+1e-6, spacingz)

    print ('length x, y, z: ', len(x), len(y), len(z))
    coords = np.array(np.meshgrid(x,y,z)).swapaxes(0,2).swapaxes(2,3)
    coords = coords.reshape(len(x)*len(y)*len(z),3) * A2B
    iux = A2B * ((x[-1]-x[0])/len(x))
    iuy = A2B * ((y[-1]-x[0])/len(y))
    iuz = A2B * ((z[-1]-z[0])/len(z))
    vol = iux * iuy * iuz / ( A2B * A2B * A2B )
    print ('grid size: ', iux, iuy, iuz)
    print ('grid volume (A^3): ', vol)

    print ('Calculating density of subsystem A') ##############################
    rhoA = get_2d_density(inp.cell[0], coords, inp.kpts, inp.Dmat[0])
    rhoA = rhoA.reshape(len(x),len(y),len(z))
    rA = np.log10(rhoA.sum(axis=2).transpose() * iuz)
    lA = np.linspace(rA.min(), rA.max(), 501)

    # subsystem charges and coordinates
    acharges = inp.ccell.atom_charges()
    acoords = inp.ccell.atom_coords() / A2B
    Acharges = inp.cell[0].atom_charges()
    Acoords = inp.cell[0].atom_coords() / A2B
    Bcharges = inp.cell[1].atom_charges()
    Bcoords = inp.cell[1].atom_coords() / A2B

    # integrate again
    rA = rhoA.sum(axis=2).sum(axis=1) * iuy * iuz

    print ('Calculating density of subsystem B') ##############################
    rhoB = get_2d_density(inp.cell[1], coords, inp.kpts, inp.Dmat[1])
    rhoB = rhoB.reshape(len(x),len(y),len(z))
    rB = np.log10(rhoB.sum(axis=2).transpose() * iuz)
    lB = np.linspace(rB.min(), rB.max(), 501)

    rB = rhoB.sum(axis=2).sum(axis=1) * iuy * iuz

    # Read supermolecular density from HDF5 ###################################
    print ('Calculating density of supermolecular system')
    inp.Dsup = inp.h5py['dsup'][...]
    rhoS = get_2d_density(inp.ccell, coords, inp.kpts, inp.Dsup)
    rhoS = rhoS.reshape(len(x),len(y),len(z))
    rS = np.log10(rhoS.sum(axis=2).transpose() * iuz)
    lS = np.linspace(rS.min(), rS.max(), 501)

    rS = rhoS.sum(axis=2).sum(axis=1) * iuy * iuz

    # Density difference ######################################################
    rDiff = rhoA + rhoB - rhoS
    delta_abs = np.abs(rDiff).sum() * iux * iuy * iuz
    nelec = rhoS.sum() * iux * iuy * iuz
    print ('Nelectron: ', nelec)
    print ('Absolute density difference (e): ', delta_abs)
    print ('Number of displaced electrons: ', delta_abs / 2.)
    print ('Absolute density difference per electron: ', delta_abs / nelec )

    # Contour separation ######################################################
    rD = rDiff.sum(axis=2).transpose() * iuz
    rceil = np.log10(max(abs(rD.min()), abs(rD.max())))
    rmax = 10**rceil
    if rmax > 1e-3:
        rmax = np.round(np.round(rmax, 3) + 4e-3, 2)
    else:
        rmax = np.round(np.round(rmax, 7) + 4e-7, 6)
    lD = np.linspace(-rmax, rmax, 501)

    # Periodic-in-periodic difference contour #################################
    if lcontour:
        lhi = np.arange(-6,-0.9,1)
        lhi = 10**lhi
        llo = np.arange(-1,-6.1,-1)
        llo = -10**llo
        lvls = np.append(llo, np.array([0]))
        lvls = np.append(lvls, lhi)
        print (lvls)
#        lvls = np.array([-1,-1e-1,-1e-2,-1e-3,-1e-4,-1e-5,-1e-6,-1e-7,
#                         0,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1])
        plt.contourf(x,y,rD,levels=lvls,cmap=cm.Spectral,extend='both',
                norm=colors.SymLogNorm(linthresh=1e-6,linscale=1e-6,vmin=-0.1,vmax=0.1))
        plt = plot_coords_sub(plt, Acoords, Acharges, Bcoords, Bcharges, 10)
#        cticks = np.array([-1,-1e-1,-1e-2,-1e-3,-1e-4,-1e-5,-1e-6,
#                            0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1])
        cticks = np.array([-1e-1,-1e-2,-1e-3,-1e-4,-1e-5,-1e-6,0,
                            1e-6,1e-5,1e-4,1e-3,1e-2,1e-1])
        cbar = plt.colorbar()
        cbar.set_ticks(cticks)
        plt.xlabel('x / \AA')
        plt.ylabel('y / \AA')
        plt.title('$\Delta \\rho$ / $ea_0^{-2}$')
        plt.show()

    if 'dmA' in inp.h5py: #####################################################
        print ('Calculating density of finite A')
        dmA = inp.h5py['dmA'][...]
        mol = inp.cell[0].to_mol()
        tempdmA = np.zeros((len(inp.kpts), dmA.shape[0], dmA.shape[1]))
        for nk in range(len(inp.kpts)):
            tempdmA[nk] = dmA
        rhoFA = get_2d_density(inp.cell[0], coords, inp.kpts, tempdmA)
        rhoFA = rhoFA.reshape(len(x),len(y),len(z))
        rFA = np.log10(rhoFA.sum(axis=2).transpose() * iuz)
        lFA = np.linspace(rFA.min(), rFA.max(), 501)

        rFA = rhoFA.sum(axis=2).sum(axis=1) * iuy * iuz

        print ('Calculating density of periodic B') ###########################
        dmB = inp.Dmat[1]
        rhoFB = get_2d_density(inp.cell[1], coords, inp.kpts, dmB)
        rhoFB = rhoFB.reshape(len(x),len(y),len(z))
        rFB = np.log10(rhoFB.sum(axis=2).transpose() * iuz)
        lFB = np.linspace(rFB.min(), rFB.max(), 501)

        rFB = rhoFB.sum(axis=2).sum(axis=1) * iuy * iuz

        print ('Total finite electron A in cell : ', rhoFA.sum() * iux * iuy * iuz )
        print ('Total electrons in cell : ', rhoS.sum() * iux * iuy *iuz )
        rDiff = rhoFA + rhoFB - rhoS
        delta_abs = np.abs(rDiff).sum() * iux * iuy * iuz
        print ('Absolute density difference (e): ', delta_abs)
        print ('Number of displaced electrons: ' , delta_abs / 2.)
        print ('Absolute density difference per electron: ', delta_abs / nelec)
        rD = rDiff.sum(axis=2).transpose() * iuz

        if lcontour:
            plt.contourf(x,y,rD,levels=lvls,cmap=cm.Spectral,
                norm=colors.SymLogNorm(linthresh=1e-7,linscale=1e-7,vmin=-0.1,vmax=0.1))
            plt = plot_coords(plt, acoords, acharges, 10)
#            plt = plot_coords_sub(plt, Acoords, Acharges, Bcoords, Bcharges, 10)
            cbar = plt.colorbar()
            cbar.set_ticks(cticks)
            plt.xlabel('x / \AA')
            plt.ylabel('y / \AA')
            plt.title('$|\Delta \\rho|$ / $ea_0^{-2}$')
            plt.show()

#            rD = rhoFA - rhoA
#            rD = np.abs(rD).sum(axis=2).transpose() * iuz
#            plt.contourf(x,y,rD,levels=lvls,cmap=cm.Spectral,
#                norm=colors.SymLogNorm(linthresh=1e-7,linscale=1e-7,vmin=-0.1,vmax=0.1))
#             plt = plot_coords_sub(plt, Acoords, Acharges, Bcoords, Bcharges, 10)
#            cbar = plt.colorbar()
#            cbar.set_ticks(cticks)
#            plt.xlabel('x / \AA')
#            plt.ylabel('y / \AA')
#            plt.title('$|\Delta \\rho|$ / $ea_0^{-2}$')
#            plt.show()

##        plt.plot(x, rS, 'k-', label='KS-DFT')
#        plt.plot(x, rA, color='#1fb45aff', label='periodic A')
#        plt.plot(x, rFA, color='#ff7f0eff', ls='--', label='cluster A')
#        plt.plot(x, rB, color='#1f77b4ff', label='periodic B')
##        plt.plot(x, rFB, 'b-', label='cluster B')
##        plt.plot(x, rFA+rFB, 'm-', label='sum cluster A+B')
#
##        plt.plot(x, rFA - rA, 'r-', label='diff A')
##        plt.plot(x, rFB - rB, 'b-', label='diff B')
##        plt.plot(x, rFA+rFB-rS, 'k-', label='diff total')
#        plt.legend()
#        plt.show()

#    plt.plot(x, rS, 'k-', label='KS-DFT')
#    plt.plot(x, rA, color='#1fb45aff', label='subsys. A')
#    plt.plot(x, rB, color='#1f77b4ff', label='subsys. B')
#    plt.plot(x, rA+rB, color='#ff7f0eff', ls='--', label='sum A+B')
#    plt.legend()
#    plt.show()

#    X = [x, rS, rA, rB]
#    pickle.dump(X, open('density.pickle', 'wb'))


def get_2d_density(cell, coords, kpts, dmat):
    '''Gets the density on a set of coordinates.'''

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
    '''Places a black circular marker at the location of the atoms.'''

    for i in range(len(coords)):
        x = coords[i]
        if charges[i] <= 2:
            ms = size * 0.7
        elif charges[i] <= 10:
            ms = size * 1.0
        elif charges[i] <= 18:
            ms = size * 1.3
        else:
            ms = size * 1.6
        plt.plot(x[0],x[1],'ko',markersize=ms, markerfacecolor='none')
    return plt


def plot_coords_sub(plt, Acoords, Acharges, Bcoords, Bcharges, size):
    '''Places a circular marker at the location of the atoms.
    Uses red for subsystem A and blue for subsystem B.'''

    for i in range(len(Acoords)):
        x = Acoords[i]
        if Acharges[i] <= 0.:
            continue
        elif Acharges[i] <= 2:
            ms = size * 0.7 
        elif Acharges[i] <= 10: 
            ms = size * 1.0 
        elif Acharges[i] <= 18: 
            ms = size * 1.3 
        else:
            ms = size * 1.6 
        plt.plot(x[0],x[1],'ro',alpha=0.6,markersize=ms, markerfacecolor='none')

    for i in range(len(Bcoords)):
        x = Bcoords[i]
        if Bcharges[i] <= 0.:
            continue
        elif Bcharges[i] <= 2:
            ms = size * 0.7 
        elif Bcharges[i] <= 10: 
            ms = size * 1.0 
        elif Bcharges[i] <= 18: 
            ms = size * 1.3 
        else:
            ms = size * 1.6 
        plt.plot(x[0],x[1],'bo',alpha=0.6,markersize=ms, markerfacecolor='none')
    return plt


if __name__ == '__main__':

    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from textwrap import dedent
    import sys 

    parser = ArgumentParser(description=dedent(main.__doc__),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('input_files', nargs='*', default=sys.stdin,
                        help='The input files to submit.')
    parser.add_argument('-x', type=float)
    parser.add_argument('-y', type=float)
    parser.add_argument('-z', type=float)
    args = parser.parse_args()
    for filename in args.input_files:
        main(filename)
