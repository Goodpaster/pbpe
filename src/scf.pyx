import numpy as np
cimport numpy as np
import scipy as sp
from functools import reduce

FTYPE = np.float
ctypedef np.float_t FTYPE_t
CTYPE = np.complex
ctypedef np.complex_t CTYPE_t


def do_embedding(int A, inp, ldiag=True, llast=False):

    from pyscf.pbc import dft as pbcdft
    from pyscf import lib
    from .integrals import get_xc_pot
    from .integrals import get_2e_matrix
    import pickle
    from pyscf.scf.diis import CDIIS

    cdef int nk = inp.nkpts
    cdef int nA = inp.cSCF[A].cell.nao_nr()
    cdef int nS = inp.sSCF.cell.nao_nr()

    sub2sup = inp.sub2sup

    # initialize embedding matrix, and add core hamiltonian
    if A==0 or inp.embed.subcycles > 1:
        Fock = np.zeros((nk, nS, nS), dtype=complex)
        Fock += inp.hcore[...]

        # get the 2e contributions
        inp.timer.start('2e matrices')
        Fock += get_2e_matrix(inp)
        inp.timer.end('2e matrices')

        # apply mixing
        if inp.mix is not None and inp.ift>1 and inp.FOCK is not None:
            Fock = Fock * (1. - inp.mix) + inp.FOCK * inp.mix

        # use DIIS on total fock matrix
        if inp.embed.subcycles == 1:
            if inp.diis and inp.ift>=inp.diis:
                if inp.DIIS[A] is None:
                    inp.DIIS[A] = CDIIS()
                    inp.DIIS[A].space = 20
                # make supermolecular density matrix
                dm = np.zeros((nk, nS, nS), dtype=complex)
                for i in range(inp.nsub):
                    dm[np.ix_(range(nk), sub2sup[i], sub2sup[i])] += inp.Dmat[i][...]
                Fock = inp.DIIS[A].update(inp.Smat[...], dm, Fock)

        # save the total Fock matrix
        inp.FOCK = np.copy(Fock)

    # use saved Fock matrix for subsystem B (more stable and saves time)
    else:
        Fock = np.copy(inp.FOCK)

    # get B (2 subsystems only)
    if A==0: B = 1
    if A==1: B = 0

    SmatAA = inp.Smat[...][np.ix_(range(nk), sub2sup[A], sub2sup[A])]
    SmatAB = inp.Smat[...][np.ix_(range(nk), sub2sup[A], sub2sup[B])]
    SmatBA = inp.Smat[...][np.ix_(range(nk), sub2sup[B], sub2sup[A])]

    # initialize projection operator
    POp = np.zeros((nk, nA, nA), dtype=complex)

    # get manby projection operator
    if inp.embed.operator.__class__ in (int, float):
        inp.timer.start('mu operator')
        for k in range(nk):
            POp[k] = inp.embed.operator * np.dot(SmatAB[k], np.dot(inp.Dmat[B][k], SmatBA[k]))
        inp.timer.end('mu operator')

    # Huzinaga projection operator
    else:
        inp.timer.start('huzinaga operator')
        tF = Fock[np.ix_(range(nk), sub2sup[A], sub2sup[B])]

        # shift the orbital energies by the fermi energy
        if inp.huzfermi:
            efermi = max(inp.Fermi[0], inp.Fermi[1])
            tF -= SmatAB * efermi

        # add operator for each k-point
        for k in range(nk):
            FDS = np.dot(tF[k], np.dot(inp.Dmat[B][k], SmatBA[k]))
            SDF = FDS.transpose().conjugate()
            POp[k] = - 0.5 * ( FDS + SDF )
        inp.timer.end('huzinaga operator')

    # get the elements of the Fock matrix for this subsystem
    Fock = Fock[np.ix_(range(nk), sub2sup[A], sub2sup[A])]
    Hcore = inp.hcore[...][np.ix_(range(nk), sub2sup[A], sub2sup[A])]

    # add projection operator to fock matrix
    Fock += POp

    # Save this Fock matrix
    inp.Fock[A] = np.copy(Fock)
    if '{0}/fock'.format(A) in inp.h5py:
        inp.h5py['{0}/fock'.format(A)][...] = Fock
        Fock = inp.h5py['{0}/fock'.format(A)]
    else:
        Fock = inp.h5py.create_dataset('{0}/fock'.format(A), data=Fock)

    # update hcore of this subsystem (for correct energies)
    if llast:
        inp.timer.start('update hcore')
        inp.timer.start('2e matrices')
        if inp.cSCF[A].cell.a is None: inp.cSCF[A].cell.a = inp.lattice
        try:
            vA = inp.cSCF[A].get_veff(dm=inp.Dmat[A][...])
        except TypeError:
            vA = inp.cSCF[A].get_veff(dm_kpts=inp.Dmat[A][...])
        inp.timer.end('2e matrices')
        hcore = Fock[...] - vA
        if inp.lgamma: hcore = hcore[0]
        inp.cSCF[A].get_hcore = lambda *args: hcore
        inp.timer.end('update hcore')

    # shift virtual orbital energies
    if inp.shift and ldiag:
        for k in range(nk):
            Fock[...][k] += ( SmatAA[k] - np.dot(np.dot(SmatAA[k],
                    inp.Dmat[A][k] * 0.5), SmatAA[k]) ) * inp.shift

    # use DIIS on the fock matrix
    if inp.embed.subcycles > 1:
        if inp.diis and inp.ift>=inp.diis:
            if inp.DIIS[A] is None:
                inp.DIIS[A] = CDIIS()
                inp.DIIS[A].space = 30
            Fock[...] = inp.DIIS[A].update(SmatAA, inp.Dmat[A][...], Fock[...])

    # do diagonalization
    if ldiag:
        inp.timer.start('diagonalization')
        if A==0:
            smear = None
        else:
            smear = inp.smear

        Dnew, C, n, inp.Energy[A], inp.Fermi[A] = diagonalize(Fock[...],
            SmatAA, inp.kpts, inp.cSCF[A].cell.nelectron, sigma=smear)

        # mix in old Density matrix
        if inp.dmix is not None:
            Dnew = Dnew * (1. - inp.dmix) + inp.Dmat[A][...] * inp.dmix

        inp.timer.end('diagonalization')

        # save coefficient matrix to file
        if '{0}/cmat'.format(A) in inp.h5py:
            inp.h5py['{0}/cmat'.format(A)][...] = C
        else:
            C = inp.h5py.create_dataset('{0}/cmat'.format(A), data=C)

        # get energy
        energy = np.einsum('kab,kba', Fock[...], Dnew) / ( 2. * nk )
        energy += np.einsum('kab,kba', Hcore, Dnew) / ( 2. * nk )

        # remove energy shift
        if inp.shift:
            inp.Energy[A][inp.Energy[A]>inp.Fermi[A]] -= inp.shift

    else:
        return Fock[...]

    return Dnew, energy.real

def diagonalize(Fock, Smat, kpts, nelectron, sigma=None):

    # initialize some matrices
    cdef int k
    cdef int nkpts = len(kpts)
    cdef int nao = Fock.shape[1]
    cdef np.ndarray[FTYPE_t, ndim=2] e = np.zeros((nkpts, nao))
    cdef np.ndarray[CTYPE_t, ndim=3] c = np.zeros((nkpts, nao, nao), dtype=complex)

    # do diagonalizations of each k-point
    for k in range(nkpts):
        e[k], c[k] = sp.linalg.eigh(Fock[k], Smat[k])

    # get fermi energy (average of HOMO and LUMO)
    norbs = ( nelectron * nkpts ) // 2
    e_sorted = np.sort(e.ravel())
    fermi = ( e_sorted[norbs] + e_sorted[norbs-1] ) / 2.
    eorder = np.argsort(e, axis=None)

    # print warning for band gap
    if sigma is None:
        bandgap = abs(e_sorted[norbs] - e_sorted[norbs-1])
        if bandgap <= 0.001: print ('WARNING: Small band gap: {0:12.6f} a.u.'.format(bandgap))

    # get molecular occupation
    if sigma is None:
        mo_occ = np.zeros_like(e)
        mo_occ[e<fermi] = 2.
    else:
        mo_occ = ( e - fermi ) / sigma
        ie = np.where( mo_occ < 1000 )
        i0 = np.where( mo_occ >= 1000 )
        mo_occ[ie] = 2. / ( np.exp( mo_occ[ie] ) + 1. )
        mo_occ[i0] = 0.

        # print occupation
        print ('MO occupancy:')
        for i in range(max(0,norbs-4),min(norbs+4,nao*nkpts)):
            print ('{0:>3d}     {1:12.6f}     {2:12.6f}'.format(
                   i+1, e.flatten()[eorder][i], mo_occ.flatten()[eorder][i]))

    # get density matrix
    if nkpts == 1:
        dmat = np.zeros((nkpts, nao, nao), dtype=float)
    else:
        dmat = np.zeros((nkpts, nao, nao), dtype=complex)
    for k in range(nkpts):
        dmat[k] = np.dot(c[k] * mo_occ[k], c[k].transpose().conjugate())

    return dmat, c, mo_occ, e, fermi

def print_orbital_energies(np.ndarray[FTYPE_t, ndim=2] E, float fermi):

    cdef int nk = E.shape[0]
    cdef int no = E.shape[1]
    cdef int io, ik, ika, ikb, kb
    cdef float h2ev = 27.2113961
    Eev = E * h2ev

    print ("")
    print ("Crystalline Orbital Energies in eV (Fermi = {0:9.5f} eV):".format(fermi * h2ev))
    Eev = Eev.transpose()

    # figure out which bands to print (8 below and 3 above the fermi energy)
    idx = []
    Etemp = Eev - ( fermi * h2ev )
    Ediff = np.abs(Etemp).min(axis=1)
    a = np.argsort(Ediff)

    iocc = 0
    ivir = 0
    for i in range(len(a)):
        if (Etemp[a[i]] <= 0.).any():
            if iocc < 8:
                iocc += 1
                idx.append(a[i])
        else:
            if ivir < 5:
                ivir += 1
                idx.append(a[i])
    idx = np.array(idx)
    idx.sort()

    # print band energies
    print (repr(Eev[idx]))
    print ("")

def get_total_energy(inp, kpts, dsup=None, ldosup=True):
    '''Calculates the effective potential due to both density
    matrices A and B.'''

    from copy import deepcopy as copy

    cdef int nS = inp.sSCF.cell.nao_nr()
    cdef int nk = len(kpts)

    sub2sup = inp.sub2sup

    # make supermolecular density matrix
    if nk == 1:
        dm = np.zeros((nk, nS, nS), dtype=float)
    else:
        dm = np.zeros((nk, nS, nS), dtype=complex)
    for i in range(inp.nsub):
        dm[np.ix_(range(nk), sub2sup[i], sub2sup[i])] += inp.Dmat[i][...]

    # get effective potential
    inp.timer.start('2e matrices')
    try:
        V = inp.sSCF.get_veff(dm=dm, kpts=kpts)
    except TypeError:
        try:
            V = inp.sSCF.get_veff(dm_kpts=dm, kpts=kpts)
        except TypeError:
            V = inp.sSCF.get_veff(dm=dm)
    inp.timer.end('2e matrices')

    if inp.lgamma:
        etot = inp.sSCF.energy_tot(dm=dm[0])
    else:
        etot = inp.sSCF.energy_tot(dm=dm)

    # supermolecular DFT
    if ldosup:
        inp.timer.start('supermolecular calc.')
        if dsup is None: dsup = np.copy(dm)

        if 'energy-supermolecular' in inp.h5py:
            eold = inp.h5py['energy-supermolecular'][...]
        else:
            eold = np.einsum('kij,kji', inp.hcore[...], dm) / float(nk)
            eold += np.einsum('kij,kji', V, dm) / ( 2. * float(nk) )
        inp.Esup, inp.MOsup, inp.Dsup = do_supermol_scf(inp, inp.sSCF, dsup,
                              inp.kpts, hcore=inp.hcore[...], smat=inp.Smat[...],
                              nmax=inp.supcycles, conv=inp.conv, eold=eold,
                              sigma=inp.smear, lgamma=inp.lgamma)
        inp.timer.end('supermolecular calc.')

    return etot

def do_supermol_scf(inp, mf, dm, kpts, hcore=None, smat=None, nmax=50, eold=None,
                    conv=1e-8, sigma=None, lgamma=False):
    '''Do the supermolecular SCF.'''

    from pyscf import lib
    from .pstr import pstr
    import pickle

    cdef int nk = len(kpts)
    cdef int nao = dm.shape[1]
    cdef int nelectron = mf.cell.nelectron
    cdef float bandgap = 0.
    cdef np.ndarray[FTYPE_t, ndim=2] e = np.zeros((nk, nao))
    cdef np.ndarray[CTYPE_t, ndim=3] c = np.zeros((nk, nao, nao), dtype=complex)
    DIIS = lib.diis.DIIS()
    DIIS.space = 10

    # convergence of the density
    dconv = np.sqrt(conv)

    # get hcore
    if hcore is None: hcore = mf.get_hcore(kpts=kpts)
    if smat is None: smat = mf.get_ovlp(kpts=kpts)
    v_last = 0
    dm_last = 0

    # set some initial values
    iCyc = 0
    err  = 1.
    erd  = 1.
    fermi = 0.
    dconv = np.sqrt(conv)

    while (err > conv or erd > dconv) and iCyc < nmax:

        iCyc += 1

        # get 2e potentials
        inp.timer.start('2e matrices')
        try:
            veff = mf.get_veff(dm=dm, kpts=kpts, dm_last=dm_last, vhf_last=v_last)
        except TypeError:
            try:
                veff = mf.get_veff(dm_kpts=dm, kpts=kpts, dm_last=dm_last, vhf_last=v_last)
            except TypeError:
                veff = mf.get_veff(dm=dm, dm_last=dm_last, vhf_last=v_last)
        inp.timer.end('2e matrices')

        fock = hcore + veff

        # mix fock matrix with previous iteration
        if inp.mix is not None and iCyc > 1:
            fock = fock * (1. - inp.mix) + oldfock * inp.mix

        # shift virtual orbital energies
        if inp.shift:
            for k in range(nk):
                temp = ( smat[k] - np.dot(np.dot(smat[k], dm[k] * 0.5),
                                          smat[k]) ) * inp.shift
                if fock.dtype == float:
                    fock[k] += temp.real
                else:
                    fock[k] += temp

        # update fock matrix with DIIS
        if inp.diis > 0 and iCyc >= inp.diis:
            fock = DIIS.update(fock)

        # save a copy of the fock matrix
        oldfock = np.copy(fock)

        # cycle over k-points and diagonalize
        inp.timer.start('diagonalization')
        for k in range(nk):
            e[k], c[k] = sp.linalg.eigh(fock[k], smat[k])
        inp.timer.end('diagonalization')

        # get fermi energy
        norbs = ( nelectron * nk ) // 2
        e_sorted = np.sort(e.ravel())
        if e_sorted[norbs-1] < 0. and e_sorted[norbs] > 0. and sigma is None:
            fermi = 0.
        else:
            fermi = ( e_sorted[norbs] + e_sorted[norbs-1] ) / 2.

        # print warning for band gap
        if sigma is None:
            bandgap = abs(e_sorted[norbs] - e_sorted[norbs-1])
            if bandgap <= 0.001:
                print ('WARNING: Small band gap: {0:12.6f} a.u.'.format(bandgap))
        eorder = np.argsort(e, axis=None)

        # get molecular occupation
        if sigma is None:
            mo_occ = np.zeros_like(e)
            mo_occ[e<fermi] = 2.
        else:
            mo_occ = ( e - fermi ) / sigma
            ie = np.where( mo_occ < 25 )
            i0 = np.where( mo_occ >= 25 )
            mo_occ[ie] = 2. / ( np.exp( mo_occ[ie] ) + 1. )
            mo_occ[i0] = 0.

            # print occupation
            print ('MO occupancy:')
            for i in range(max(0,norbs-4),min(norbs+4,nao*nk)):
                print ('{0:>3d}     {1:12.6f}     {2:12.6f}'.format(
                       i+1, e.flatten()[eorder][i], mo_occ.flatten()[eorder][i]))

        # get density matrix
        if nk == 1:
            dmat = np.zeros((nk, nao, nao), dtype=float)
        else:
            dmat = np.zeros((nk, nao, nao), dtype=complex)
        for k in range(nk):
            dmat[k] = np.dot( c[k] * mo_occ[k], c[k].transpose().conjugate())

        # get errors
        enew = np.einsum('kij,kji', hcore, dmat) / float(nk)
        enew += np.einsum('kij,kji', veff, dmat) / ( 2. * float(nk) )
        erd = 0.0
        for ik in range(nk):
            erd += sp.linalg.norm(dm[ik] - dmat[ik])
        if eold is None:
            err = np.abs(enew)
        else:
            err = np.abs(eold - enew)
        print ('iter: {0:<3d}     |dE|: {1:16.12f}     |ddm|: {2:16.12f}'.format(
               iCyc, err, erd))

        # save matrices to file
        if 'fock' in inp.h5py:
            inp.h5py['fock'][...] = fock

        # copy new veff and dm
        v_last = np.copy(veff)
        dm_last = np.copy(dm)
        dm = np.copy(dmat)
        eold = np.copy(enew)

    # save energy
    if 'energy-supermolecular' in inp.h5py:
        inp.h5py['energy-supermolecular'][...] = eold
    else:
        inp.h5py.create_dataset('energy-supermolecular', data=eold)

    # remove energy shift
    if inp.shift:
        e[e>fermi] -= inp.shift

    print_orbital_energies(e, fermi)

    if err <= conv:
        pstr ("SCF converged after {0} cycles".format(iCyc), delim='!', addline=False)
    else:
        pstr ("SCF NOT converged", delim='!', addline=False)

    # get energy
    inp.timer.start('2e matrices')
    try:
        veff = mf.get_veff(dm=dm, kpts=kpts)
    except TypeError:
        try:
            veff = mf.get_veff(dm_kpts=dm, kpts=kpts)
        except TypeError:
            veff = mf.get_veff(dm=dm)
    inp.timer.end('2e matrices')

    if lgamma:
        esup = mf.energy_tot(dm=dm[0], h1e=hcore[0])
    else:
        esup = mf.energy_tot(dm=dm, h1e=hcore)

    # generate FCIDUMP file
    if inp.fcidump:
        fcidumpfile = inp.filename[:-4]+'.fcidump'
        fcidump(inp, fcidumpfile, inp.sSCF, c, inp.hcore, tol=1e-8)

    return esup, e, dm

def fcidump(inp, fn, mf, C, hcore, tol=1e-12):

    # get some variables
    nk = len(mf.kpts)
    nao = mf.cell.nao_nr()
    nmo = nk*nao
    nel = nk*mf.cell.nelectron
    enuc = mf.energy_nuc()

    # make float for gamma point only
    if nk == 1:
        dtype = float
    else:
        dtype = complex

    # get kpoints
    if inp.kgroup.__class__ is tuple:
        nprop = np.array(inp.kgroup)
    else:
        nprop = np.zeros((3), dtype=int)
        nprop[0] = nk
        nprop[1] = 1
        nprop[2] = 1

    # make h1e
    h1e = np.zeros((nmo,nmo), dtype=dtype)
    for ik in range(nk):
        rk = range(ik*nao,ik*nao+nao)
        temp = reduce(np.dot, (C[ik].T.conjugate(), hcore[ik], C[ik]))
        if nk == 1: temp = temp.real
        h1e[np.ix_(rk,rk)] = temp

    # make h2e
    h2e = np.zeros((nmo,nmo,nmo,nmo), dtype=dtype)
    if nk == 1:
        temp = mf.with_df.get_mo_eri(C[0])
        temp = temp.real
        i1 = -1
        for i in range(nao):
            for j in range(i,nao):
                i1 += 1
                i2 = -1
                for k in range(nao):
                    for l in range(nao):
                        i2 += 1
                        h2e[i][j][k][l] = temp[i1][i2]
    else:
        for ik in range(nk):
            rk = range(ik*nao,ik*nao+nao)
            temp = mf.with_df.get_mo_eri(C, kpts=inp.kpts[ik])
            temp = temp.reshape(nao,nao,nao,nao)
            h2e[np.ix_(rk,rk,rk,rk)] = temp

    # write header of FCIDUMP file
    fout = open(fn, 'w')
    fout.write('&FCI NORB={0},NELEC={1},MS2=0,\n'.format(nmo, nel))
    fout.write(' ORBSYM={0}\n'.format('1,'*nmo))
    fout.write(' ISYM=1\n')
    fout.write(' NPROP= {0} {1} {2}\n'.format(nprop[0], nprop[1], nprop[2]))
    fout.write(' PROPBITLEN= 15\n')
    fout.write('&END\n')

    # write 2MO integrals
    for i in range(nmo):
        for j in range(i,nmo):
            for k in range(nmo):
                for l in range(k,nmo):
                    if np.abs(h2e[i,j,k,l]) > tol:
                        if nk == 1:
                            fout.write('{0:15.8e} {1:3d} {2:3d} {3:3d} {4:3d}\n'.format(
                                h2e[i,j,k,l], i+1, j+1, k+1, l+1))
                        else:
                            fout.write('({0:15.8e},{1:15.8e}) {2:3d} {3:3d} {4:3d} {5:3d}\n'.format(
                                h2e[i,j,k,l].real, h2e[i,j,k,l].imag, i+1, j+1, k+1, l+1))

    # write 1MO integrals
    for i in range(nmo):
        for j in range(i,nmo):
            if np.abs(h1e[i,j]) > tol:
                if nk == 1:
                    fout.write('{0:15.8e} {1:3d} {2:3d} {3:3d} {4:3d}\n'.format(
                        h1e[i,j], i+1, j+1, 0, 0))
                else:
                    fout.write('({0:15.8e},{1:15.8e}) {2:3d} {3:3d} {4:3d} {5:3d}\n'.format(
                        h1e[i,j].real, h1e[i,j].imag, i+1, j+1, 0, 0))

    # write nuc energy
    if nk == 1:
        fout.write('{0:15.8e} {1:>3d} {2:>3d} {3:>3d} {4:>3d}\n'.format(
            enuc, 0, 0, 0, 0))
    else:
        fout.write('{0:15.8e}                   {1:>3d} {2:>3d} {3:>3d} {4:>3d}\n'.format(
            enuc, 0, 0, 0, 0))
