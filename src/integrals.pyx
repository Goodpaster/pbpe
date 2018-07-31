import numpy as np
cimport numpy as np
import scipy as sp

FTYPE = np.float
ctypedef np.float_t FTYPE_t
CTYPE = np.complex
ctypedef np.complex_t CTYPE_t

def intor(mA, mB, intor_name, hermi=0):

    from pyscf import gto 

    # get AOs of A and B
    cdef int nA = mA.nao_nr()
    cdef int sA = len(mA._bas)
    cdef int nB = mB.nao_nr()
    cdef int sB = len(mB._bas)
    cdef int nAB = nA+nB

    # create a supermolecule
    atmAB, basAB, envAB = gto.conc_env(mA._atm, mA._bas, mA._env,
                                       mB._atm, mB._bas, mB._env)

    # get shell slice
    shls = (0, sA, sA, sA+sB)

    # get integral
    I = gto.moleintor.getints2c(intor_name, atmAB, basAB, envAB,
                                shls_slice=shls, hermi=hermi)
    return I

def get_overlap_matrices(inp):
    '''Calculates the overlap and core matrix
    over basis of A and B.'''

    from copy import deepcopy as copy
    from .mole import concatenate_cells
    from pyscf.pbc import scf

    # get number of AOs of A and B
    cdef int nA = inp.nao[0]
    cdef int nB = inp.nao[1]
    cdef int nk = len(inp.kpts)

    # get integrals
    S = inp.tSCF.get_ovlp(kpts=inp.kpts)
    h = inp.tSCF.get_hcore(kpts=inp.kpts)

    # get different elements
    rk = range(nk)
    rA = range(nA)
    rB = range(nA, nA+nB)

    SAA = S[np.ix_(rk, rA, rA)]
    SAB = S[np.ix_(rk, rA, rB)]
    SBA = S[np.ix_(rk, rB, rA)]
    SBB = S[np.ix_(rk, rB, rB)]
    S = [[SAA, SAB], [SBA, SBB]]

    hAA = h[np.ix_(rk, rA, rA)]
    hAB = h[np.ix_(rk, rA, rB)]
    hBA = h[np.ix_(rk, rB, rA)]
    hBB = h[np.ix_(rk, rB, rB)]
    h = [[hAA, hAB], [hBA, hBB]]

    return S, h

def get_2e_matrix(inp):
    '''Calculates the effective potential due to both density
    matrices A and B.'''

    cdef int nA = inp.nao[0]
    cdef int nB = inp.nao[1]
    cdef int nS = inp.sSCF.cell.nao_nr()
    cdef int nk = len(inp.kpts)

    sub2sup = inp.sub2sup

    # make supermolecular density matrix
    if nk == 1:
        dm = np.zeros((nk, nS, nS), dtype=float)
    else:
        dm = np.zeros((nk, nS, nS), dtype=complex)
    for i in range(inp.nsub):
        dm[np.ix_(range(nk), sub2sup[i], sub2sup[i])] += inp.Dmat[i][...]

    # get effective potential
    try:
        V = inp.sSCF.get_veff(dm=dm, kpts=inp.kpts)
    except TypeError:
        try:
            V = inp.sSCF.get_veff(dm_kpts=dm, kpts=inp.kpts)
        except TypeError:
            V = inp.sSCF.get_veff(dm=dm)

    return V

def get_sub2sup(inp):

    # get nao slice for each subsystem
    nssl = [None for i in range(inp.nsub)]
    for i in range(inp.nsub):
        mi = inp.cSCF[i].cell
        nssl[i] = np.zeros((mi.natm), dtype=int)
        for j in range(mi.natm):
            ib = np.where(mi._bas.transpose()[0]==j)[0][0]
            ie = np.where(mi._bas.transpose()[0]==j)[0][-1]
            ir = mi.nao_nr_range(ib,ie+1)
            ir = ir[1] - ir[0]
            nssl[i][j] = ir

    # get nao slice for each atom in supermolecule
    ns = inp.sSCF.cell
    nsl = np.zeros((ns.natm), dtype=int)
    for i in range(ns.natm):
        ib = np.where(ns._bas.transpose()[0]==i)[0][0]
        ie = np.where(ns._bas.transpose()[0]==i)[0][-1]
        ir = ns.nao_nr_range(ib,ie+1)
        ir = ir[1] - ir[0]
        nsl[i] = ir

    # see which nucleii matches up
    sub2sup = [None for i in range(inp.nsub)]
    for i in range(inp.nsub):
        sub2sup[i] = np.zeros((inp.nao[i]), dtype=int)
        for a in range(inp.cSCF[i].cell.natm):
            match = False
            for b in range(inp.sSCF.cell.natm):
                d = inp.sSCF.cell.atom_coord(b) - inp.cSCF[i].cell.atom_coord(a)
                d = np.dot(d,d)
                if d < 1e-3:
                    match = True
                    ia = nssl[i][0:a].sum()
                    ja = ia + nssl[i][a]
                    ib = nsl[0:b].sum()
                    jb = ib + nsl[b]
                    sub2sup[i][ia:ja] = range(ib,jb)
            if not match: print ('ERROR: I did not find an atom match!')

    inp.sub2sup = sub2sup

def get_supercell_gpoints(inp):
    '''Returns the supercell overlap and kinetic
    matrices, and G-vectors.'''

    from .mole import concatenate_mols, g_2_mol

    cdef float error, eP, eM
    cdef bint ldoP, ldoM, ldoI
    intor_ovlp = 'cint1e_ovlp_sph'
    intor_kin  = 'cint1e_kin_sph'

    # initialize some values
    mA = inp.ccell.to_mol()
    gmax = 20
    vec = inp.ccell.lattice_vectors()
    thresh = inp.conv
    cdef int nA = mA.nao_nr()

    # generate translated, copied mole object
    coords   = mA.atom_coords()
    symbols  = [mA.atom_symbol(i) for i in range(mA.natm)]

    # initialize matrices
    inp.gSmat = None
    inp.Gvec  = None

    # generate real-space grid coordinates
    if inp.dimension >= 1:
        xx = np.append([0], np.array([i*j for i in range(1,gmax) for j in [1, -1]]))
    else:
        xx = np.zeros((1))
    if inp.dimension >= 2:
        yy = np.append([0], np.array([i*j for i in range(1,gmax) for j in [1, -1]]))
    else:
        yy = np.zeros((1))
    if inp.dimension >= 3:
        zz = np.append([0], np.array([i*j for i in range(1,gmax) for j in [1, -1]]))
    else:
        zz = np.zeros((1))
    xx = np.outer(xx, vec[0])
    yy = np.outer(yy, vec[1])
    zz = np.outer(zz, vec[2])
    Tvec = np.array([xx[i]+yy[j]+zz[k] for i in range(len(xx)) for j in range(len(yy))
                     for k in range(len(zz))])

    # sort by distance from origin cell
    dist = (Tvec*Tvec).sum(axis=1)
    indx = np.argsort(dist)
    Tvec = Tvec[indx]

    # cycle over all cells
    minS = 1e8
    iG = 0
    for i in range(len(Tvec)):

        # get translational vector
        gvec = Tvec[i]

        # translate mole object
        mC = mA.copy()
        mC.atom = []
        mC.unit = 'bohr'
        for i in range(mA.natm):
            new_coords = coords[i] + gvec
            mC.atom.append([symbols[i], new_coords])
        mC.build()

        # get integral
        tempS = intor(mA, mC, intor_ovlp)
        if np.abs(tempS).max() < thresh: break

        # are we converged? otherwise append
        if np.abs(tempS).max() < minS: minS = np.abs(tempS).max()
        iG += 1
        if inp.gSmat is None:
            inp.gSmat = np.array([tempS])
            inp.Gvec  = np.array([gvec])
            gmole     = mA.copy()
        else:
            inp.gSmat = np.append(inp.gSmat, [tempS], axis=0)
            inp.Gvec  = np.append(inp.Gvec, [gvec], axis=0)
            gmole     = concatenate_mols(gmole, mC, ghost=True)


    # g-vector
    inp.gmole = gmole
    inp.nGv = len(inp.Gvec)
    if (nA*inp.nGv != gmole.nao_nr()): print ('ERROR: get_supercell_gpoints')

    # reshape and return
    print ('Smat conv: {0}     nCells: {1}'.format(minS, iG))
    inp.gSmat = g_2_mol(inp.gSmat, nA, inp.nGv)
    return

def get_xc_pot(mA, mG, dm, g, xc):

    from pyscf import dft

    cdef int nA = mA.nao_nr()
    cdef int nG = mG.nao_nr()

    # get new density matrix
    dm2 = np.zeros((nG, nG))
    dm2[np.ix_(range(0,nA), range(0,nG))] = dm[:]

    # get xc energy and potential
    NI = dft.numint._NumInt()
    n, exc, vxc = NI.nr_rks(mG, g, xc, dm2, hermi=1)
    vxc = vxc[np.ix_(range(0,nA), range(0,nG))]

    return exc, vxc

def get_tot_xc_pot(mA, mG, mB, mH, DA, DB, g, xc):

    from pyscf import dft
    from .mole import concatenate_mols

    cdef int nA = mA.nao_nr()
    cdef int nG = mG.nao_nr()
    cdef int nB = mB.nao_nr()
    cdef int nH = mH.nao_nr()

    # get supermolecule
    mI = concatenate_mols(mG, mH, ghost=True)
    cdef int nI = mI.nao_nr()

    # get new density matrix
    dm = np.zeros((nI, nI))
    dm[np.ix_(range(0,nA), range(0,nG))] = DA[:]
    dm[np.ix_(range(nG,nG+nB), range(nG,nG+nH))] = DB[:]

    # get xc energy and potential
    NI = dft.numint._NumInt()
    n, exc, vxc = NI.nr_rks(mI, g, xc, dm, hermi=1)
    vxc = vxc[np.ix_(range(0,nA), range(0,nG))]

    return exc, vxc

def get_coulomb_energies(mA, mG, vec, dm, gmax, thresh=1e-10):
    '''Get Nuc-El and El-El Coulomb potentials.
    Must be done this way, since Nuc-El or El-El alone
    are divergent.'''

    from copy import deepcopy as copy
    from pyscf import gto
    import _vhf

    cdef int ix, iy, iz, iN, iG
    cdef bint ldoI

    # Initialize energies
    E = None
    ee = None
    Ne = None
    NN = None

    # get some values
    intor_nuc = 'cint1e_rinv_sph'
    intor_J   = 'cint2e_sph'
    atm, bas, env = mG._atm, mG._bas, mG._env
    Cn = mA.atom_coords()
    Zn = mA.atom_charges()
    CnG = mG.atom_coords()
    Sn = [mG.atom_symbol(i) for i in range(mG.natm)]
    cdef int sA = len(mA._bas)
    cdef int sG = len(mG._bas)
    shlsN = (0, sA, 0, sG)
    shlsJ = (0, sA, 0, sG, sG, sG+sG, sG, sG+sA)
    cdef int iRinv = gto.PTR_RINV_ORIG

    xx = [i*j for i in range(gmax[0]) for j in [1, -1]]
    yy = [i*j for i in range(gmax[1]) for j in [1, -1]]
    zz = [i*j for i in range(gmax[2]) for j in [1, -1]]
    ldo = np.array(np.zeros((len(xx),len(yy),len(zz)))+1, dtype=bool)
    xx.pop(0); yy.pop(0); zz.pop(0)

    # cycle over all directions
    iG = 0
    minval = 1e8
    for ix in range(len(xx)):
        for iy in range(len(yy)):
            for iz in range(len(zz)):

                if ldo[ix][iy][iz]:

                    tempN = None
                    tempNN = None

                    # cycle over all nucleii
                    for iN in range(len(Zn)):

                        # get nuclear potential for this nucleii
                        envC = copy(env)
                        envC[iRinv+0] = Cn[iN][0] + ((xx[ix] * vec[0])[0]
                                      + (yy[iy] * vec[1])[0] + (zz[iz] * vec[2])[0])
                        envC[iRinv+1] = Cn[iN][1] + ((xx[ix] * vec[0])[1] 
                                      + (yy[iy] * vec[1])[1] + (zz[iz] * vec[2])[1])
                        envC[iRinv+2] = Cn[iN][2] + ((xx[ix] * vec[0])[2] 
                                      + (yy[iy] * vec[1])[2] + (zz[iz] * vec[2])[2])

                        temp = gto.moleintor.getints2c(intor_nuc, atm, bas, envC,
                               shls_slice=shlsN, hermi=0)
                        temp *= - Zn[iN]
                        temp = np.trace(np.dot(temp, dm.transpose()))

                        if tempN is None:
                            tempN = copy(temp)
                        else:
                            tempN += temp

                        # get nuc-nuc energy for this nucleii
                        if ix!=0 or iy!=0 or iz!=0:
                            dist    = np.zeros((3))
                            dist[0] = Cn[iN][0] + ((xx[ix] * vec[0])[0]
                                    + (yy[iy] * vec[1])[0] + (zz[iz] * vec[2])[0])
                            dist[1] = Cn[iN][1] + ((xx[ix] * vec[0])[1] 
                                    + (yy[iy] * vec[1])[1] + (zz[iz] * vec[2])[1])
                            dist[2] = Cn[iN][2] + ((xx[ix] * vec[0])[2] 
                                    + (yy[iy] * vec[1])[2] + (zz[iz] * vec[2])[2])
                            dist = np.sqrt(np.sum((Cn - dist)**2, axis=1))
                            temp = np.sum(Zn * Zn[iN] / dist) / 2.

                            if tempNN is None:
                                tempNN = copy(temp)
                            else:
                                tempNN += temp
                        else:
                            tempNN = 0.

                    # translate mole object
                    mC = mG.copy()
                    mC.atom = []
                    gvec = xx[ix] * vec[0] + yy[iy] * vec[1] + zz[iz] * vec[2]
                    mC.unit = 'bohr'
                    for iN in range(mG.natm):
                        new_coords = CnG[iN] + gvec
                        mC.atom.append([Sn[iN], new_coords])
                    mC.build()

                    # do Coulomb integral
                    atmAB, basAB, envAB = gto.conc_env(atm, bas, env,
                                          mC._atm, mC._bas, mC._env)
                    tempJ = _vhf.direct_mapdm(intor_J, 's1', 'lk->s1ij', dm, 1,
                           atmAB, basAB, envAB, vhfopt=None, shls_slice=shlsJ)
                    tempJ = 0.5 * np.trace(np.dot(tempJ, dm.transpose()))

                    # check if tempI is converged
                    e = tempNN + tempN + tempJ
                    ldoI = np.abs(e) > thresh
                    iG += 1
#                    print ('cell ({0:3d},{1:3d},{2:3d})     |max| {3:15.10f}'.format(xx[ix],
#                           yy[iy], zz[iz], e))
                    if (np.abs(e) < minval): minval = np.abs(e)
                    if E is None:
                        E = copy(e)
                        Ne = copy(tempN)
                        NN = copy(tempNN)
                        ee = copy(tempJ)
                    else:
                        E += e
                        Ne += tempN
                        NN += tempNN
                        ee += tempJ

                    if not ldoI:
                        # if converged, dont calculate others in the series
                        if ix==0:
                            ldo[:,iy:len(yy)+1:2,iz:len(zz)+1:2] = False
                        if iy==0:
                            ldo[ix:len(xx)+1:2,:,iz:len(zz)+1:2] = False
                        if iz==0:
                            ldo[ix:len(xx)+1:2,iy:len(yy)+1:2,:] = False

    # return
    print ('Convergence of Coulomb energies: {0}     nCells: {1}'.format(minval, iG))
    return NN, Ne, ee

def nuc_energy(mA, tvec, gvec, gmax, vol, eta=0.001, thresh=1e-10):
    '''Calculate the nuclear energy.'''

    from copy import deepcopy as copy

    # initialize
    sqrtpi = 1.7724538509055159
    q = mA.atom_charges()
    Q = q.sum()
    eNN = 0.

    # generate real-space grid coordinates
    xx = np.append([0], np.array([i*j for i in range(1,gmax[0]) for j in [1, -1]]))
    yy = np.append([0], np.array([i*j for i in range(1,gmax[1]) for j in [1, -1]]))
    zz = np.append([0], np.array([i*j for i in range(1,gmax[2]) for j in [1, -1]]))
    xx = np.outer(xx, tvec[0])
    yy = np.outer(yy, tvec[1])
    zz = np.outer(zz, tvec[2])
    Tvec = np.array([xx[i]+yy[j]+zz[k] for i in range(len(xx)) for j in range(len(yy))
                    for k in range(len(zz))])

    # sort by distance from origin cell
    dist = (Tvec*Tvec).sum(axis=1)
    indx = np.argsort(dist)
    Tvec = Tvec[indx]

    # short range / real-space part
    for i in range(mA.natm):
        for j in range(mA.natm):

            t = mA.atom_coord(i) - mA.atom_coord(j)

            iT = -1
            er = 1e8
            while (iT<len(Tvec)-1) and (er > thresh or iT < 1):

                iT += 1

                if (iT==0 and i==j): continue

                d = sp.linalg.norm(t + Tvec[iT])

                temp = 0.5 * q[i] * q[j] * sp.special.erfc( eta * d ) / d

                er = copy(temp)
                eNN += temp
            print (i, j, iT, er)

    # generate k-space grid coordinates
    xx = np.append([0], np.array([i*j for i in range(1,gmax[0]) for j in [1, -1]]))
    yy = np.append([0], np.array([i*j for i in range(1,gmax[1]) for j in [1, -1]]))
    zz = np.append([0], np.array([i*j for i in range(1,gmax[2]) for j in [1, -1]]))
    xx = np.outer(xx, gvec[0])
    yy = np.outer(yy, gvec[1])
    zz = np.outer(zz, gvec[2])
    Gvec = np.array([xx[i]+yy[j]+zz[k] for i in range(len(xx)) for j in range(len(yy))
                    for k in range(len(zz))])

    # sort Gvec by distance
    dist = (Gvec*Gvec).sum(axis=1)
    indx = np.argsort(dist)
    Gvec = Gvec[indx]

    # long range / k-space part
    iG = 0
    er = 1e8
    while (iG<len(Gvec)-1 and er > thresh) or iG < 2:

        iG += 1
        temp = 0.
        f1 = np.pi * np.pi * np.dot(Gvec[iG], Gvec[iG]) / (eta*eta)
        temp = 0.

        for i in range(mA.natm):
            for j in range(mA.natm):

                r = mA.atom_coord(i) - mA.atom_coord(j)
                m2 = np.dot(Gvec[iG], Gvec[iG])
                f2 = 2.j * np.pi * np.dot(Gvec[iG], r)

                temp += ( q[i] * q[j] * np.exp( f2 - f1 ) ) / ( 2. * np.pi * vol * m2 )

        er = copy(temp)
        eNN += temp.real

        print ('k-space ', iG, er)

    # get constant part
    temp = - eta * Q / np.sqrt(np.pi)
    print ('CONSTANT PART ', temp)
    eNN -= temp

    print (eNN)

def get_coulomb_potentials(mA, mG, mB, mH, vec, dm, gmax, thresh=1e-10):
    '''Get Nuc-El and El-El Coulomb potentials.
    Must be done this way, since Nuc-El or El-El alone
    are divergent.'''

    from copy import deepcopy as copy
    from pyscf import gto
    from .vhf import direct_mapdm

    cdef int ix, iy, iz, iN, iG

    # Initialize nuclear potential
    N = None
    J = None
    I = None

    # get some values
    intor_nuc = 'cint1e_rinv_sph'
    intor_J   = 'cint2e_sph'
    atm, bas, env = mG._atm, mG._bas, mG._env
    Cn = mB.atom_coords()
    Zn = mB.atom_charges()
    CnG = mH.atom_coords()
    Sn = [mH.atom_symbol(i) for i in range(mH.natm)]
    cdef int sA = len(mA._bas)
    cdef int sG = len(mG._bas)
    cdef int sB = len(mB._bas)
    cdef int sH = len(mH._bas)
    shlsN = (0, sA, 0, sG)
    shlsJ = (0, sA, 0, sG, sG, sG+sH, sG, sG+sB)
    cdef int iRinv = gto.PTR_RINV_ORIG

    # generate real-space grid coordinates
    xx = np.append([0], np.array([i*j for i in range(1,gmax[0]) for j in [1, -1]]))
    yy = np.append([0], np.array([i*j for i in range(1,gmax[1]) for j in [1, -1]]))
    zz = np.append([0], np.array([i*j for i in range(1,gmax[2]) for j in [1, -1]]))
    xx = np.outer(xx, vec[0])
    yy = np.outer(yy, vec[1])
    zz = np.outer(zz, vec[2])
    Tvec = np.array([xx[i]+yy[j]+zz[k] for i in range(len(xx)) for j in range(len(yy))
                    for k in range(len(zz))])

    # sort by distance from origin cell
    dist = (Tvec*Tvec).sum(axis=1)
    indx = np.argsort(dist)
    Tvec = Tvec[indx]

    # cycle over all cells
    iG = -1
    cdef float err = 1e8
    cdef float minval = 1e8
    while err > thresh and iG < len(Tvec)-1:

        iG += 1
        tempN = None

        # cycle over all nucleii
        for iN in range(len(Zn)):

            # get nuclear potential for this nucleii
            envC = copy(env)
            envC[iRinv+0] = Cn[iN][0] + Tvec[iG][0]
            envC[iRinv+1] = Cn[iN][1] + Tvec[iG][1]
            envC[iRinv+2] = Cn[iN][2] + Tvec[iG][2]

            temp = gto.moleintor.getints2c(intor_nuc, atm, bas, envC,
                   shls_slice=shlsN, hermi=0)
            temp *= - Zn[iN]

            if tempN is None:
                tempN = copy(temp)
            else:
                tempN += temp

        # translate mole object
        mC = mH.copy()
        mC.atom = []
        mC.unit = 'bohr'
        for iN in range(mG.natm):
            new_coords = CnG[iN] + Tvec[iG]
            mC.atom.append([Sn[iN], new_coords])
        mC.build()

        # do Coulomb integral
        atmAB, basAB, envAB = gto.conc_env(atm, bas, env,
                              mC._atm, mC._bas, mC._env)
        tempJ = direct_mapdm(intor_J, 's1', 'lk->s1ij', dm, 1,
               atmAB, basAB, envAB, vhfopt=None, shls_slice=shlsJ)

        # check if tempI is converged
        err = np.abs(tempN+tempJ).max()
#        print ('cell ({0:3d},{1:3d},{2:3d})     |max| {3:15.10f}'.format(xx[ix],
#               yy[iy], zz[iz], np.abs(tempN+tempJ).max()))
        if err < minval: minval = copy(err)
        if N is None and J is None:
            N = copy(tempN)
            J = copy(tempJ)
        else:
            N += tempN
            J += tempJ

    # return
    print ('Convergence of Nuc-El and El-El matrices: {0}      nCells: {1}'.format(minval, iG))
    return N, J

def get_supermol_1e_matrices(inp):
    '''Get supermolecular 1e matrices.'''

    import sys
    from .pstr import pstr

    inp.timer.start('1e matrices')
    print ('Getting 1e matrices:')

    # get smat
    if 'smat' in inp.h5py:
        print ('Reading overlap matrix from file...')
        inp.Smat = inp.h5py['smat']
    else:
        print ('Calculating overlap matrix...')
        try:
            inp.Smat = inp.sSCF.get_ovlp(kpts=inp.kpts)
        except TypeError:
            inp.Smat = np.array([inp.sSCF.get_ovlp()])
        inp.Smat = inp.h5py.create_dataset('smat', data=inp.Smat)

        # check that Smat is not singular
        lsing = False
        for i in range(inp.nkpts):
            lsing = np.linalg.cond(inp.Smat[i][...]) > 1.0 / sys.float_info.epsilon
            if lsing:
                pstr ("ERROR: Singular overlap matrix", delim="!")
                sys.exit()

    # get hcore
    if 'hmat' in inp.h5py:
        print ('Reading hcore from file...')
        inp.hcore = inp.h5py['hmat']
    else:
        print ('Calculating hcore...')
        try:
            inp.hcore = inp.sSCF.get_hcore(kpts=inp.kpts)
        except TypeError:
            inp.hcore = np.array([inp.sSCF.get_hcore()])
        inp.hcore = inp.h5py.create_dataset('hmat', data=inp.hcore)

    inp.timer.end('1e matrices')

    # return
    return inp

def get_subsystem_densities(inp):
    '''Get the subsystem density matrices.'''

    from .pstr import pstr

    for i in range(inp.nsub):
        if '{0}/dmat'.format(i) in inp.h5py:
            print ('Reading initial density of subsystem {0} from file...'.format(i))
            inp.Dmat[i] = inp.h5py['{0}/dmat'.format(i)]
        else:
            inp.timer.start('initial density')
            pstr ('Initial Density of Subsystem {0}'.format(i))
            inp.Dmat[i] = dm_init_guess(inp, inp.cSCF[i], inp.kpts)
            inp.Dmat[i] = inp.h5py.create_dataset('{0}/dmat'.format(i), data=inp.Dmat[i])
            inp.timer.end('initial density')

    return inp


def dm_init_guess(inp, cSCF, kpts):

    import pyscf
    from pyscf.gto import Mole
    from pyscf.scf import RKS

    cdef int nkpts = len(kpts)
    cdef int nao = cSCF.cell.nao_nr()

    if nkpts == 1:
        dm0 = np.zeros((1, nao, nao), dtype=float)
    else:
        dm0 = np.zeros((nkpts, nao, nao), dtype=complex)

    # create a simple mole object to guess the density
    mol = cSCF.cell.to_mol()
    mf = RKS(mol)
    mf.xc = inp.embed.method
    mf.max_cycle = 20
    if inp.smear is not None:
        mf.damp = inp.smear
    try:
        e = mf.kernel()
        dm = mf.make_rdm1()
    except np.linalg.LinAlgError:
        dm = init_guess_by_atom(inp, mol, mf)

    for i in range(nkpts):
        dm0[i] = dm[:,:]

    return dm0


def init_guess_by_atom(inp, mol, mf):

    import numpy as np
    import pyscf
    from pyscf.gto import Mole
    from pyscf.scf import RKS

    nao = mol.nao_nr()
    dm = np.zeros((nao, nao))

    # create a mole object for each atom
    for i in range(mol.natm):
        tmol = Mole()
        tmol.atom = '{0} 0.0 0.0 0.0'.format(mol.atom_symbol(i))
        if 'ghost' in mol.atom_symbol(i).lower():
            continue
        tmol.basis = mol.basis
        tmol.ecp = mol.ecp
        tmol.verbose = 0
        tmol.build()

        tmf = RKS(tmol)
        tdm = tmf.get_init_guess()

        ib = mol.aoslice_by_atom()[i][-2]
        ie = mol.aoslice_by_atom()[i][-1]

        dm[np.ix_(range(ib,ie), range(ib,ie))] += tdm[:,:]

    return dm 
