import numpy as np
cimport numpy as np

FTYPE = np.float
ctypedef np.float_t FTYPE_t
CTYPE = np.complex
ctypedef np.complex_t CTYPE_t

def intor(mA, mB, intor_name):

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
    I = gto.moleintor.getints2c(intor_name, atmAB, basAB, envAB, shls_slice=shls, hermi=0)

    return I

def k2origin(M):
    '''Takes an AO matrix in K-space and
    returns the corresponding AO matrix
    in real-space for ONLY the origin cell.
    M[nk,nA,nB] -> N[nA,nB]'''

    cdef float nk = float(M.shape[0])
    N = M.sum(axis=0) / nk
    return N.real

def k2real(M, kpts, gvec,):
    '''Takes an AO matrix in k-space and
    returns the corresponding AO matrix
    in real-space.
    M[nk,nA,nB] -> N[ng,nA,nB]'''

    cdef int nk = len(kpts)
    cdef int ng = len(gvec)
    cdef int nA = M.shape[1]
    cdef int nB = M.shape[2]
    cdef int nC = nA * nB

    eikg = np.exp(-1.j * np.dot(gvec, kpts.transpose()))
    N = (np.dot(eikg, M.reshape(nk,nC)) / float(nk)).reshape(ng,nA,nB)

    return N

def real2k(M, kpts, gvec):
    '''Take the AO matrix in real-space and
    returns the corresponding AO matrix
    in k-space.
    M[ng,nA,nB] -> N[nk,nA,nB]'''

    cdef int nk = len(kpts)
    cdef int ng = len(gvec)
    cdef int nA = M.shape[1]
    cdef int nB = M.shape[2]
    cdef int nC = nA * nB

    eikg = np.exp(1.j * np.dot(kpts, gvec.transpose()))
    N = (np.dot(eikg, M.reshape(ng,nC))).reshape(nk,nA,nB)

    return N

def origin2k(M, kpts):
    '''take the AO matrix in real-space at
    the origin cell and returns the corresponding
    k-space matrix.
    M[nA,nB] -> N[nk,nA,nB]'''

    cdef int nk = len(kpts)
    N = np.array([M for i in range(nk)])
    return N

def origin2k2(M, gmol, kpts, gvec):
    '''take the AO matrix in real-space at
    the origin cell and returns the corresponding
    k-space matrix.
    M[nA,nB] -> N[nk,nA,nB]'''

    cdef int nk = len(kpts)
    cdef int ng = len(gvec)
    cdef int nA = M.shape[0]
    cdef int nB = M.shape[1]
    cdef int nC = nA * nB

    Smat = gmol.intor('cint1e_ovlp_sph')
    cdef int nG = Smat.shape[0]
    Smat = Smat[np.ix_(range(nA), range(nG))]
    M = np.dot(M, Smat)
    M = M.reshape(nA,ng,nB).swapaxes(0,1)
    eikg = np.exp(1.j * np.dot(kpts, gvec.transpose()))
    N = (np.dot(eikg, M.reshape(ng,nC))).reshape(nk,nA,nB)

    return N

def g2mol(M):
    cdef int ng = M.shape[0]
    cdef int nA = M.shape[1]
    cdef int nB = M.shape[2]
    N = M.swapaxes(0,1).reshape(nA,ng*nB)
    return N

def mol2g(M, ng):
    cdef int nA = M.shape[0]
    cdef int nB = M.shape[1] // ng
    N = M.reshape(nA,ng,nB).swapaxes(0,1)
    return N

def get_Gv(np.ndarray gs, np.ndarray lattice, bohr=False):
    '''Get g-vectors.'''

    cdef int i, j, ngs

    # get unit conversion (always output in bohr)
    cdef float a2b
    if bohr:
        a2b = 1.0
    else:
        a2b = 1.8897261328856432

    cdef np.ndarray[FTYPE_t, ndim=2] temp = np.zeros((1,3))

    # make g-vectors
    Gv = np.zeros((1,3))
    for i in range(3):
        ngs = gs[i]
        for j in range(1, ngs):
                temp[0] = j * lattice[i] * a2b
                Gv = np.append(Gv, temp, axis=0)
                temp[0] = - j * lattice[i] * a2b 
                Gv = np.append(Gv, temp, axis=0)

    # return
    return Gv

def get_real_intor(mA, mB, intor_name, int ndim, np.ndarray lattice,
    np.ndarray gs, thresh=1e-10):
    '''Get an integral on the real-space g vector grid.
    NB: "lattice" is given in bohr.'''

    from copy import deepcopy as copy

    cdef int i, igs, ia
    cdef int nA = mA.nao_nr(), nB = mB.nao_nr()
    cdef float error, eP, eM
    cdef bint ldoI

    # generate translated, copied mole object
    coords = mB.atom_coords()
    symbols = [mB.atom_symbol(i) for i in range(mA.natm)]

    # do integrals
    temp = intor(mA, mB, intor_name)
    I = np.array([temp])
    Tvec = np.zeros((1,3), dtype=float)

    # cycle over each Cartesian direction
    for ia in range(ndim):
        ldoI = True

        # cycle over number of translations in this direction
        for igs in range(1, gs[ia]+1):

            # should we do integrals?
            if ldoI:

                # translate cell in positive direction
                mC = mB.copy()
                mC.atom = []
                gvec = float(igs) * lattice[ia]
                mC.unit = 'bohr'
                for i in range(mA.natm):
                    new_coords = coords[i] + gvec
                    mC.atom.append([symbols[i], new_coords])
                mC.build()

                # get integral and append
                tempP = intor(mA, mC, intor_name)
                I = np.append(I, [tempP], axis=0)
                Tvec = np.append(Tvec, [gvec], axis=0)

                # translate cell in negative direction
                mC = mB.copy()
                mC.atom = []
                gvec = -1. * float(igs) * lattice[ia]
                mC.unit = 'bohr'
                for i in range(mA.natm):
                    new_coords = coords[i] + gvec
                    mC.atom.append([symbols[i], new_coords])
                mC.build()

                # get integral and append
                tempM = intor(mA, mC, intor_name)
                I = np.append(I, [tempM], axis=0)
                Tvec = np.append(Tvec, [gvec], axis=0)

                # are we converged?
                ldoI = ((np.abs(tempP).max() > thresh) or
                        (np.abs(tempM).max() > thresh))

    return I, Tvec

#def get_j_kpts(mA, D, hermi, kpts, kpt_band):
#
#    # get Dmat in real space
#    Dr = k2real(D)
#
#    # generate translated, copied mole object
#    coords = mA.atom_coords()
#    symbols = [mA.atom_symbol(i) for i in range(mA.natm)]
#
#    temp = intor(mA, mB, intor_name)
#    I = np.array([temp])
#
#    for ia in range(3):
#        igs = 1
#        ldoP = True
#        ldoM = True
#        for igs in range(1, gs[ia]):
#
#            # do positive direction
#            if ldoP:
#                mC = mB.copy()
#                mC.atom = []
#                gvec = igs * vec[ia] * a2b 
#                mC.unit = 'bohr'
#                for i in range(mA.natm):
#                    new_coords = coords[i] + gvec
#                    mC.atom.append([symbols[i], new_coords])
#                mC.build()
#
#                # get integral and append
#                temp = intor(mA, mC, intor_name)
#                I = np.append(I, [temp], axis=0)
#
#                # are we converged?
#                ldoP = np.abs(temp).max() > thresh
#            else:
#                temp = np.zeros((1, nA, nB), dtype=I.dtype)
#                I = np.append(I, temp, axis=0)
#
#            # do negative direction
#            if ldoM:
#                mC = mB.copy()
#                mC.atom = []
#                gvec = - igs * vec[ia] * a2b 
#                mC.unit = 'bohr'
#                for i in range(mA.natm):
#                    new_coords = coords[i] + gvec
#                    mC.atom.append([symbols[i], new_coords])
#                mC.build()
#
#                # get integral and append
#                temp = intor(mA, mC, intor_name)
#                I = np.append(I, [temp], axis=0)
#
#                # are we converged?
#                ldoM = np.abs(temp).max() > thresh
#            else:
#                temp = np.zeros((1, nA, nB), dtype=I.dtype)
#                I = np.append(I, temp, axis=0) 
