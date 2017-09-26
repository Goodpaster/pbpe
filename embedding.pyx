from __future__ import print_function
import numpy as np
cimport numpy as np
import scipy as sp

FTYPE = np.float
ctypedef np.float_t FTYPE_t
CTYPE = np.complex
ctypedef np.complex_t CTYPE_t

####################################################################################################
#                                       Periodic-in-Periodic                                       #
####################################################################################################
def plow_in_plow(inp):
    '''Periodic-in-periodic embedding.'''

    from scf import do_embedding
    from scf import print_orbital_energies
    from scf import get_total_energy
    from main import pstr

    # do freeze-and-thaw cycles
    if inp.embed.cycles > -1:
        pstr ("pDFT-in-pDFT Embedding")
        inp.timer.start('periodic embedding')

        # set some initial values
        inp.ift = 0
        dconv = np.sqrt(inp.embed.conv)
        error = 1.
        errD  = 1.
        if 'energy-periodic' in inp.h5py:
            eold = inp.h5py['energy-periodic']
        else:
            eold = inp.h5py.create_dataset('energy-periodic', data=np.zeros((2)))
        enew = [None, None]

        # begin cycles
        while (error > inp.embed.conv or errD > dconv) and inp.ift < inp.embed.cycles:
            inp.ift += 1

            # embed A in B
            DAnew, enew[0] = do_embedding(0, inp)
            errD = sp.linalg.norm(inp.Dmat[0][...] - DAnew)
            if eold[0] == 0.:
                error = np.abs(enew[0])
            else:
                error = np.abs(eold[0] - enew[0])
            inp.Dmat[0][...] = np.copy(DAnew)
            del (DAnew)

            # embed B in A
            if not inp.embed.freezeb:
                DBnew, enew[1] = do_embedding(1, inp)
                errD += sp.linalg.norm(inp.Dmat[1][...] - DBnew)
                if eold[1] == 0.:
                    error += np.abs(enew[1])
                else:
                    error += np.abs(eold[1] - enew[1])
                inp.Dmat[1][...] = np.copy(DBnew)
                del (DBnew)

            print_error(inp.ift, error, errD)
            eold[0] = enew[0].real
            eold[1] = enew[1].real

        if error <= inp.embed.conv:
            pstr ("SCF converged after {0} cycles".format(inp.ift), delim='!', addline=False)
        elif inp.embed.cycles > 0:
            pstr ("SCF NOT converged", delim='!', addline=False)

        # do final embedding for subsystem A
        inp.ift += 1
        Dnew, enew[0] = do_embedding(0, inp, llast=True)
        errD = sp.linalg.norm(inp.Dmat[0][...] - Dnew)
        error = np.abs(eold[0] - enew[0])
        inp.Dmat[0][...] = np.copy(Dnew)
        del (Dnew)
        print_error(inp.ift, error, errD)

        for i in range(2):
            print_orbital_energies(inp.Energy[i], inp.Fermi[i])

        inp.timer.end('periodic embedding')

####################################################################################################
#                                  Periodic-in-Periodic Energies                                   #
####################################################################################################

    # make supermolecular density matrix
    pstr ("Supermolecular pDFT")
    if 'dsup' in inp.h5py:
        print ('Reading supermolecular density from file...')
        inp.Dsup = inp.h5py['dsup'][...]
    else:
        inp.Dsup = None

    # get total energy and do canonical KS-DFT
    etot = get_total_energy(inp, inp.kpts, dsup=inp.Dsup)
    if 'dsup' in inp.h5py:
        inp.h5py['dsup'][...] = inp.Dsup
    else:
        inp.Dsup = inp.h5py.create_dataset('dsup', data=inp.Dsup)

    # print energies
    if inp.embed.cycles > -1:
        pstr ("pDFT-in-pDFT Energies")
        if inp.lgamma:
            v = inp.cSCF[0].get_veff(dm=inp.Dmat[0][0])
            esub = inp.cSCF[0].energy_tot(dm=inp.Dmat[0][0])
        else:
            esub = inp.cSCF[0].energy_tot(dm=inp.Dmat[0][...])
        inp.etot = etot
        inp.esub = esub
        print ("Subsystem pDFT           {0:17.12f}".format(esub))
        print ("Embedding                {0:17.12f}".format(etot-esub))
        print ("pDFT-in-pDFT             {0:17.12f}".format(etot))
        print ("Supermolecular pDFT      {0:17.12f}".format(inp.Esup))
        print ("Difference               {0:17.12f}".format(etot-inp.Esup))
    else:
        print ("")
        print ("Supermolecular pDFT      {0:17.12f}".format(inp.Esup))

    # return
    return inp

####################################################################################################
#                                Periodic HIGH-in-Periodic LOW DFT                                 #
####################################################################################################
def phigh_in_plow(inp):
    '''Periodic high level-in-periodic low level embedding.'''

    from main import pstr
    from scf import diagonalize
    from pyscf import lib
    from pyscf import pbc

    if inp.method != inp.embed.method and not inp.method in ('ccsd', 'ccsd(t)') and not inp.finite:
        pstr ('pHIGH-in-pLOW Embedding')
        inp.timer.start('emb pHIGH')

        # get embedding potential
        fock = inp.Fock[0]
        try:
            v = inp.cSCF[0].get_veff(dm=inp.Dmat[0][...])
        except TypeError:
            v = inp.cSCF[0].get_veff(dm_kpts=inp.Dmat[0][...])
        v_embed = fock - v 

        # read density from file
        if 'dpA' in inp.h5py:
            dmA = inp.h5py['dpA']
        else:
            dmA = np.copy(inp.Dmat[0][...])
            dmA = inp.h5py.create_dataset('dpA', data=dmA)

        # initialize calculation
        if inp.method in ('hf', 'hartree-fock'):
            kmf = pbc.scf.KRHF(inp.cell[0], inp.kpts)
        else:
            kmf = pbc.dft.KRKS(inp.cell[0], inp.kpts)
            kmf.xc = inp.method
        kmf.kpts = inp.kpts
        kmf.grids = inp.sSCF.grids
        kmf.init_guess = '1e'
        kmf.verbose = inp.verbose
        kmf.max_cycle = inp.maxiter

        # build df object
        kmf.with_df = inp.DF(inp.fcell[0])
        kmf.with_df.kpts = inp.kpts
        kmf.with_df.auxbasis = inp.auxbasis
        try:
            j_only = abs(kmf._numint.hybrid_coeff(kmf.xc)) < 1e-3
        except AttributeError:
            j_only = False
        kmf.with_df.build(j_only=j_only)

        # do SCF
        kmf.get_hcore = lambda *args: v_embed
        err = 1.
        erd = 1.
        eold = None
        icyc = 0 
        diis = lib.diis.DIIS()
        smat = kmf.get_ovlp(kpts=inp.kpts)

        while (icyc < inp.embed.cycles) and (err > inp.embed.conv or erd > inp.embed.dconv):
            icyc += 1
            V = kmf.get_veff(dm=dmA[...])
            fock = v_embed + V
            fock = diis.update(fock)
            if eold is None: eold = np.einsum('kab,kba', dmA[...], fock) / (2. * inp.nkpts)
            dnew, c, mo_occ, e, fermi = diagonalize(fock, smat, inp.kpts, inp.cell[0].nelectron)
            enew = np.einsum('kab,kba', dnew, fock) / (2. * inp.nkpts)
            erd = np.linalg.norm(dnew - dmA[...])
            err = np.abs(eold - enew)

            print_error(icyc, err, erd)

            eold = np.copy(enew)
            dmA[...] = np.copy(dnew)

        # are we converged?
        if err <= inp.embed.conv:
            pstr ("SCF converged after {0} cycles".format(icyc), delim='!', addline=False)
        else:
            pstr ("SCF NOT converged", delim='!', addline=False)

        # diagonalize one final time
        emb_high = kmf.kernel(dm0=dmA[...])

        # new errors
        dnew = kmf.make_rdm1()
        erd = np.linalg.norm(dnew - dmA[...])
        dmA[...] = np.copy(dnew)
        enew = np.einsum('kab,kba', dnew, fock) / (2. * inp.nkpts)
        err = np.abs(eold - enew)
        print_error(icyc+1, err, erd)

        pstr ("pHIGH-in-pLOW Energies")
        print ("Subsystem pHIGH           {0:16.12f}".format(emb_high))
        print ("Embedding                {0:17.12f}".format(inp.etot-inp.esub))
        emb_tot = inp.etot - inp.esub + emb_high
        emb_cor = emb_tot + inp.Esup - inp.etot
        print ("pHIGH-in-pLOW             {0:16.12f}".format(emb_tot))
        print ("pHIGH-in-pLOW Corrected   {0:16.12f}".format(emb_cor))

        inp.timer.end('emb pHIGH')

    # return
    return inp

####################################################################################################
#                                     Finite-in-Periodic (low)                                     #
####################################################################################################
def low_in_plow(inp):
    '''Finite low level-in-periodic low level embedding.'''

    from scf import do_embedding
    from scf import print_orbital_energies
    from scf import get_total_energy
    from main import pstr
    from intor import k2origin, origin2k
    from pyscf import lib, scf

    if inp.finite:
        pstr ("LOW-in-pLOW Embedding")
        inp.timer.start('finite embedding')

        # create the molecular object
        mol = inp.cell[0].to_mol()
        inp.mol = mol

        # transform density matrix to real space
        # or read from file if present
        if 'dmA' in inp.h5py:
            print ('Reading finite density from file')
            dmA = inp.h5py['dmA']
        else:
            dmA = k2origin(inp.Dmat[0][...])
            dmA = inp.h5py.create_dataset('dmA', data=dmA)

        # create molecular SCF object at low level of theory
        if inp.embed.method in ('hf', 'hartree-fock', 'ccsd', 'ccsd(t)'):
            mfAl = scf.RHF(mol)
        else:
            mfAl = scf.RKS(mol)
            mfAl.xc = inp.embed.method
            mfAl.grids = inp.sSCF.grids
        mfAl.max_cycle = inp.embed.subcycles

        # get number of filled orbitals
        n = np.zeros((inp.nao[0]))
        n[0:mol.nelectron//2] = 2.
        sA = mfAl.get_ovlp()
        print ('Number of transformed electrons: {0:12.6f}'.format(
               np.trace(np.dot(dmA[...], sA))))
        DIIS = lib.diis.DIIS()

        # do second freeze-and-thaw with finite-in-periodic systems
        err   = 1.
        erd   = 1.
        enew  = None
        icyc  = 0
        hcore = k2origin(inp.hcore[...])[np.ix_(inp.sub2sup[0], inp.sub2sup[0])]
        hcore = hcore[np.ix_(inp.sub2sup[0], inp.sub2sup[0])]
        hB    = inp.hcore[...][np.ix_(range(inp.nkpts), inp.sub2sup[1], inp.sub2sup[1])]

        # get fock matrix, transform to real space
        fock = do_embedding(0, inp, ldiag=False)
        fock = k2origin(fock)
        vA   = mfAl.get_veff(dm=dmA[...])
        hembed = fock - vA

        # get old energies
        eold  = 0.5 * np.trace(np.dot(dmA[...], hembed))
        eold += 0.5 * np.trace(np.dot(dmA[...], fock))
        eBold = np.einsum('kab,kab', hB, inp.Dmat[1]).real

        fAold = None
        while err > inp.embed.conv and icyc < inp.embed.cycles:
            icyc += 1

            # get fock matrix, transform to real space
            if icyc > 1:
                fock = do_embedding(0, inp, ldiag=False)
                fock = k2origin(fock)

                # get the embedding potential
                vA = mfAl.get_veff(dm=dmA[...])
                hembed = fock - vA
            mfAl.get_hcore = lambda *args: hembed

            # diagonalize in subcycles
            eAold = None
            dAold = np.copy(dmA[...])
            jcyc = 0
            DIIS = lib.diis.DIIS()
            while (err > inp.embed.conv or erd > inp.embed.dconv) and jcyc < inp.embed.subcycles:
                jcyc += 1

                # update fock matrix
                if jcyc > 1: vA = mfAl.get_veff(dm=dAold)
                fock = hembed + vA

                # apply mixing
                if inp.mix is not None and fAold is not None:
                    fock = fock * (1. - inp.mix) + fAold * inp.mix
                fAold = np.copy(fock)

                # apply diis
                if inp.diis:
                    fock = DIIS.update(fock)

                # apply virtual orbital shifting
                if inp.shift:
                    fock += ( sA - np.dot(np.dot(dA, dAold*0.5), sA) ) * inp.shift

                # diagonalize
                e, c = sp.linalg.eigh(fock, sA)
                dA = np.dot(c * n, c.transpose())

                # new energies
                enew  = 0.5 * np.trace(np.dot(dA, hembed))
                enew += 0.5 * np.trace(np.dot(dA, fock))
                if eAold is None: eAold = np.copy(eold)
                err = np.abs(eAold - enew)
                eAold = np.copy(enew)
                erd = np.linalg.norm(dAold - dA)
                dAold = np.copy(dA)

                if inp.embed.subcycles > 1:
                    print ('Subsys A   iter:{0:<2d}   |dE|:{1:16.12f}   '
                           '|ddm|:{2:16.12f}'.format(jcyc, err, erd))

            # new density error
            erd = sp.linalg.norm(dmA[...] - dA)
            dmA[...] = np.copy(dA)

            # new energies
            enew  = 0.5 * np.trace(np.dot(dA, hembed))
            enew += 0.5 * np.trace(np.dot(dA, fock))
            err = np.abs(eold - enew)
            eold = np.copy(enew)

            # transform density matrix back into k-space
            inp.Dmat[0] = origin2k(dA, inp.kpts)

            # do periodic embedding of subsystem B
            if inp.ldob:
                dmBold = np.copy(inp.Dmat[1][...])
                jcyc = 0
                erB  = 1.
                errB = 1.
                while jcyc < 1 and errB > inp.embed.conv:
                    jcyc += 1

                    # update the total fock matrix in periodic space
                    fock = do_embedding(0, inp, ldiag=False)
                    fock = k2origin(fock)

                    # get new density of B
                    dB, enew = do_embedding(1, inp)
                    erB = sp.linalg.norm(inp.Dmat[1][...] - dB)
                    inp.Dmat[1] = np.copy(dB)

                    eBnew = np.einsum('kab,kab', hB, inp.Dmat[1]).real
                    errB = np.abs(eBold - eBnew)
                    eBold = np.copy(eBnew)
                    print ('Subsys B   iter:{0:<2d}   |dE|:{1:16.12f}   '
                           '|ddm|:{2:16.12f}'.format(jcyc, errB, erB))

            print_error(icyc, err, erd)

        # are we converged?
        if err <= inp.embed.conv:
            pstr ("SCF converged after {0} cycles".format(icyc), delim='!', addline=False)
        else:
            pstr ("SCF NOT converged", delim='!', addline=False)

        # converged; final iteration
        # get fock matrix, transform to real space
        fock = do_embedding(0, inp, ldiag=False)
        fock = k2origin(fock)
        vA = mfAl.get_veff(dm=dmA[...])
        hembed = fock - vA
        if 'cluster/hembed' in inp.h5py:
            inp.h5py['cluster/hembed'][...] = np.copy(hembed)
        else:
            inp.h5py.create_dataset('cluster/hembed', data=hembed)

        # do kernel
        mfAl.get_hcore = lambda *args: hembed
        mfAl.max_cycle = inp.maxiter
        mfAl.kernel(dm0=dmA[...])
        dA = mfAl.make_rdm1()

        # new energies
        erd = sp.linalg.norm(dmA[...] - dA)
        dmA[...] = np.copy(dA)
        enew  = 0.5 * np.trace(np.dot(dA, hembed))
        enew += 0.5 * np.trace(np.dot(dA, fock))
        err = np.abs(eold - enew)
        eold = np.copy(enew)
        print_error(icyc+1, err, erd)

        # print orbital energies
        print ('')
        print ('Cluster Orbital Energies in eV:')
        print (repr(mfAl.mo_energy * 27.2113961)) # h2ev
        print_orbital_energies(inp.Energy[1], inp.Fermi[1])

        # get and print energies
        pstr ("LOW-in-pLOW Energies")
        inp.Dmat[0] = origin2k(dmA[...], inp.kpts)
        etot = get_total_energy(inp, inp.kpts, ldosup=False)
        vA = mfAl.get_veff(dm=dmA[...])
        esub = mfAl.energy_tot(dm=dmA[...])
        inp.eemb = eemb = etot - esub
        inp.ecorr = ecorr = inp.Esup - etot
        print ("Subsystem LOW            {0:17.12f}".format(esub))
        print ("Embedding                {0:17.12f}".format(eemb))
        print ("LOW-in-pLOW              {0:17.12f}".format(etot))
        print ("Supermolecular pLOW      {0:17.12f}".format(inp.Esup))
        print ("Difference               {0:17.12f}".format(-ecorr))

        inp.timer.end('finite embedding')

    # return
    return inp

####################################################################################################
#                                     Finite-in-Periodic (high)                                    #
####################################################################################################
def high_in_plow(inp):
    '''Finite high level-in-periodic low level embedding.'''

    from main import pstr
    from pyscf import lib, scf, cc

    if inp.finite and inp.method != inp.embed.method:
        inp.timer.start('finite embedding high')
        pstr ("HIGH-in-pLOW Embedding")
        # create molecular SCF object at higher level of theory
        mol = inp.mol
        if inp.method in ('hf', 'hartree-fock', 'ccsd', 'ccsd(t)'):
            mfA = scf.RHF(mol)
        else:
            mfA = scf.RKS(mol)
            mfA.xc = inp.method
            mfA.grids = inp.sSCF.grids
        mfA.max_cycle = inp.embed.subcycles

        # read density from file
        if 'DHA' in inp.h5py:
            dmA = inp.h5py['DHA'][...]
        else:
            dmA = inp.h5py['dmA'][...]
            dmA = inp.h5py.create_dataset('DHA', data=dmA)

        # create the embedding potential for A
        sA = mfA.get_ovlp()
        hembed = inp.h5py['cluster/hembed'][...]
        mfA.get_hcore = lambda *args: hembed

        # get number of filled orbitals
        n = np.zeros((mol.nao_nr()))
        n[0:mol.nelectron//2] = 2.

        # do SCF
        DIIS = lib.diis.DIIS()
        err = 1.
        erd = 1.
        eold = None
        icyc = 0 
        while (icyc < inp.maxiter) and (err > inp.conv):
            icyc += 1

            # get fock matrix
            vA = mfA.get_veff(dm=dmA[...])
            fock = hembed + vA
            fock = DIIS.update(fock)

            # diagonalize
            eA, cA = sp.linalg.eigh(fock, sA) 
            dA = np.dot( cA * n, cA.transpose().conjugate())

            # errors
            erd = sp.linalg.norm(dA - dmA[...])
            if eold is None: eold = np.trace(np.dot(dmA[...], fock))
            enew = np.trace(np.dot(dA, fock))
            err = np.abs(eold - enew)

            dmA[...] = np.copy(dA)
            eold = np.copy(enew)
            print_error(icyc, err, erd)

        if err <= inp.conv:
            pstr ("SCF converged after {0} cycles".format(icyc), delim='!', addline=False)
        else:
            pstr ("SCF NOT converged", delim='!', addline=False)

        # do 1 final SCF to set some values
        icyc += 1
        mfA.max_cycle = inp.maxiter
        esub = mfA.kernel(dm0=dmA[...])
        dA = mfA.make_rdm1()
        erd = sp.linalg.norm(dA - dmA[...])
        dmA[...] = np.copy(dA)
        enew = np.trace(np.dot(dA, fock))
        err = np.abs(eold - enew)
        print_error(icyc, err, erd)

        # get and print energies
        pstr ("HIGH-in-pLOW Energies")
        print ("Subsystem HIGH           {0:17.12f}".format(esub))
        print ("Embedding                {0:17.12f}".format(inp.eemb))
        print ("HIGH-in-pLOW             {0:17.12f}".format(esub+inp.eemb))
        print ("HIGH-in-pLOW Corrected   {0:17.12f}".format(esub+inp.eemb+inp.ecorr))

        inp.timer.end('finite embedding high')

####################################################################################################
#                                      Finite WF calculation                                       #
####################################################################################################

        # do CCSD/CCSD(T) calculation
        if inp.method in ('ccsd', 'ccsd(t)'):
            if inp.method in ('ccsd(t)'):
                pstr ("Finite Embedded CCSD(T) Calculation")
            else:
                pstr ("Finite Embedded CCSD Calculation")

            inp.timer.start('finite emb ccsd')
            mCCSD = cc.CCSD(mfA)
            ecc, t1, t2 = mCCSD.kernel()

            if inp.method in ('ccsd(t)'):
                inp.timer.start('finite emb ccsd(t)')
                from pyscf.cc import ccsd_t
                eris = mCCSD.ao2mo()
                ec3 = ccsd_t.kernel(mCCSD, eris)
                inp.timer.end('finite emb ccsd(t)')
                pstr ("CCSD(T)-in-pDFT Energies")
            else:
                pstr ("CCSD-in-pDFT Energies")

            print ("Subsystem HF             {0:17.12f}".format(esub))
            print ("Subsystem Correlation    {0:17.12f}".format(ecc))
            print ("Subsystem CCSD           {0:17.12f}".format(esub+ecc))
            if inp.method in ('ccsd(t)'):
                print ("Subsystem CCSD(T)        {0:17.12f}".format(esub+ecc+ec3))
            print ("Embedding                {0:17.12f}".format(inp.eemb))
            print ("CCSD-in-pDFT             {0:17.12f}".format(esub+ecc+inp.eemb))
            print ("CCSD-in-pDFT Corrected   {0:17.12f}".format(esub+ecc+inp.eemb+inp.ecorr))
            if inp.method in ('ccsd(t)'):
                print ("CCSD(T)-in-pDFT          {0:17.12f}".format(esub+ecc+ec3+inp.eemb))
                print ("CCSD(T)-in-pDFT Corrected {0:16.12f}".format(esub+ecc+ec3+inp.eemb+inp.ecorr))
            inp.timer.end('finite emb ccsd')

    # return
    return inp

def print_error(icyc, de, ddm):
    print ('iter: {0:<3d}     |dE|: {1:16.12f}     |ddm|: {2:16.12f}'.format(
            icyc, de, ddm))
