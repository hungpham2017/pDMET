#!/usr/bin/env python -u 
'''
pDMET: Density Matrix Embedding theory for Periodic Systems
Copyright (C) 2018 Hung Q. Pham. All Rights Reserved.
A few functions in pDMET are modifed from QC-DMET Copyright (C) 2015 Sebastian Wouters

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Email: Hung Q. Pham <pqh3.14@gmail.com>
'''

import numpy as np
import scipy
from scipy import fftpack
from functools import reduce
from pyscf.pbc import tools
from pyscf.pbc.tools import pywannier90, k2gamma
from pyscf import lib, ao2mo
from pDMET.tools import tchkfile, tunix
from pDMET.pdmet import helper, df
from pDMET.lib.build import libdmet

    
    
class WF:
    def __init__(self, cell, kmf, w90, chkfile=None):
        '''
        TODO: need to be written
        Args:
            kmf        : a k-dependent mean-field wf
            w90        : a converged wannier90 object
        '''        
        
        # Collect cell and kmf object information
        self.cell = cell
        self.spin = cell.spin        
        self.e_tot = kmf.e_tot
        self.w90 = w90
        self.kmesh = w90.mp_grid_loc
        self.kmf = kmf   
        self.ovlp = self.kmf.get_ovlp()
        self.kpts = kmf.kpts
        self.nkpts = kmf.kpts.shape[0]    
        self.nao = cell.nao_nr()
        
        self.scell, self.phase = self.get_phase(self.cell, self.kpts, self.kmesh)
        self.nLs = self.phase.shape[0]
        self.tmap = self.make_tmap(w90.mp_grid_loc)
        self.ao2lo = self.get_ao2lo()    # Used to transform AO to local basis in k-space

        #-------------------------------------------------------------
        # Construct the effective Hamiltonian due to the frozen core  | 
        #-------------------------------------------------------------  
        
        # Active part info
        self.active = np.zeros([cell.nao_nr()], dtype=int)

        for orb in range(cell.nao_nr()):
            if (orb+1) not in w90.exclude_bands: self.active[orb] = 1
        self.nactorbs = np.sum(self.active)   
        self.norbs = self.nkpts * self.nactorbs
        self.nactelecs = np.int32(cell.nelectron - np.sum(kmf.mo_occ_kpts[0][self.active==0]))        
        self.nelec = self.nkpts * self.nactelecs

        # TODO: decide which part can be stored in a check file later
        # chkfile_exist = None     
        # if chkfile is not None: chkfile_exist = tunix.check_exist(chkfile)
        # if chkfile_exist == None or chkfile_exist == False:
       

        fullOEI_kpts = kmf.get_hcore()
        mo_kpts = kmf.mo_coeff_kpts
        coreDM_kpts = []
        for kpt in range(self.nkpts):
            coreDMmo  = kmf.mo_occ_kpts[kpt].copy()
            coreDMmo[self.active==1] = 0
            coreDMao = reduce(np.dot, (mo_kpts[kpt], np.diag(coreDMmo), mo_kpts[kpt].T.conj()))
            coreDM_kpts.append(coreDMao)
    
        self.coreDM_kpts = np.asarray(coreDM_kpts, dtype=np.complex128)
        coreJK_kpts = kmf.get_veff(cell, self.coreDM_kpts, hermi = 1, kpts = self.kpts, kpts_band = None)

        # Core energy from the frozen orbitals
        self.e_core = cell.energy_nuc() + 1./self.nkpts *np.einsum('kij,kji->', fullOEI_kpts + 0.5*coreJK_kpts, self.coreDM_kpts).real        
               
        # 1e integral for the active part
        self.actOEI_kpts = fullOEI_kpts + coreJK_kpts     
        self.loc_actOEI_kpts = self.to_local(self.actOEI_kpts, self.ao2lo)

        # Fock for the active part          
        self.fullfock_kpts = kmf.get_fock()            
        self.loc_actFOCK_kpts = self.to_local(self.fullfock_kpts, self.ao2lo)     
        self.actJK_kpts = self.fullfock_kpts - self.actOEI_kpts     
        # self.loc_actOEI_Ls = self.to_Ls(self.loc_actOEI_kpts)
  
        
        # 2e integral for the active part
        from pyscf.pbc.tools import pbc as pbctools
        kconserv = pbctools.get_kconserv(cell, self.kpts)   
        # self.loc_actTEI_kpts = self.get_tei_kpts(kconserv, self.ao2lo)            
        # self.loc_actTEI_Ls = self.to_Ls2e(self.loc_actTEI_kpts, kconserv) 
        
        # Save integrals to chkfile:            
        # if chkfile_exist == False:
            # tchkfile.save_pdmet_int(self, chkfile)
            # print('-> Chkfile saving ... done')                 
                        
        # elif chkfile_exist == True:
            # print('-> Load the integral ...')
            # savepdmet = tchkfile.load_pdmet_int(chkfile)
            # self.ao2lo               = savepdmet.CO
            # self.WFs              = savepdmet.WFs    
            # self.e_core           = savepdmet.e_core
            # self.coreDM_kpts      = savepdmet.coreDM_kpts
            # self.loc_actOEI_kpts  = savepdmet.loc_actOEI_kpts
            # self.loc_actOEI_Ls    = savepdmet.loc_actOEI_Ls
            # self.loc_actTEI_kpts  = savepdmet.loc_actTEI_kpts   
            # self.loc_actTEI_Ls    = savepdmet.loc_actTEI_Ls
            # self.loc_actFOCK_kpts = savepdmet.loc_actFOCK_kpts    
            # self.loc_actFOCK_Ls   = savepdmet.loc_actFOCK_Ls         
            # self.loc_actVHF_kpts  = savepdmet.loc_actVHF_kpts             

        
    def construct_locOED_kpts(self, umat, OEH_type='FOCK', verbose=0):
        '''
        Construct MOs/one-electron density matrix at each k-point in the local basis
        with a certain k-independent correlation potential umat
        '''    

        #Two choices for the one-electron Hamiltonian
        if OEH_type == 'OEI':
            OEH_kpts = self.loc_actOEI_kpts + umat
        elif OEH_type == 'FOCK':
            OEH_kpts = self.loc_actFOCK_kpts + umat  
        elif OEH_type == 'proj':
            OEH_kpts = umat                 # umat here is simply the new FOCK from the correlated DM
        else:
            raise Exception('the current one-electron Hamiltonian type is not supported')
    
        if self.spin == 0:
            eigvals, eigvecs = np.linalg.eigh(OEH_kpts)
            idx_kpts = eigvals.argsort()
            eigvals = np.asarray([eigvals[kpt][idx_kpts[kpt]] for kpt in range(self.nkpts)], dtype=np.float64)
            eigvecs = np.asarray([eigvecs[kpt][:,idx_kpts[kpt]] for kpt in range(self.nkpts)], dtype=np.complex128)
            mo_occ = helper.get_occ_r(self.nelec, eigvals)  
            loc_OED = np.asarray([np.dot(eigvecs[kpt][:,mo_occ[kpt]>0]*mo_occ[kpt][mo_occ[kpt]>0], eigvecs[kpt][:,mo_occ[kpt]>0].T.conj())
                                                for kpt in range(self.nkpts)], dtype=np.complex128)       
            
            return loc_OED
        else:
            pass 
            # TODO: contruct RDM for a ROHF wave function            
        
    def construct_locOED_Ls(self, umat, OEH_type='FOCK', verbose=0):
        '''
        Construct MOs/one-electron density matrix dm_{pq}^{0L} at each lattice vector
        with a certain k-independent correlation potential umat
        '''    
    
        loc_OED_kpts = self.construct_locOED_kpts(umat, OEH_type, verbose=verbose)
        ao_D_kpts = self.kmf.make_rdm1()
        loc_OED_Ls = self.k_to_R(loc_OED_kpts)
        return loc_OED_kpts, loc_OED_Ls
        
    def dmet_oei(self, ao2eo):
        '''Get embedding OEI'''
        oei = np.einsum('kum,kuv,kvn->mn', ao2eo.conj(), self.actOEI_kpts, ao2eo)
        assert(abs(oei.imag).max() < 1e-7)
        return oei.real

    def dmet_fock(self, ao2eo):
        '''Get embedding FOCK used to get core JK in embedding space without explicitly computing core JK in local space, need more efficient algorithm'''    
        fock = np.einsum('kum,kuv,kvn->mn', ao2eo.conj(), self.fullfock_kpts, ao2eo)
        assert(abs(fock.imag).max() < 1e-7)
        return fock.real

    def dmet_JK(self, ao2eo):
        '''Get embedding JK used to get core JK in embedding space without explicitly computing core JK in local space'''    
        dmetJK = np.einsum('kum,kuv,kvn->mn', ao2eo.conj(), self.actJK_kpts, ao2eo)
        assert(abs(dmetJK.imag).max() < 1e-7)
        return dmetJK.real 

    def dmet_corejk(self, ao2eo, dmetTEI, dmet_1RDM):
        '''Get embedding core JK'''
        ''' TODO: need to debug and make more efficient
        '''
        dmetJK = self.dmet_JK(ao2eo)
        J = np.einsum('pqrs,rs->pq', dmetTEI, dmet_1RDM)
        K = np.einsum('prqs,rs->pq', dmetTEI, dmet_1RDM) 
        dmetJKemb = J - 0.5*K  
        dmetJKcore = dmetJK - dmetJKemb
        return dmetJKcore
        
    def dmet_tei(self, ao2eo):
        '''Get embedding TEI with density fitting'''
        mydf = self.kmf.with_df   
        tei = df.get_emb_eri_fast(self.cell, mydf, ao2eo)[0]       
        return tei
        
    def get_phase(self, cell=None, kpts=None, kmesh=None):
        '''
        Get a super cell and the phase matrix that transform from real to k-space 
        '''
        if kmesh is None : kmesh = w90.mp_grid_loc
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        
        a = cell.lattice_vectors()
        Ts = lib.cartesian_prod((np.arange(kmesh[0]), np.arange(kmesh[1]), np.arange(kmesh[2])))
        Ls = np.dot(Ts, a)
        nLs = Ls.shape[0]
        phase = 1/np.sqrt(nLs) * np.exp(1j*Ls.dot(kpts.T))
        scell = tools.super_cell(cell, kmesh)
        
        return scell, phase

    def get_ao2lo(self, w90=None):
        '''
        Compute the k-space Wannier orbitals
        '''
        if w90 is None: w90 = self.w90
        ao2lo = []
        for kpt in range(self.nkpts):
            mo_included = w90.mo_coeff_kpts[kpt][:,w90.band_included_list]
            mo_in_window = w90.lwindow[kpt]         
            C_opt = mo_included[:,mo_in_window].dot(w90.U_matrix_opt[kpt].T)           
            ao2lo.append(C_opt.dot(w90.U_matrix[kpt].T))        
           
        ao2lo = np.asarray(ao2lo, dtype=np.complex128)
        return ao2lo
        
    def get_ao2eo(self, emb_orbs):
        '''
        Get the transformation matrix from AO to EO
            u, v    : k-space ao indices 
            i, j    : k-space mo/local indices
            m, n    : embedding indices
            capital : R-space indices 
            R, S, T : R-spacce lattice vector
        ''' 
        emb_orbs_Ls = emb_orbs.reshape(self.nLs, self.nactorbs, 4)
        lo2eo = np.einsum('kR, Rim -> kim', self.phase.conj().T, emb_orbs_Ls) 
        ao2eo = np.einsum('kui, kim -> kum', self.ao2lo, lo2eo) 
        return ao2eo

    def to_kspace(self, M):
        '''
        Transform an one-electron matrix M_{pq}(L) to the k-space
        TODO: considered to remove
        '''
        return libdmet.FT1e(self.nkpts, self.kpts, self.nLs, self.Ls, M)  
        
    def to_Ls_old(self, Mk):
        '''
        Transform an one-electron matrix M_{pq}(k) to the L-space
        TODO: will be removed
        '''
        if Mk.ndim == 2: Mk = npasarray([Mk for kpt in range(self.nkpts)])
        return libdmet.iFFT1e(self.tmap, self.phase, Mk).real       
        
    def to_Ls_sparse(self, Mat_kpt, kpt):
        '''
        Transform an one-electron matrix M_{pq}(k) to the L-space
        TODO: will be removed
        '''
        return libdmet.iFT1e_sparse(self.nkpts, kpt, self.nLs, self.Ls, Mat_kpt).real     

    def to_local(self, Mk, ao2lo=None):
        '''
        Transform an one-electron operator M_{pq}(k) in the ao basis to the local basis
        '''      
        if ao2lo is None: ao2lo = self.ao2lo
        return np.einsum('kui,kuv,kvj->kij', ao2lo.conj(), Mk, ao2lo) 
        
    def make_tmap(self, kmesh):  
        '''Exploring translational symmetry
           TODO: I have to call it a translational map now. Should have better name and better algorithm  
           consider to remove and find another way to reconstruct the supercell 2e ERIs
        '''
        nimgs = [kpt//2 for kpt in kmesh]
        Ts = lib.cartesian_prod((np.arange(-nimgs[0],nimgs[0]+1),np.arange(-nimgs[1],nimgs[1]+1),np.arange(-nimgs[2],nimgs[2]+1)))
        map = np.zeros([self.nLs,self.nLs],dtype=np.int64)  
        for a in range(self.nLs):
            Ta = Ts[a]
            deltaT = Ts - Ta         
            for b in range(self.nLs):    
                for i in range(3):
                    if deltaT[b,i] < -(kmesh[i]//2): deltaT[b,i] = deltaT[b,i] + kmesh[i]
                    if deltaT[b,i] > kmesh[i]//2: deltaT[b,i] = deltaT[b,i] - kmesh[i]
                for L in range(self.nLs//2+1): 
                        if (deltaT[b] == Ts[L]).all(): map[a,b] = L
                        if (deltaT[b] == Ts[self.nLs-1-L]).all(): map[a,b] = self.nLs-1-L
                    
        return map
        
    def k_to_R(self, M_kpts):  
        '''Transform k-space to R-space, used for both AO and LO integral/1-RDM
            u, v    : k-space ao indices 
            i, j    : k-space mo/local indices
            m, n    : embedding indices
            capital : R-space indices 
            R, S, T : R-spacce lattice vector
        ''' 
        M_Ls = np.einsum('Rk,kuv,Sk->RuSv', self.phase, M_kpts, self.phase.conj())
        M_Ls = M_Ls.reshape(self.nLs*self.nao, self.nLs*self.nao)
        assert(abs(M_Ls.imag).max() < 1e-7)
        return M_Ls.real
        
    def get_TEI(self, ao2eo): 
        '''Get embedding TEI without density fitting'''
        from pyscf.pbc.tools import pbc as pbctools
        kconserv = pbctools.get_kconserv(self.cell, self.kpts)
        
        Nk, nao, neo = ao2eo.shape
        TEI = 0.0
        for i in range(self.nkpts):
            for j in range(self.nkpts):
                for k in range(self.nkpts):            
                    l = kconserv[i,j,k]    
                    ki, COi = self.kpts[i], ao2eo[i]
                    kj, COj = self.kpts[j], ao2eo[j]
                    kk, COk = self.kpts[k], ao2eo[k]
                    kl, COl = self.kpts[l], ao2eo[l]                
                    TEI += self.kmf.with_df.ao2mo([COi,COj,COk,COl], [ki,kj,kk,kl], compact = False)
                    
        return TEI.reshape(neo,neo,neo,neo).real/self.nLs
        
    def get_loc_TEI(self, ao2lo=None):  
        '''Get local TEI in R-space without density fitting''' 
        
        from pyscf.pbc.tools import pbc as pbctools
        kconserv = pbctools.get_kconserv(self.cell, self.kpts)
        if ao2lo is None: ao2lo = self.ao2lo
        
        Nk, nao, nlo = ao2lo.shape
        size = Nk*nlo
        mo_phase = lib.einsum('kui,Rk->kuRi', ao2lo, self.phase.conj()).reshape(Nk,nao, size)
        TEI = 0.0
        for i in range(self.nkpts):
            for j in range(self.nkpts):
                for k in range(self.nkpts):            
                    l = kconserv[i,j,k]    
                    ki, COi = self.kpts[i], mo_phase[i]
                    kj, COj = self.kpts[j], mo_phase[j]
                    kk, COk = self.kpts[k], mo_phase[k]
                    kl, COl = self.kpts[l], mo_phase[l]            
                    TEI += self.kmf.with_df.ao2mo([COi,COj,COk,COl], [ki,kj,kk,kl], compact = False)   
        assert(abs(TEI.imag).max() < 1e-7)
        return TEI.reshape(size,size,size,size).real/self.nLs
        
    def locTEI_to_dmetTEI(self, locTEI, emb_orbs):
        '''Transform local TEI in R-space to embedding space''' 
        nao,neo = emb_orbs.shape
        tei = ao2mo.incore.full(ao2mo.restore(8, locTEI, nao), emb_orbs, compact=False)
        tei = tei.reshape(neo,neo,neo,neo)
        return tei   
   
  