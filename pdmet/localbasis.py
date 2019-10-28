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
from functools import reduce
from pyscf.pbc.tools import pbc as pbctools
from pyscf import lib, ao2mo
from pDMET.tools import tchkfile, tunix
from pDMET.pdmet import helper, df
from pDMET.lib.build import libdmet

    
    
class Local:
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
        self.kmesh = w90.mp_grid_loc
        self.kmf = kmf   
        self.kpts = kmf.kpts
        self.Nkpts = kmf.kpts.shape[0]    
        self.nao = cell.nao_nr()

        
        scell, self.phase = self.get_phase(self.cell, self.kpts, self.kmesh)
        self.ao2lo = self.get_ao2lo(w90)    # Used to transform AO to local basis in k-space
        self.nlo = self.ao2lo.shape[-1]
        
        #-------------------------------------------------------------
        # Construct the effective Hamiltonian due to the frozen core  | 
        #-------------------------------------------------------------  
        
        # Active part info
        self.active = np.zeros([cell.nao_nr()], dtype=int)

        for orb in range(cell.nao_nr()):
            if (orb+1) not in w90.exclude_bands: self.active[orb] = 1  
        self.nelec_per_cell = np.int32(cell.nelectron - np.sum(kmf.mo_occ_kpts[0][self.active==0]))     
        self.nelec_total = self.Nkpts * self.nelec_per_cell             # per computional super cell

        
        full_OEI_k = kmf.get_hcore()
        mo_k = kmf.mo_coeff_kpts
        coreDM_kpts = []
        for kpt in range(self.Nkpts):
            coreDMmo  = kmf.mo_occ_kpts[kpt].copy()
            coreDMmo[self.active==1] = 0
            coreDMao = reduce(np.dot, (mo_k[kpt], np.diag(coreDMmo), mo_k[kpt].T.conj()))
            coreDM_kpts.append(coreDMao)
    
        self.coreDM_kpts = np.asarray(coreDM_kpts, dtype=np.complex128)
        coreJK_kpts = kmf.get_veff(cell, self.coreDM_kpts, hermi = 1, kpts = self.kpts, kpts_band = None)

        # Core energy from the frozen orbitals
        self.e_core = cell.energy_nuc() + 1./self.Nkpts *lib.einsum('kij,kji->', full_OEI_k + 0.5*coreJK_kpts, self.coreDM_kpts).real        
               
        # 1e integral for the active part
        self.actOEI_kpts = full_OEI_k + coreJK_kpts     
        self.loc_actOEI_kpts = self.ao_2_loc(self.actOEI_kpts, self.ao2lo)

        # Fock for the active part          
        self.fullfock_kpts = kmf.get_fock()            
        self.loc_actFOCK_kpts = self.ao_2_loc(self.fullfock_kpts, self.ao2lo)     
        self.actJK_kpts = self.fullfock_kpts - self.actOEI_kpts     

        
    def make_loc_1RDM_kpts(self, umat, OEH_type='FOCK'):
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
            eigvals = np.asarray([eigvals[kpt][idx_kpts[kpt]] for kpt in range(self.Nkpts)])
            eigvecs = np.asarray([eigvecs[kpt][:,idx_kpts[kpt]] for kpt in range(self.Nkpts)])
            mo_occ = helper.get_occ_r(self.nelec_total, eigvals)  
            loc_OED = np.asarray([np.dot(eigvecs[kpt][:,mo_occ[kpt]>0]*mo_occ[kpt][mo_occ[kpt]>0], eigvecs[kpt][:,mo_occ[kpt]>0].T.conj())
                                                for kpt in range(self.Nkpts)], dtype=np.complex128)       
            
            return loc_OED
        else:
            pass 
            # TODO: contruct RDM for a ROHF wave function            
        
    def make_loc_1RDM(self, umat, OEH_type='FOCK'):
        '''
        Construct the local 1-RDM at the reference unit cell
        '''    
    
        loc_1RDM_kpts = self.make_loc_1RDM_kpts(umat, OEH_type)
        loc_1RDM_R0 = self.k_to_R0(loc_1RDM_kpts)
        return loc_1RDM_kpts, loc_1RDM_R0
        
    def get_emb_OEI(self, ao2eo):
        '''Get embedding OEI'''
        OEI = lib.einsum('kum,kuv,kvn->mn', ao2eo.conj(), self.actOEI_kpts, ao2eo)
        self.is_real(OEI)
        return OEI.real

    def get_emb_FOCK(self, ao2eo):
        '''Get embedding FOCK used to get core JK in embedding space without explicitly computing core JK in local space, need more efficient algorithm'''    
        FOCK = lib.einsum('kum,kuv,kvn->mn', ao2eo.conj(), self.fullfock_kpts, ao2eo)
        self.is_real(FOCK)  
        return FOCK.real

    def get_emb_JK(self, ao2eo):
        '''Get embedding JK used to get core JK in embedding space without explicitly computing core JK in local space'''   
        emb_JK = lib.einsum('kum,kuv,kvn->mn', ao2eo.conj(), self.actJK_kpts, ao2eo)
        self.is_real(emb_JK)
        return emb_JK.real 
        
    def get_emb_1RDM(self, loc_1RDM_kpts, emb_orbs):
        '''Get 1-RDM rotated in the embedding space'''   
        lo2eo = lib.einsum('kR, Rim -> kim', self.phase.conj().T, emb_orbs) 
        emb_1RDM = lib.einsum('kum,kuv,kvn->mn', lo2eo.conj(), loc_1RDM_kpts, lo2eo)
        self.is_real(emb_1RDM)
        return emb_1RDM.real 

    def get_emb_coreJK(self, ao2eo, emb_TEI, emb_1RDM):
        '''Get embedding core JK'''
        ''' TODO: need to debug and make more efficient
        '''
        emb_JK = self.get_emb_JK(ao2eo)
        J = lib.einsum('pqrs,rs->pq', emb_TEI, emb_1RDM)
        K = lib.einsum('prqs,rs->pq', emb_TEI, emb_1RDM) 
        emb_actJK = J - 0.5*K  
        emb_coreJK = emb_JK - emb_actJK
        return emb_coreJK
        
    def get_emb_TEI(self, ao2eo):
        '''Get embedding TEI with density fitting'''
        mydf = self.kmf.with_df   
        TEI = df.get_emb_eri_fast(self.cell, mydf, ao2eo)[0]       
        return TEI
        
    def get_TEI(self, ao2eo): 
        '''Get embedding TEI without density fitting'''
        kconserv = pbctools.get_kconserv(self.cell, self.kpts)
        
        Nkpts, nao, neo = ao2eo.shape
        TEI = 0.0
        for i in range(Nkpts):
            for j in range(Nkpts):
                for k in range(Nkpts):            
                    l = kconserv[i,j,k]    
                    ki, COi = self.kpts[i], ao2eo[i]
                    kj, COj = self.kpts[j], ao2eo[j]
                    kk, COk = self.kpts[k], ao2eo[k]
                    kl, COl = self.kpts[l], ao2eo[l]                
                    TEI += self.kmf.with_df.ao2mo([COi,COj,COk,COl], [ki,kj,kk,kl], compact = False)
                    
        return TEI.reshape(neo,neo,neo,neo).real/Nkpts
        
    def get_loc_TEI(self, ao2lo=None):  
        '''Get local TEI in R-space without density fitting''' 
        kconserv = pbctools.get_kconserv(self.cell, self.kpts)
        if ao2lo is None: ao2lo = self.ao2lo
        
        Nkpts, nao, nlo = ao2lo.shape
        size = Nkpts*nlo
        mo_phase = lib.einsum('kui,Rk->kuRi', ao2lo, self.phase.conj()).reshape(Nkpts,nao, size)
        TEI = 0.0
        for i in range(Nkpts):
            for j in range(Nkpts):
                for k in range(Nkpts):            
                    l = kconserv[i,j,k]    
                    ki, COi = self.kpts[i], mo_phase[i]
                    kj, COj = self.kpts[j], mo_phase[j]
                    kk, COk = self.kpts[k], mo_phase[k]
                    kl, COl = self.kpts[l], mo_phase[l]            
                    TEI += self.kmf.with_df.ao2mo([COi,COj,COk,COl], [ki,kj,kk,kl], compact = False)   
        self.is_real(TEI)
        return TEI.reshape(size,size,size,size).real/Nkpts
        
    def locTEI_to_dmetTEI(self, loc_TEI, emb_orbs):
        '''Transform local TEI in R-space to embedding space''' 
        NRs, nlo, neo = emb_orbs.shape
        emb_orbs = emb_orbs.reshape([NRs*nlo,neo])
        TEI = ao2mo.incore.full(ao2mo.restore(8, loc_TEI, nao), emb_orbs, compact=False)
        TEI = TEI.reshape(neo,neo,neo,neo)
        return TEI  
        
    def get_1RDM_Rs(self, imp_1RDM):
        '''Construct a R-space 1RDM from the reference cell 1RDM''' 
        NRs, nlo = imp_1RDM.shape[:2]
        imp_1RDM_kpts = lib.einsum('Rij,Rk->kij', imp_1RDM, self.phase)*np.sqrt(self.Nkpts)
        RDM1_Rs = self.k_to_R(imp_1RDM_kpts)
        return RDM1_Rs
        
    def get_phase(self, cell=None, kpts=None, kmesh=None):
        '''
        Get a super cell and the phase matrix that transform from real to k-space 
        '''
        if kmesh is None : kmesh = w90.mp_grid_loc
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        
        a = cell.lattice_vectors()
        Ts = lib.cartesian_prod((np.arange(kmesh[0]), np.arange(kmesh[1]), np.arange(kmesh[2])))
        Rs = np.dot(Ts, a)
        NRs = Rs.shape[0]
        phase = 1/np.sqrt(NRs) * np.exp(1j*Rs.dot(kpts.T))
        scell = pbctools.super_cell(cell, kmesh)
        
        return scell, phase

    def get_ao2lo(self, w90):
        '''
        Compute the k-space Wannier orbitals
        '''
        ao2lo = []
        for kpt in range(self.Nkpts):
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
        lo2eo = lib.einsum('Rk, Rim -> kim', self.phase.conj(), emb_orbs) 
        ao2eo = lib.einsum('kui, kim -> kum', self.ao2lo, lo2eo) 
        return ao2eo   

    def ao_2_loc(self, M_kpts, ao2lo=None):
        '''
        Transform an k-space AO integral to local orbitals
        '''      
        if ao2lo is None: ao2lo = self.ao2lo
        return lib.einsum('kui,kuv,kvj->kij', ao2lo.conj(), M_kpts, ao2lo) 
        
    def k_to_R(self, M_kpts):  
        '''Transform AO or LO integral/1-RDM in k-space to R-space
            u, v    : k-space ao indices 
            i, j    : k-space mo/local indices
            m, n    : embedding indices
            capital : R-space indices 
            R, S, T : R-spacce lattice vector
        ''' 
        NRs, Nkpts = self.phase.shape
        nao = M_kpts.shape[-1]
        M_Rs = lib.einsum('Rk,kuv,Sk->RuSv', self.phase, M_kpts, self.phase.conj())
        M_Rs = M_Rs.reshape(NRs*nao, NRs*nao)
        self.is_real(M_Rs)
        return M_Rs.real
        
    def k_to_R0(self, M_kpts):  
        '''Transform AO or LO integral/1-RDM in k-space to the reference unit cell
            u, v    : k-space ao indices 
            i, j    : k-space mo/local indices
            m, n    : embedding indices
            capital : R-space indices 
            R, S, T : R-spacce lattice vector
            
            M(k) -> M(0,R) with index Ruv
        ''' 
        NRs, Nkpts = self.phase.shape
        nao = M_kpts.shape[-1]
        M_R0 = lib.einsum('Rk,kuv,k->Ruv', self.phase, M_kpts, self.phase[0].conj())
        self.is_real(M_R0)
        return M_R0.real
        
    def R_to_k(self, M_Rs):  
        '''Transform AO or LO integral/1-RDM in R-space to k-space 
            u, v    : k-space ao indices 
            i, j    : k-space mo/local indices
            m, n    : embedding indices
            capital : R-space indices 
            R, S, T : R-spacce lattice vector
        ''' 
        NRs, Nkpts = self.phase.shape
        nao = M_Rs.shape[0]//NRs
        M_Rs = M_Rs.reshape(NRs,nao,NRs,nao)
        M_kpts = lib.einsum('Rk,RuSv,Sk->kuv', self.phase.conj(), M_Rs, self.phase)
        return M_kpts
        
    def is_real(self, M, threshold=1.e-7):
        '''Check if a matrix is real with a threshold'''
        assert(abs(M.imag).max() < threshold), 'The imaginary part is larger than %s' % (str(threshold)) 
   
  