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
from pyscf.pbc import scf
from pdmet import helper, df, df_hamiltonian
from pdmet.tools import tchkfile, tunix

    
    
class Local:
    def __init__(self, cell, kmf, w90, xc_omega=0.2):
        '''
        TODO: need to be written
        Args:
            kmf        : a k-dependent mean-field wf
            w90        : a converged wannier90 object
            
        Indices
            u, v    : k-space ao indices 
            i, j    : k-space mo/local indices
            m, n    : embedding indices
            capital : R-space indices 
            R, S, T : R-spacce lattice vector
            
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
        self.nelec_total = 0
        for mo_occ in kmf.mo_occ_kpts:
            self.nelec_total += int(mo_occ[w90.band_included_list].sum())
        self.nelec_per_cell = self.nelec_total // self.Nkpts

        full_OEI_k = kmf.get_hcore()
        coreDM_kpts = []
        for kpt, mo_coeff in enumerate(kmf.mo_coeff_kpts):
            core_band = np.asarray(mo_coeff.shape[1] * [True])
            core_band[w90.band_included_list] = False
            coreDMmo  = kmf.mo_occ_kpts[kpt][core_band].copy()
            mo_k = mo_coeff[:, core_band]
            coreDMao = reduce(np.dot, (mo_k, np.diag(coreDMmo), mo_k.T.conj()))
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
        self.loc_actJK_kpts = self.ao_2_loc(self.actJK_kpts, self.ao2lo)      #DEBUG: used to debug, may be removed 
        
        # DF-like DMET
        self.xc_omega = xc_omega
        dm_kpts = self.kmf.make_rdm1()
        self.vj, self.vk = self.kmf.get_jk(dm_kpts=dm_kpts)
        self.dm = self.kmf.make_rdm1()
        self.h_core = self.kmf.get_hcore()
        
        self.kks = scf.KKS(self.cell, self.kpts).density_fit()  
        self.kks.with_df._cderi = self.kmf.with_df._cderi 
        if self.xc_omega is not None:
            self.vklr = self.kmf.get_k(self.cell, self.dm, 1, self.kpts, None, omega=self.xc_omega)
            self.vksr = self.vk - self.vklr
        
    def make_loc_1RDM_kpts(self, umat, OEH_type='FOCK', get_band=False, get_ham=False, dft_HF=None):
        '''
        Construct 1-RDM at each k-point in the local basis given a u mat
        '''    
        
        # Get modified mean-field Hamiltinian: h_tilder = h + u
        if OEH_type == 'FOCK':
            OEH_kpts = self.loc_actFOCK_kpts + umat  
        else:
            # For DF-like cost function
            OEH_kpts = df_hamiltonian.get_OEH_kpts(self, umat,  xc_type=OEH_type,  dft_HF=dft_HF)
            
        if self.spin == 0:
            eigvals, eigvecs = np.linalg.eigh(OEH_kpts)
            idx_kpts = eigvals.argsort()
            eigvals = np.asarray([eigvals[kpt][idx_kpts[kpt]] for kpt in range(self.Nkpts)])
            eigvecs = np.asarray([eigvecs[kpt][:,idx_kpts[kpt]] for kpt in range(self.Nkpts)])
            mo_occ = helper.get_occ_r(self.nelec_total, eigvals)  
            loc_OED = np.asarray([np.dot(eigvecs[kpt][:,mo_occ[kpt]>0]*mo_occ[kpt][mo_occ[kpt]>0], eigvecs[kpt][:,mo_occ[kpt]>0].T.conj())
                                                for kpt in range(self.Nkpts)], dtype=np.complex128)       

            if get_band:
                return eigvals, eigvecs
            elif get_ham:
                return OEH_kpts, eigvals, eigvecs
            else:
                return OEH_kpts, loc_OED
        else:
            pass 
            # TODO: contruct RDM for a ROHF wave function            
        
    def make_loc_1RDM(self, umat, OEH_type='FOCK', dft_HF=None):
        '''
        Construct the local 1-RDM at the reference unit cell
        '''    
        loc_OEH_kpts, loc_1RDM_kpts = self.make_loc_1RDM_kpts(umat, OEH_type=OEH_type, dft_HF=dft_HF)
        loc_1RDM_R0 = self.k_to_R0(loc_1RDM_kpts)
        return loc_OEH_kpts, loc_1RDM_kpts, loc_1RDM_R0
        
    def get_emb_OEI(self, ao2eo):
        '''Get OEI projected into the embedding basis'''
        OEI = lib.einsum('kum,kuv,kvn->mn', ao2eo.conj(), self.actOEI_kpts, ao2eo)
        self.is_real(OEI)
        return OEI.real
        
    def get_core_OEI(self, ao2core):
        '''Get OEI projected into the core (unentangled) basis'''
        OEI = lib.einsum('kum,kuv,kvn->mn', ao2core.conj(), self.actOEI_kpts, ao2core)
        self.is_real(OEI)
        return OEI.real

    def get_emb_FOCK(self, emb_orbs, loc_OEH_kpts):
        '''Get modified FOCK in embedding basis'''    
        lo2eo = lib.einsum('Rk, Rim -> kim', self.phase.conj(), emb_orbs)
        emb_fock_kpts = lib.einsum('kim,kij,kjn->mn', lo2eo.conj(), loc_OEH_kpts, lo2eo)
        self.is_real(emb_fock_kpts)  
        return emb_fock_kpts.real

    def get_emb_JK(self, loc_1RDM_kpts, ao2eo):
        '''Get embedding JK from a local 1-RDM'''   
        ao_1RDM_kpts = self.loc_2_ao(loc_1RDM_kpts)
        ao_JK = self.kmf.get_veff(self.cell, ao_1RDM_kpts, hermi=1, kpts=self.kpts, kpts_band=None)
        emb_JK = lib.einsum('kum,kuv,kvn->mn', ao2eo.conj(), ao_JK, ao2eo)
        self.is_real(emb_JK)
        return emb_JK.real 
        
    def get_core_JK(self, ao2core, loc_core_1RDM):
        '''Get JK projected into the core (unentangled) basis''' 
        ao_core_kpts = self.loc_2_ao(loc_core_1RDM)
        ao_core_JK = self.kmf.get_veff(self.cell, ao_core_kpts, hermi=1, kpts=self.kpts, kpts_band=None)
        core_JK = lib.einsum('kum,kuv,kvn->mn', ao2core.conj(), ao_core_JK, ao2core)
        self.is_real(core_JK)
        return core_JK.real 

    def get_emb_coreJK(self, emb_JK, emb_TEI, emb_1RDM):
        '''Get embedding core JK
           Attributes:
            emb_JK  : total JK projected into the embedding space
            emb_TEI : TEI projected into the embedding space
            emb_1RDM: 1RDM projected into the embedding space
        '''
        J = lib.einsum('pqrs,rs->pq', emb_TEI, emb_1RDM)
        K = lib.einsum('prqs,rs->pq', emb_TEI, emb_1RDM) 
        emb_actJK = J - 0.5*K
        emb_coreJK = emb_JK - emb_actJK     # Subtract JK from the active space (frag + bath) from the totak JK
        return emb_coreJK
        
    def get_emb_TEI(self, ao2eo):
        '''Get embedding TEI with density fitting'''
        mydf = self.kmf.with_df   
        TEI = df.get_emb_eri_gdf(self.cell, mydf, ao2eo)[0]       
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
        
    def loc_to_emb_TEI(self, loc_TEI, emb_orbs):
        '''Transform local TEI in R-space to embedding space''' 
        NRs, nlo, neo = emb_orbs.shape
        emb_orbs = emb_orbs.reshape([NRs*nlo,neo])
        TEI = ao2mo.incore.full(ao2mo.restore(8, loc_TEI, nao), emb_orbs, compact=False)
        TEI = TEI.reshape(neo,neo,neo,neo)
        return TEI  
        
    def emb_to_loc_kpts(self, emb_matrix, emb_orbs):
        '''Get k-space embedding 1e quantities in the k-space local basis
        TODO: DEBUGGING THIS
        
        '''  
        lo2eo = lib.einsum('Rk, Rim -> kim', self.phase.conj(), emb_orbs) 
        loc_coreJK_kpts = lib.einsum('kim,mn,kjn->kij', lo2eo, emb_matrix, lo2eo.conj())
        return loc_coreJK_kpts
        
    def loc_kpts_to_emb(self, RDM_kpts, emb_orbs):
        '''Transform k-space 1-RDM in local basis to embedding basis'''   
        lo2eo = lib.einsum('Rk, Rim -> kim', self.phase.conj(), emb_orbs) 
        emb_1RDM = lib.einsum('kim, kij, kjn -> mn', lo2eo.conj(), RDM_kpts, lo2eo)
        self.is_real(emb_1RDM)
        return emb_1RDM.real 
        
    def get_emb_mf_1RDM(self, emb_FOCK, Nelec_in_emb):
        '''Get k-space 1-RDM  or derivative 1-RDM in the embedding basis''' 
        npairs = Nelec_in_emb // 2
        sigma, C = np.linalg.eigh(emb_FOCK)
        C = C[:, sigma.argsort()]
        emb_mf_1RDM = 2 * np.dot(C[:,:npairs], C[:,:npairs].T.conj())
        return emb_mf_1RDM
        
    def get_emb_guess_1RDM(self, emb_FOCK, Nelec_in_emb, Nimp, chempot):
        '''Get guessing 1RDM for the embedding problem''' 
        Nemb = emb_FOCK.shape[0]
        npairs = Nelec_in_emb // 2
        chempot_vector = np.zeros(Nemb)
        chempot_vector[:Nimp] = chempot
        emb_FOCK = emb_FOCK - np.diag(chempot_vector)
        sigma, C = np.linalg.eigh(emb_FOCK)
        C = C[:, sigma.argsort()]
        DMguess = 2 * np.dot(C[:,:npairs], C[:,:npairs].T.conj())
        return DMguess

    def get_core_mf_1RDM(self, lo2core, Nelec_in_core, loc_OEH_kpts):
        '''Get k-space 1-RDM  or derivative 1-RDM in the embedding basis''' 
        npairs = Nelec_in_core // 2
        core_FOCK = lib.einsum('kim,kij,kjn->mn', lo2core.conj(), loc_OEH_kpts, lo2core)
        self.is_real(core_FOCK)  
        sigma, C = np.linalg.eigh(core_FOCK.real)
        C = C[:, sigma.argsort()]
        core_mf_1RDM = 2 * np.dot(C[:,:npairs], C[:,:npairs].T.conj())
        return core_mf_1RDM 
        
    def get_1RDM_Rs(self, loc_1RDM_R0):
        '''Construct a R-space 1RDM from the reference cell 1RDM''' 
        NRs, nlo = loc_1RDM_R0.shape[:2]
        loc_1RDM_kpts = lib.einsum('Rk,Rij,k->kij', self.phase.conj(), loc_1RDM_R0, self.phase[0])*self.Nkpts
        loc_1RDM_Rs = self.k_to_R(loc_1RDM_kpts)
        return loc_1RDM_Rs
        
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
            C_opt = mo_included[:,mo_in_window].dot(w90.U_matrix_opt[kpt][ :, mo_in_window].T)           
            ao2lo.append(C_opt.dot(w90.U_matrix[kpt].T))        
           
        ao2lo = np.asarray(ao2lo, dtype=np.complex128)
        return ao2lo
        
    def get_ao2eo(self, emb_orbs):
        '''
        Get the transformation matrix from AO to EO
        ''' 
        lo2eo = lib.einsum('Rk, Rim -> kim', self.phase.conj(), emb_orbs) 
        ao2eo = lib.einsum('kui, kim -> kum', self.ao2lo, lo2eo) 
        return ao2eo  
        
    def get_lo2core(self, core_orbs):
        '''
        Get the transformation matrix from AO to the unentangled orbitals
        ''' 
        lo2core = lib.einsum('Rk, Rim -> kim', self.phase.conj(), core_orbs) 
        return lo2core 
        
    def get_ao2core(self, core_orbs):
        '''
        Get the transformation matrix from AO to the unentangled orbitals
        ''' 
        lo2core = lib.einsum('Rk, Rim -> kim', self.phase.conj(), core_orbs) 
        ao2core = lib.einsum('kui, kim -> kum', self.ao2lo, lo2core) 
        return ao2core 

    def ao_2_loc(self, M_kpts, ao2lo=None):
        '''
        Transform an k-space AO integral to local orbitals
        '''      
        if ao2lo is None: ao2lo = self.ao2lo
        return lib.einsum('kui,kuv,kvj->kij', ao2lo.conj(), M_kpts, ao2lo) 
        
    def loc_2_ao(self, M_kpts, ao2lo=None):
        '''
        Transform an k-space local integral to ao orbitals
        '''      
        if ao2lo is None: ao2lo = self.ao2lo
        return lib.einsum('kui,kij,kvj->kuv', ao2lo, M_kpts, ao2lo.conj()) 
        
    def k_to_R(self, M_kpts):  
        '''Transform AO or LO integral/1-RDM in k-space to R-space
        ''' 
        NRs, Nkpts = self.phase.shape
        nao = M_kpts.shape[-1]
        M_Rs = lib.einsum('Rk,kuv,Sk->RuSv', self.phase, M_kpts, self.phase.conj())
        M_Rs = M_Rs.reshape(NRs*nao, NRs*nao)
        self.is_real(M_Rs)
        return M_Rs.real
        
    def k_to_R0(self, M_kpts):  
        '''Transform AO or LO integral/1-RDM in k-space to the reference unit cell
            M(k) -> M(0,R) with index Ruv
        ''' 
        NRs, Nkpts = self.phase.shape
        nao = M_kpts.shape[-1]
        M_R0 = lib.einsum('Rk,kuv,k->Ruv', self.phase, M_kpts, self.phase[0].conj())
        self.is_real(M_R0)
        return M_R0.real
        
    def R_to_k(self, M_Rs):  
        '''Transform AO or LO integral/1-RDM in R-space to k-space 
        ''' 
        NRs, Nkpts = self.phase.shape
        nao = M_Rs.shape[0]//NRs
        M_Rs = M_Rs.reshape(NRs,nao,NRs,nao)
        M_kpts = lib.einsum('Rk,RuSv,Sk->kuv', self.phase.conj(), M_Rs, self.phase)
        return M_kpts
        
    def R0_to_k(self, M_R0):  
        '''Transform AO or LO integral/1-RDM in R-space to k-space 
        ''' 
        NRs, nao = M_R0.shape[:2]
        M_kpts = lib.einsum('Rk,Ruv,k->kuv', self.phase.conj(), M_R0, self.phase[0])
        return M_kpts * NRs
        
    def is_real(self, M, threshold=1.e-6):
        '''Check if a matrix is real with a threshold'''
        assert(abs(M.imag).max() < threshold), 'The imaginary part is larger than %s' % (str(threshold)) 
   
  