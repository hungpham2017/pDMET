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
from functools import reduce
from pyscf.pbc.tools import pywannier90
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
        self.kmf = kmf   
        self.ovlp = self.kmf.get_ovlp()
        self.kpts = kmf.kpts
        self.nkpts = kmf.kpts.shape[0]    
        self.nao = cell.nao_nr()
        
        self.nLs, self.Ls, self.phase = self.get_phase(self.cell, self.w90, self.kpts)
        self.tmap = self.make_tmap(w90.mp_grid_loc)
        self.ao2lo = self.get_WFs(self.w90)[0]    # Used to transform AO to local basis in k-space

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
        actOEI_kpts = fullOEI_kpts + coreJK_kpts     
        self.loc_actOEI_kpts = self.to_local(actOEI_kpts, self.ao2lo)
        # self.loc_actOEI_Ls = self.to_Ls(self.loc_actOEI_kpts)
  
        
        # 2e integral for the active part
        from pyscf.pbc.tools import pbc as pbctools
        kconserv = pbctools.get_kconserv(cell, self.kpts)   
        # self.loc_actTEI_kpts = self.get_tei_kpts(kconserv, self.ao2lo)            
        # self.loc_actTEI_Ls = self.to_Ls2e(self.loc_actTEI_kpts, kconserv) 
        
        # Fock for the active part          
        fullfock_kpts = kmf.get_fock()            
        self.loc_actFOCK_kpts = self.to_local(fullfock_kpts, self.ao2lo)      
            
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

        
    def construct_locOED_kpts(self, umat, OEH_type, verbose=0):
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


        
    def construct_locOED_Ls(self, umat, OEH_type, verbose=0):
        '''
        Construct MOs/one-electron density matrix dm_{pq}^{0L} at each lattice vector
        with a certain k-independent correlation potential umat
        '''    
    
        loc_OED_kpts = self.construct_locOED_kpts(umat, OEH_type, verbose=verbose)
        loc_OED_Ls = libdmet.iFFT1e(self.tmap, self.phase, loc_OED_kpts).real        
        return loc_OED_kpts, loc_OED_Ls
        
    def construct_Fock_kpts(self, DMloc_kpts, local=True):
        '''
        Construct total Fock in the ao basis (local = False) or active Fock in the local basis (locala = True)
        '''    
        kpts = self.kmf.kpts
        DMao_kpts = np.asarray([reduce(np.dot,(self.ao2lo[kpt], DMloc_kpts[kpt],self.ao2lo[kpt].T.conj())) for kpt in range(self.nkpts)])
        if not local:
            dm_kpts = self.coreDM_kpts + DMao_kpts
            JKao = self.kmf.get_veff(cell=self.cell, dm_kpts=dm_kpts, kpts=kpts, kpts_band=kpts)
            return self.kmf.get_hcore(self.cell, kpts) + JKao
        else:
            dm_kpts = DMao_kpts
            JKao = self.kmf.get_veff(cell=self.cell, dm_kpts=dm_kpts, kpts=kpts, kpts_band=kpts)
            JKloc = np.asarray([reduce(np.dot,(self.ao2lo[kpt].T.conj(), JKao[kpt],self.ao2lo[kpt])) for kpt in range(self.nkpts)])
            return self.loc_actOEI_kpts + JKloc
        
        
    def dmet_oei(self, ao2eo):
        '''Get embedding OEI'''
        oei = 0.0
        for kpt in range(self.nkpts):
            oei += reduce(np.dot,(ao2eo[kpt].T.conj(), self.loc_actOEI_kpts[kpt], ao2eo[kpt]))
        return oei

    def dmet_tei(self, ao2eo):
        '''Get embedding TEI or 2e ERI'''
        ''' TODO: modify this to adapt it to GDF'''
        mydf = self.kmf.with_df   
        tei = df.get_emb_ERIs(self.cell, mydf, ao2eo)
        return tei        

    def dmet_fock(self, ao2eo):
        '''Get embedding FOCK used to get core JK in embedding space without explicitly computing core JK in local space, need more efficient algorithm'''
        fock = 0.0
        for kpt in range(self.nkpts):
            fock += reduce(np.dot,(ao2eo[kpt].T.conj(), self.loc_actFOCK_kpts[kpt], ao2eo[kpt]))
            
        return fock

    def dmet_corejk(self, ao2eo, dmetOEI, dmetTEI, dmet1RDM):
        '''Get embedding core JK'''
        ''' TODO: need to debug and make more efficient
        '''
        dmetFock = self.dmet_fock(ao2eo)
        J = np.einsum('pqrs,rs->pq', dmetTEI, dmet1RDM)
        K = np.einsum('prqs,rs->pq', dmetTEI, dmet1RDM) 
        dmetJK = dmetFock - dmetOEI - (J - 0.5*K)   
        return dmetJK
        
    def get_phase(self, cell=None, w90=None, kpts=None):
        '''
        Generate real space lattice vector corresponding to the kmesh the phase factors
        '''
        if w90 is None : w90 = self.w90
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        
        kmesh = w90.mp_grid_loc
        a = cell.lattice_vectors()
        Ts = lib.cartesian_prod((np.arange(kmesh[0]), np.arange(kmesh[1]), np.arange(kmesh[2])))
        Ls = np.dot(Ts, a)
        nLs = Ls.shape[0]
        phase = 1/nLs * np.exp(1j*Ls.dot(kpts.T))
        
        return nLs, Ls, phase
    
    def make_supcell(self, cell, nimgs):
        '''
        Make the computational supercell instance used to get the oei and tei in real space
        Note: this orbitals are consistent with the self.Ls vector
        '''
        from pyscf.pbc.tools import pbc
        supcell = cell.copy()
        a = cell.lattice_vectors()
        Ts = lib.cartesian_prod((np.arange(nimgs[0])-nimgs[0]//2,
                        np.arange(nimgs[1])-nimgs[1]//2,
                        np.arange(nimgs[2])-nimgs[2]//2))
        Ls = np.dot(Ts, a)
        symbs = [atom[0] for atom in cell._atom] * len(Ls)
        coords = Ls.reshape(-1,1,3) + cell.atom_coords()
        supcell.atom = list(zip(symbs, coords.reshape(-1,3)))
        supcell.unit = 'B'
        supcell.a = np.einsum('i,ij->ij', nimgs, a)
        supcell.mesh = np.array([nimgs[0]*cell.mesh[0],
                                 nimgs[1]*cell.mesh[1],
                                 nimgs[2]*cell.mesh[2]])
        supcell.build(False, False, verbose=0)
        supcell.verbose = cell.verbose
        return supcell        

    def get_WFs(self, w90):
        '''
        Compute the Wannier functions at the reference cell in the basis of local Gaussian
        '''
        
        ao2lo = []
        for kpt in range(self.nkpts):
            mo_included = w90.mo_coeff_kpts[kpt][:,w90.band_included_list]
            mo_in_window = w90.lwindow[kpt]         
            C_opt = mo_included[:,mo_in_window].dot(w90.U_matrix_opt[kpt].T)              
            ao2lo.append(C_opt.dot(w90.U_matrix[kpt].T))        
            
        ao2lo = np.asarray(ao2lo, dtype=np.complex128)
        
        #TODO: for debugging purposes, need to be removed
        WFs = 0.0
        for kpt in range(self.nkpts):
            WFs += ao2lo[kpt]
            
        #WFs = libdmet.iFFT1e(self.tmap, self.phase, ao2lo)
        
        # Check of WFs are real     
        assert WFs.imag.max() < 1.e-7, 'WFs are not real'
        
        return ao2lo, WFs.real
        
    def get_ao2eo(self, emb_orbs, Norb_in_imp):
        '''
        Get the transformation matrix from AO to EO
        '''
        emb_orbs_Ls = emb_orbs.reshape(self.nLs, self.nactorbs, Norb_in_imp)
        lo2eo = np.einsum('kR, Rim -> kim', self.phase.conj().T, emb_orbs_Ls) 
        ao2eo = np.einsum('kim, kmj -> kij', self.ao2lo, lo2eo) 
        return ao2eo

    def to_kspace(self, M):
        '''
        Transform an one-electron matrix M_{pq}(L) to the k-space
        '''
        return libdmet.FT1e(self.nkpts, self.kpts, self.nLs, self.Ls, M)  
        
    def to_Ls(self, Mk):
        '''
        Transform an one-electron matrix M_{pq}(k) to the L-space
        '''
        if Mk.ndim == 2: Mk = npasarray([Mk for kpt in range(self.nkpts)])
        return libdmet.iFFT1e(self.tmap, self.phase, Mk).real       
        
    def to_Ls_sparse(self, Mat_kpt, kpt):
        '''
        Transform an one-electron matrix M_{pq}(k) to the L-space
        '''
        return libdmet.iFT1e_sparse(self.nkpts, kpt, self.nLs, self.Ls, Mat_kpt).real     

    def to_Ls2e(self, Mijk, kconserv):
        '''
        Transform an one-electron matrix M_{pqrs}(ijkl) to the L-space
        Note: The momentum conservation requires: j - i + l - k = 0
        '''
        return libdmet.iFFT2e(self.tmap, self.phase, kconserv, Mijk).real           

    def to_local(self, Mk, ao2lo):
        '''
        Transform an one-electron operator M_{pq}(k) in the ao basis to the local basis
        '''      
        loc_Mk = np.asarray([reduce(np.dot, (ao2lo[kpt].T.conj(), Mk[kpt], ao2lo[kpt])) for kpt in range(self.nkpts)])
        return loc_Mk

    def get_tei_kpts(self, kconserv, ao2lo):
        '''
        Get TEI at sampled k-point g_pqrs^{ijkl}
        Note:
            - the momentum conservation requires: j - i + l - k = G, G = 0 if i,j,k,l \subset the reciprocal unit cell
        '''
        
        TEIijk = np.zeros([self.nkpts,self.nkpts,self.nkpts,self.nactorbs,self.nactorbs,self.nactorbs,self.nactorbs], dtype = np.complex128)
        for i in range(self.nkpts):
            for j in range(self.nkpts):
                for k in range(self.nkpts):            
                    l = kconserv[i,j,k]    
                    ki, COi = self.kpts[i], ao2lo[i]
                    kj, COj = self.kpts[j], ao2lo[j]
                    kk, COk = self.kpts[k], ao2lo[k]
                    kl, COl = self.kpts[l], ao2lo[l]                
                    TEI = self.kmf.with_df.ao2mo([COi,COj,COk,COl], [ki,kj,kk,kl], compact = False)
                    TEIijk[i,j,k] = TEI.reshape(self.nactorbs,self.nactorbs,self.nactorbs,self.nactorbs)
                    
        return TEIijk
         
        
    def make_tmap(self, kmesh):  
        '''Exploring translational symmetry
           TODO: I have to call it a translational map now. Should have better name and better algorithm  
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