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
from pDMET.pdmet import helper
from pDMET.lib.build import libdmet

class WF:
    def __init__(self, cell, kmf, w90, chkfile = None):
        '''
        Prepare the Wannier functions, transform OEI and TEI to the real-space representation
        Args:
            kmf        : a k-dependent mean-field wf
            w90        : a converged wannier90 object
        '''        
        
        # Collect cell and kmf object information
        self.spin = cell.spin        
        self.e_tot = kmf.e_tot
        self.w90 = w90
        self.kmf = kmf   
        self.ovlp = self.kmf.get_ovlp()
        self.kpts = kmf.kpts
        self.nkpts = kmf.kpts.shape[0]    
        self.nao = cell.nao_nr()
        
        # The k-point number has to be odd, since the fragment is assumed to be in the middle of the supercell
        assert (np.asarray([kpt%2 == 0 for kpt in w90.mp_grid_loc]).all() == False)
        nimgs = [kpt//2 for kpt in w90.mp_grid_loc]    
        self.Ls = cell.get_lattice_Ls(nimgs)
        self.nLs = self.Ls.shape[0]
        self.phase = np.exp(1j*self.Ls.dot(self.kpts.T))
        self.tmap = self.make_tmap(w90.mp_grid_loc)
        assert self.nLs == self.nkpts
        
        
        # Active part info
        self.active = np.zeros([cell.nao_nr()], dtype=int)

        for orb in range(cell.nao_nr()):
            if (orb+1) not in w90.exclude_bands: self.active[orb] = 1
        self.nactorbs = np.sum(self.active)    
        self.norbs = self.nkpts * self.nactorbs
        self.nactelecs = np.int32(cell.nelectron - np.sum(kmf.mo_occ_kpts[0][self.active==0]))        
        self.nelec = self.nkpts * self.nactelecs
        
        #-------------------------------------------------------------
        # Construct the effective Hamiltonian due to the frozen core  | 
        #-------------------------------------------------------------    
        chkfile_exist = None     
        if chkfile != None: chkfile_exist = tunix.check_exist(chkfile+'_int')
        
        if chkfile_exist == None or chkfile_exist == False:
            self.CO, self.WFs = self.make_WFs(self.w90)    # WFs basis in k- and L- space
            print('-> 1e integrals ...') 
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
            if self.kmf.exxdiv != None: raise Exception('The pDMET has not been developed for the HF with exxdiv != None')  
            # if self.kmf.exxdiv == 'ewald': actOEI_kpts += self.exxdiv_ewald(cell) 
            # TODO: if self.kmf.exxdiv != None, consider to run two SCF (one with and one without exx treatment
            # to get the finite correction, see https://github.com/pyscf/pyscf/issues/250    
            
            self.loc_actOEI_kpts = self.to_local(actOEI_kpts, self.CO)
            self.loc_actOEI_Ls = self.to_Ls(self.loc_actOEI_kpts)
            print('-> 1e integrals ... done')      
            
            # 2e integral for the active part
            print('-> 2e integrals ...') 
            from pyscf.pbc.tools import pbc as pbctools
            kconserv = pbctools.get_kconserv(cell, self.kpts)
            print(' Computing local k-space TEI')             
            self.loc_actTEI_kpts = self.get_tei_kpts(kconserv, self.CO)
            print(' Transforming k-space TEI to real space')              
            self.loc_actTEI_Ls = self.to_Ls2e(self.loc_actTEI_kpts, kconserv) 
            print('-> 2e integrals ... done') 
            
            # Fock for the active part  
            print('-> Fock matrix  ...')            
            fullfock_kpts = kmf.get_fock()            
            self.loc_actFOCK_kpts = self.to_local(fullfock_kpts, self.CO)
            self.loc_actVHF_kpts = self.loc_actFOCK_kpts - self.loc_actOEI_kpts         
            self.loc_actFOCK_Ls = self.to_Ls(self.loc_actFOCK_kpts)         
            print('-> Fock matrix  ... done') 
                 
            # Save integrals to chkfile:            
            if chkfile_exist == False:
                tchkfile.save_pdmet_int(self, chkfile+'_int')
                print('-> Chkfile saving ... done')                 
                        
        elif chkfile_exist == True:
            print('-> Chkfile loading ...')
            savepdmet = tchkfile.load_pdmet_int(chkfile+'_int')
            self.CO               = savepdmet.CO
            self.WFs              = savepdmet.WFs    
            self.e_core           = savepdmet.e_core
            self.coreDM_kpts      = savepdmet.coreDM_kpts
            self.loc_actOEI_kpts  = savepdmet.loc_actOEI_kpts
            self.loc_actOEI_Ls    = savepdmet.loc_actOEI_Ls
            self.loc_actTEI_kpts  = savepdmet.loc_actTEI_kpts   
            self.loc_actTEI_Ls    = savepdmet.loc_actTEI_Ls
            self.loc_actFOCK_kpts = savepdmet.loc_actFOCK_kpts    
            self.loc_actFOCK_Ls   = savepdmet.loc_actFOCK_Ls         
            self.loc_actVHF_kpts  = savepdmet.loc_actVHF_kpts             

        
    def construct_locOED_kpts(self, umat, OEH_type, doSCF=False, verbose=0):
        '''
        Construct MOs/one-electron density matrix at each k-point in the local basis
        with a certain k-independent correlation potential umat
        '''    

        #Two choices for the one-electron Hamiltonian
        if OEH_type == 'OEI':
            OEH_kpts = self.loc_actOEI_kpts + umat
        elif OEH_type == 'FOCK':
            OEH_kpts = self.loc_actFOCK_kpts + umat         
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
            
            if doSCF == True:
                loc_OED = helper.KRHF(OEH_kpts, self.loc_actTEI_kpts, self.nactelecs, self.kpts, loc_OED, verbose=verbose)
        else:
            pass 
            # TODO: contruct RDM for a ROHF wave function            

        return loc_OED
        
    def construct_locOED_Ls(self, umat, OEH_type):
        '''
        Construct MOs/one-electron density matrix dm_{pq}^{0L} at each lattice vector
        with a certain k-independent correlation potential umat
        '''    
    
        loc_OED = self.construct_locOED_kpts(umat, OEH_type)
        loc_OED_Ls = libdmet.iFFT1e(self.tmap, self.phase, loc_OED).real        
        return loc_OED_Ls
        
    def dmet_oei(self, FBEorbs, Norb_in_imp):
        oei = reduce(np.dot,(FBEorbs[:,:Norb_in_imp].T, self.loc_actOEI_Ls, FBEorbs[:,:Norb_in_imp]))        
        return oei

    def dmet_tei(self, FBEorbs, Norb_in_imp):
        tei = ao2mo.incore.full(ao2mo.restore(8, self.loc_actTEI_Ls, self.norbs), FBEorbs[:,:Norb_in_imp], compact=False)
        tei = tei.reshape(Norb_in_imp, Norb_in_imp, Norb_in_imp, Norb_in_imp)
        return tei        

    def dmet_corejk(self, FBEorbs, Norb_in_imp, core1RDM_loc):
        J = np.einsum('pqrs,rs->pq', self.loc_actTEI_Ls, core1RDM_loc)
        K = np.einsum('prqs,rs->pq', self.loc_actTEI_Ls, core1RDM_loc)    
        JK = reduce(np.dot,(FBEorbs[:,:Norb_in_imp].T, J -0.5*K, FBEorbs[:,:Norb_in_imp]))        
        return JK
    
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

    def make_WFs(self, w90):
        '''
        Compute the Wannier functions at the reference cell in the basis of local Gaussian
        '''
        
        CO = []
        for k_id, kpt in enumerate(self.kpts):
            mo_included = w90.mo_coeff_kpts[k_id][:,w90.band_included_list]
            mo_in_window = w90.lwindow[k_id]         
            C_opt = mo_included[:,mo_in_window].dot(w90.U_matrix_opt[k_id].T)              
            CO.append(C_opt.dot(w90.U_matrix[k_id].T))        
            
        CO = np.asarray(CO, dtype=np.complex128)
        WFs = libdmet.iFFT1e(self.tmap, self.phase, CO)
        
        # Check of WFs are real     
        if WFs.imag.max() >= 1.e-7: raise Exception('WFs are not real')
        
        return CO, WFs.real

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

    def to_local(self, Mk, CO):
        '''
        Transform an one-electron operator M_{pq}(k) in the ao basis to the local basis
        '''      
        loc_Mk = np.asarray([reduce(np.dot, (CO[kpt].T.conj(), Mk[kpt], CO[kpt])) for kpt in range(self.nkpts)])
        return loc_Mk

    def get_tei_kpts(self, kconserv, CO):
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
                    ki, COi = self.kpts[i], CO[i]
                    kj, COj = self.kpts[j], CO[j]
                    kk, COk = self.kpts[k], CO[k]
                    kl, COl = self.kpts[l], CO[l]                
                    TEI = self.kmf.with_df.ao2mo([COi,COj,COk,COl], [ki,kj,kk,kl], compact = False)
                    TEIijk[i,j,k] = TEI.reshape(self.nactorbs,self.nactorbs,self.nactorbs,self.nactorbs)
                    
        return TEIijk
        
    def exxdiv_ewald(self, cell):
        '''
        The correction for The Exchange term when khf.exxdiv == 'ewald'
        TODO: testing ...
        '''
        
        from pyscf.pbc.df import df_jk      
        vk_kpts = np.zeros([1, self.nkpts, self.nao, self.nao], dtype = np.complex128)  
        dm_kpts = self.kmf.make_rdm1()
        dms = df_jk._format_dms(dm_kpts, self.kpts)
        df_jk._ewald_exxdiv_for_G0(cell, self.kpts, dms, vk_kpts, kpts_band = None)  
                  
        return -0.25*vk_kpts[0]    
        
    def make_tmap(self, kmesh):  
        '''TODO: write comments '''
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