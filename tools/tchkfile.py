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

from pyscf.lib.chkfile import save, load
    
def symmetrize_kmf(cell, kmf, kmesh):
    '''
        This function creates an equivalent Brillouin zone of the kmesh using in kmf calculation
        and also makes sure: C(k) = C(-k)_{*} with C is the orbital coefficient
        why?
        Due to the fact that, the eigenvectors at k C(k) is not equal to the C(-k)_{*}.
        That means the kmesh symmetry is not strictly employed in PBC-PySCF.
        This is not a problem for a HF or KS-DFT calculation, howevere, it is a big trouble for 
        calculation that requires this strict symmetry, for example, FFT of crystalline orbitals to get Wannier functions
        in Wannier90 calculations.
        
        Attributes: kmf can be either a real kmf wavefunction or a saved one from the save_kmf function.
        
        TODO: 1/ To enforce k-point symmetry in PBC-PySCF. 2/ what about an even number of k point
    '''   
    
    kmf.kpts = cell.make_kpts(kmesh, wrap_around=True)
    for i in range(kmf.kpts.shape[0]-1):
        for j in range(i+1): 
            if abs(kmf.kpts[i+1] + kmf.kpts[j]).sum() < 1.e-10: 
                kmf.mo_coeff_kpts[i+1]  = kmf.mo_coeff_kpts[j].conj()
                kmf.mo_energy_kpts[i+1] = kmf.mo_energy_kpts[j]               
                break 
                
    return kmf    
 
def save_kmf(kmf, chkfile):
    if kmf.exxdiv == None: 
        exxdiv = 'None'
    else:
        exxdiv      = kmf.exxdiv
    max_memory      = kmf.max_memory
    e_tot           = kmf.e_tot
    kpts            = kmf.kpts
    mo_occ_kpts     = kmf.mo_occ_kpts
    mo_energy_kpts  = kmf.mo_energy_kpts
    mo_coeff_kpts   = kmf.mo_coeff_kpts
    get_fock        = kmf.get_fock()
    make_rdm1       = kmf.make_rdm1()
    
    kmf_dic = { 'exxdiv'            : exxdiv, 
                'max_memory'        : max_memory,
                'e_tot'             : e_tot,
                'kpts'              : kpts,
                'mo_occ_kpts'       : mo_occ_kpts,
                'mo_energy_kpts'    : mo_energy_kpts,
                'mo_coeff_kpts'     : mo_coeff_kpts,
                'get_fock'          : get_fock,
                'make_rdm1'         : make_rdm1}
                
    save(chkfile, 'scf', kmf_dic)
    
def load_kmf(cell, kmf, kmesh, chkfile, max_memory=4000):
    '''
        Save a kmf object
    '''
    
    save_kmf = load(chkfile, 'scf')
     
    class fake_kmf:
        def __init__(self, save_kmf):  
            if save_kmf['exxdiv'] == 'None': 
                self.exxdiv     = None 
                kmf.exxdiv      = None 
            else:
                self.exxdiv     = save_kmf['exxdiv']    
                kmf.exxdiv      = save_kmf['exxdiv']  
                
            self.e_tot          = save_kmf['e_tot']
            self.kpts           = save_kmf['kpts']
            self.mo_occ_kpts    = save_kmf['mo_occ_kpts']
            self.mo_energy_kpts = save_kmf['mo_energy_kpts']
            self.mo_coeff_kpts  = save_kmf['mo_coeff_kpts']
            self.get_fock   = lambda *arg: save_kmf['get_fock']
            self.make_rdm1  = lambda *arg: save_kmf['make_rdm1']            
            self.eig        = lambda *arg, **kwargs: kmf.eig(*arg, **kwargs)
            self.get_ovlp   = lambda *arg, **kwargs: kmf.get_ovlp(*arg, **kwargs)
            self.get_hcore  = lambda *arg, **kwargs: kmf.get_hcore(*arg, **kwargs)
            self.get_jk     = lambda *arg, **kwargs: kmf.get_jk(*arg, **kwargs)
            self.get_veff   = lambda *arg, **kwargs: kmf.get_veff(*arg, **kwargs)
            self.get_bands  = lambda *arg, **kwargs: kmf.get_bands(*arg, **kwargs) 
            if hasattr(kmf,'max_memory'):
                self.max_memory = kmf.max_memory
            else:
                self.max_memory = max_memory
            self.with_df    = kmf.with_df 
            
    final_kmf = fake_kmf(save_kmf)
    
    return final_kmf
    
def save_w90(w90, chkfile):
    mp_grid_loc        = w90.mp_grid_loc
    exclude_bands      = w90.exclude_bands
    mp_grid_loc        = w90.mp_grid_loc
    mo_coeff_kpts      = w90.mo_coeff_kpts
    band_included_list = w90.band_included_list
    lwindow            = w90.lwindow
    U_matrix_opt       = w90.U_matrix_opt
    U_matrix           = w90.U_matrix
    w90_dic = { 'mp_grid_loc'           : mp_grid_loc, 
                'exclude_bands'         : exclude_bands,
                'mp_grid_loc'           : mp_grid_loc,
                'mo_coeff_kpts'         : mo_coeff_kpts,
                'band_included_list'    : band_included_list,
                'lwindow'               : lwindow,
                'U_matrix_opt'          : U_matrix_opt,
                'U_matrix'              : U_matrix}
                
    save(chkfile, 'w90', w90_dic)
    
def load_w90(chkfile):
    save_w90 = load(chkfile, 'w90')
    class fake_w90:
        def __init__(self, save_w90):
            self.mp_grid_loc            = save_w90['mp_grid_loc']
            self.exclude_bands          = save_w90['exclude_bands']
            self.mp_grid_loc            = save_w90['mp_grid_loc']
            self.mo_coeff_kpts          = save_w90['mo_coeff_kpts']
            self.band_included_list     = save_w90['band_included_list']
            self.lwindow                = save_w90['lwindow']
            self.U_matrix_opt           = save_w90['U_matrix_opt']
            self.U_matrix               = save_w90['U_matrix']   
            
    w90 = fake_w90(save_w90)
    return w90    

def save_pdmet_int(integral, chkfile):
    CO                  = integral.CO
    WFs                 = integral.WFs    
    e_core              = integral.e_core
    coreDM_kpts         = integral.coreDM_kpts
    loc_actOEI_kpts     = integral.loc_actOEI_kpts
    loc_actOEI_Ls       = integral.loc_actOEI_Ls
    loc_actTEI_kpts     = integral.loc_actTEI_kpts    
    loc_actTEI_Ls       = integral.loc_actTEI_Ls
    loc_actFOCK_kpts    = integral.loc_actFOCK_kpts    
    loc_actFOCK_Ls      = integral.loc_actFOCK_Ls 
    loc_actVHF_kpts     = integral.loc_actVHF_kpts 
    
    integral_dic = {'CO'              : CO, 
                'WFs'                 : WFs,
                'e_core'              : e_core,  
                'coreDM_kpts'         : coreDM_kpts,                  
                'loc_actOEI_kpts'     : loc_actOEI_kpts,
                'loc_actOEI_Ls'       : loc_actOEI_Ls,
                'loc_actTEI_kpts'     : loc_actTEI_kpts,    
                'loc_actTEI_Ls'       : loc_actTEI_Ls,
                'loc_actFOCK_kpts'    : loc_actFOCK_kpts,    
                'loc_actFOCK_Ls'      : loc_actFOCK_Ls,
                'loc_actVHF_kpts'     : loc_actVHF_kpts}
                
    save(chkfile, 'integral', integral_dic)
    
def load_pdmet_int(chkfile):
    save_pdmet = load(chkfile, 'integral')
    class fake_pdmet:
        def __init__(self, save_pdmet):
            self.CO                  = save_pdmet['CO']
            self.WFs                 = save_pdmet['WFs']    
            self.e_core              = save_pdmet['e_core']
            self.coreDM_kpts         = save_pdmet['coreDM_kpts']            
            self.loc_actOEI_kpts     = save_pdmet['loc_actOEI_kpts']
            self.loc_actOEI_Ls       = save_pdmet['loc_actOEI_Ls']
            self.loc_actTEI_kpts     = save_pdmet['loc_actTEI_kpts']    
            self.loc_actTEI_Ls       = save_pdmet['loc_actTEI_Ls']
            self.loc_actFOCK_kpts    = save_pdmet['loc_actFOCK_kpts']    
            self.loc_actFOCK_Ls      = save_pdmet['loc_actFOCK_Ls'] 
            self.loc_actVHF_kpts     = save_pdmet['loc_actVHF_kpts'] 
            
    pdmet = fake_pdmet(save_pdmet)
    return pdmet     
    
def save_pdmet(pdmet, chkfile):
    solver        = pdmet.solver
    chempot       = pdmet.chempot      
    uvec          = pdmet.uvec 
    umat          = pdmet.umat     
    kmesh_sym     = 'False'
    if pdmet.kmesh_sym: kmesh_sym   = 'True'    
    emb_orbs      = pdmet.emb_orbs
    env_orbs      = pdmet.env_orbs
    mf_mo         = pdmet.qcsolver.mf.mo_coeff
    core1RDMloc   = pdmet.core1RDM_local 
    actv1RDMloc   = pdmet.emb_1RDM 
    
    if pdmet.solver in ['CASCI', 'CASSCF', 'DMRG-CI', 'DMRG-SCF']:
        mc_mo       = pdmet.qcsolver.mo
        mc_mo_nat   = pdmet.qcsolver.mo_nat 
        
    pdmet_dic = {'solver'           : solver, 
                 'chempot'          : chempot,                
                 'uvec'             : uvec,
                 'umat'             : umat,                 
                 'kmesh_sym'        : kmesh_sym,
                 'emb_orbs'         : emb_orbs,
                 'env_orbs'         : env_orbs,                 
                 'mf_mo'            : mf_mo,
                 'core1RDMloc'      : core1RDMloc,
                 'actv1RDMloc'      : actv1RDMloc}
                 
    if pdmet.solver in ['CASCI', 'CASSCF', 'DMRG-CI', 'DMRG-SCF']:
        pdmet_dic['mc_mo']      = mc_mo
        pdmet_dic['mc_mo_nat']  = mc_mo_nat   
                 
    save(chkfile, 'pdmet', pdmet_dic)    
    
def load_pdmet(chkfile):
    save_pdmet  = load(chkfile, 'pdmet')
    
    class fake_pdmet:
        def __init__(self, save_pdmet):
            self.solver      = None    
            self.chempot     = 0
            self.uvec        = False 
            self.umat        = False         
            self.kmesh_sym   = None 
            self.emb_orbs    = None    
            self.env_orbs    = None             
            self.mf_mo       = None
            self.mc_mo       = None     
            self.mc_mo_nat   = None   
            if save_pdmet is not None: 
                self.solver      = save_pdmet['solver']            
                self.chempot     = save_pdmet['chempot']
                self.uvec        = save_pdmet['uvec']
                self.umat        = save_pdmet['umat']                  
                self.kmesh_sym   = False
                if save_pdmet['kmesh_sym'] == 'True': self.kmesh_sym  = True
                self.emb_orbs    = save_pdmet['emb_orbs']              
                self.env_orbs    = save_pdmet['env_orbs']                   
                self.mf_mo       = save_pdmet['mf_mo'] 
                self.core1RDMloc = save_pdmet['core1RDMloc']          
                self.actv1RDMloc = save_pdmet['actv1RDMloc'] 
                if self.solver in ['CASCI', 'CASSCF', 'DMRG-CI', 'DMRG-SCF']:              
                    self.mc_mo          = save_pdmet['mc_mo']                   
                    self.mc_mo_nat      = save_pdmet['mc_mo_nat']                     

    pdmet = fake_pdmet(save_pdmet)
    
    return pdmet                    