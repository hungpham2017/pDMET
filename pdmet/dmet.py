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

import datetime
import numpy as np
from pyscf import lib
from scipy import optimize
from functools import reduce
from pdmet import localbasis, qcsolvers, diis, helper, df_hamiltonian
from pdmet.schmidtbasis import get_bath_using_RHF_1RDM, get_bath_using_gamma_RHF_1RDM
from pdmet.tools import tchkfile, tplot, tprint, tunix, misc
from pdmet.lib.build import libdmet
import sys
sys.path.append('/panfs/roc/groups/6/gagliard/phamx494/pyWannier90/src')
import pywannier90


class pDMET:
    def __init__(self, cell, kmf, w90, solver = 'HF',  state_average_mix_=None, nevpt2_spin=None):
        '''
        Args:
            kmf                             : a rhf wave function from pyscf/pbc
            w90                                : a converged wannier90 object
            OEH_type                        : One-electron Hamiltonian used in the bath construction, h(k) = OEH(k) + umat(k) 
            SCmethod                        : 'BFGS'/'CG'/'Newton-CG' self-consistent iteration method, defaut: BFGS
            SC_threshold                    : convergence criteria for correlatiself.e_toton potential, default: 1e-6
            SC_maxcycle                     : maximum cycle for self-consistent iteration, default: 50
            umat                            : correlation potential
            chempot                         : global chemical potential
            emb_corr_1RDM                   : correlated 1RDM from high-level calculations
            emb_orbs                        : a list of the fragment and bath orbitals for each fragment            
        Return:
        
        '''        
        
        tprint.print_header()

        # Chkfiles:
        self.cell = cell          
        self.kmf = kmf
        self.w90 = w90        
        self.kmf_chkfile = None 
        self.w90_chkfile = None 
        
        # Options
        self.OEH_type = 'FOCK' # Options: FOCK/OEI        
        
        # QC Solver    
        solver_list   = ['HF', 'MP2', 'CASCI', 'DMRG-CI', 'CASSCF', 'DMRG-SCF', \
                            'SS-CASSCF', 'SS-DMRG-SCF', 'SA-CASSCF', 'SA-DMRG-SCF', \
                            'FCI', 'DMRG', 'RCCSD', 'RCCSD_T', 'SHCI'
                            ]
        assert solver in solver_list, "Solver options: HF, MP2, CASCI, DMRG-CI, \
                                     CASSCF, DMRG-SCF,SS-CASSCF, SS-DMRG-SCF, SA-CASSCF, SA-DMRG-SCF \
                                     FCI, DMRG, RCCSD, SHCI"
        self.solver   = solver        
        self.e_shift  = None         # Use to fix spin of the wrong state with FCI, hence CASCI/CASSCF solver
        self.use_GDF  = True          # Mostly using for FFTDF where density fitting is not available
        
        # Gamma sampling embedding
        self.impCluster = None
        self._impOrbs_threshold = 1.0
        self._impOrbs_rmlist = None
        self._impOrbs_addlist = None
        self._num_bath = None
        self.nroots = 10
        self.nevpt2_nroots = 10
        self.nevpt2_roots = None
        self.nevpt2_spin = nevpt2_spin
        self.state_average_ = None
        self.state_average_mix_  = None
        if solver in ['CASCI', 'CASSCF', 'SS-CASSCF', 'SS-DMRG-SCF', 'SA-CASSCF', 'SA-DMRG-SCF']:
            self.cas    = None
            self.molist = None  
            if solver in ['SS-CASSCF', 'SS-DMRG-SCF']:
                self.state_specific_ = 0
            elif solver in ['SA-CASSCF', 'SA-DMRG-SCF']:
                if state_average_mix_ is None:
                    self.state_average_ = [0.5, 0.5]
                else:
                    self.state_average_mix_ = state_average_mix_ 
                
        
        # Parameters    
        self.SC_method          = "BFGS"        # BFGS, CG, Newton-CG
        self.SC_threshold       = 1e-4            
        self.SC_maxcycle        = 200
        self.SC_CFtype          = "F" # Options: ['F','diagF', 'FB','diagFB']
        self.alt_CF             = False 
        self.dft_CF             = False
        self.dft_is_kpts        = False
        self.dft_CF_constraint  = 1
        self.dft_HF             = None  
        self.xc                 = None   
        self.xc_omega           = None
        self.damping            = 1.0 # 1.0 means no damping
        self.DIIS               = False       
        self.DIIS_m             = 1   
        self.DIIS_n             = 8  
        
        # DMET Output
        self.state_percent      = None        
        self.twoS               = None
        self.verbose            = 0
        self.max_memory         = 4000  # in MB 
        self.loc_OEH_kpts       = None      
        self.loc_1RDM_kpts      = None        
        self.loc_1RDM_R0        = None
        self.loc_corr_1RDM_R0   = None
        self.baths              = None
        self.emb_corr_1RDM      = None  
        self.emb_orbs           = None
        self.emb_mf_1RDM        = None
        self.e_tot              = 0.       # energy per unit cell     
        self.e_corr             = 0.
        self.nelec_per_cell     = None      
        
        # Others
        self.bath_truncation = True  # if self.truncate = a threshold, then a bath truncation scheme is used          
        self.chkfile         = 'pdmet.chk'    # Save integrals in the WFs basis as well as chem potential and uvec
        self.restart         = False   # Run a calculation using saved chem potential and uvec             
        self._cycle           = 1        
        
    def initialize(self, ERI = None):
        '''
        Prepare the local integrals, correlation/chemical potential    
        '''    
        
        tprint.print_msg("Initializing ...")
        
        # -------------------------------------------------        
        # General initialized attributes 
        self.kmesh = self.w90.mp_grid_loc
        if (self.kmf_chkfile is not None) and hasattr(self.kmf.with_df, '_cderi'):
            self.kmf = tchkfile.load_kmf(self.cell, self.kmf, self.kmesh, self.kmf_chkfile, max_memory=self.max_memory)
            if self.kmf.with_df._cderi == None:
                if tunix.check_exist('gdf.h5'):
                    self.kmf.with_df._cderi = 'gdf.h5'
                else:
                    print("WARNING: Provide density fitting file in initiating kmf object or make sure the saved kmf object is using the same density fitting")
            self._is_ROHF = self.kmf._is_ROHF 
        else:
            from pyscf.pbc import scf
            if isinstance(self.kmf, scf.krohf.KROHF):
                self._is_ROHF = True
            else:
                self._is_ROHF = False
            
        if self.kmf.exxdiv is not None: 
            raise Exception('The pDMET has not been developed for the RHF calculation with exxdiv is not None')
            # TODO: if self.kmf.exxdiv != None, consider to run two SCF (one with and one without exx treatment
            # if self.kmf.exxdiv == 'ewald': actOEI_kpts += self.exxdiv_ewald(cell) 
            # to get the finite correction, see https://github.com/pyscf/pyscf/issues/250   
            
        if self.w90_chkfile is not None:
            self.w90 = tchkfile.load_w90(self.w90, self.w90_chkfile)
        else:
            self.w90 = self.w90 
            
        if self.twoS is None:
            self.twoS = self.cell.spin
            
        else:
            if self.twoS != self.cell.spin:
                tprint.print_msg(" WARNING: the 2S in DMET is different from that of the mean-field wave function. \
                                   Hope you know what you're doing")
                                   
        if self.nevpt2_spin is None:
            self.nevpt2_spin = self.twoS
        
        assert (self.chkfile == None) or isinstance(self.chkfile,str)
        self.kpts = self.kmf.kpts
        self.Nkpts = self.kpts.shape[0]   
        if self.xc is not None:
            self.dft_CF = True
            self.OEH_type = self.xc
            if self.xc == 'RSH-PBE0' and self.xc_omega is None:
                self.xc_omega = 0.2
        else:
            self.dft_CF = False

        # For the Gamma-sampling DMET
        if self.impCluster is not None:
            assert np.prod(self.kmesh) == 1, "impCluster is used only for a Gamma-point sampling calculation"
            self._impOrbs, self._impAtms = misc.make_imp_orbs(self.cell, self.w90, self.impCluster, \
                                    threshold=self._impOrbs_threshold, rm_list=self._impOrbs_rmlist, add_list=self._impOrbs_addlist)
            self.Nimp = np.sum(self._impOrbs)
            self._is_gamma = True
            
            tprint.print_msg("==== Impurity cluster ====")
            tprint.print_msg(" No. of Impurity atoms   : {0}".format(len(self.impCluster)))
            tprint.print_msg(" No. of Impurity orbitals: {0}".format(self.Nimp))
            atom_coords = self.cell.atom_coords() * lib.param.BOHR
            for i, atm in enumerate(self.impCluster):
                symbol = self.cell.atom_symbol(atm - 1)
                x, y, z = atom_coords[atm - 1]
                tprint.print_msg("  {0:3d}  {1:3s}  {2:3.5f} {3:3.5f} {4:3.5f}".format(atm, symbol, x, y, z))
                impAtms = self._impAtms[i].tolist()
                nimpOrbs = len(impAtms)
                impAtms = [nimpOrbs] + impAtms
                tprint.print_msg(("       {:d} Orbitals: " + "{:d} "*nimpOrbs).format(*impAtms))

            tprint.print_msg("==========================")
        else:
            self.Nimp = self.local.nlo     # the whole reference unit cell is the imputity
            self._is_gamma = False
            assert self.twoS ==0 , "ROHF bath is only available for Gamma-sampling calculation"
            
        # Initilize the local space object
        self.local = localbasis.Local(self.cell, self.kmf, self.w90, self._is_ROHF, self.xc_omega)  
        self.e_core = self.local.e_core   
        
        # -------------------------------------------------        
        # The number of bath orbitals depends on whether one does Schmidt decomposition on RHF or ROHF wave function        
        if self._is_ROHF:
            self.bathtype = 'ROHF'          
        else:
            self.bathtype = 'RHF'            
            
        self.Norbs = self.local.nlo * self.Nkpts
        self.Nelec_total    = self.local.nelec_total
        self.Nelec_per_cell = self.local.nelec_per_cell
        self.numPairs = self.Nelec_per_cell // 2 
        
        if self.SC_CFtype in ['diagF', 'diagFB']: 
            self.Nterms = self.Nimp 
        else:            
            self.Nterms = self.Nimp*(self.Nimp + 1) // 2 

        self.mask = self.make_mask(self._is_gamma)  
        if self._is_gamma: 
            self.mask4Gamma = self.mask
        else:
            self.mask4Gamma = None
        self.H1start, self.H1row, self.H1col = self.make_H1(self._is_gamma, self._impOrbs)[1:4]    #Use in the calculation of 1RDM derivative
  
                  
        self.chempot = 0.0
        if self.dft_CF:
            self.uvec = df_hamiltonian.get_init_uvec(self.xc, self.dft_HF)
            self.bounds = df_hamiltonian.get_bounds(self.xc, self.dft_CF_constraint, self.dft_HF)           
        else:
            self.uvec = np.zeros(self.Nterms, dtype=np.float64)           
        self.umat = self.uvec2umat(self.uvec)

        # -------------------------------------------------       
        # Load/initiate chem pot, uvec, umat  
        # TODO: no longer used, Consider to remove this
        self.restart_success = False        
        if self.chkfile is not None and self.restart == True:
            if tunix.check_exist(self.chkfile):
                self.save_pdmet     = tchkfile.load_pdmet(self.chkfile)           
                self.chempot        = self.save_pdmet.chempot
                self.uvec           = self.save_pdmet.uvec
                self.umat           = self.save_pdmet.umat   
                self.emb_corr_1RDM       = self.save_pdmet.actv1RDMloc
                self.emb_orbs       = self.save_pdmet.emb_orbs
                tprint.print_msg("-> Load the pDMET chkfile")
                self.restart_success = True                 
            else:
                tprint.print_msg("-> Cannot load the pDMET chkfile") 
                self.restart_success = False
            
        if self.alt_CF == True: 
            #TODO: debugging the alternative cost function, this will be updated
            pass
        else:
            self.CF = self.cost_func            
            self.CF_grad = self.cost_func_grad  
            
        # Initializing damping procedure and DIIS object          
        if self.DIIS == True:
            assert self.DIIS_m >= 1        
            self._diis = diis.DIIS(self.DIIS_m, self.DIIS_n)   
            
        if self.damping != 1.0:
            assert (0 <= self.damping <= 1.0)          

        # Initializing the QC solver
        if self.nroots > 1:
            if self.state_percent == None: 
                self.state_percent = [1/self.nroots]*self.nroots
            else:
                assert len(self.state_percent) == self.nroots
                assert abs(sum(self.state_percent) - 1.0) < 1.e-10      # The total percent has be 1   
                
        if self.twoS != 0 and self.solver == 'RCCSD': 
            raise Exception('RCCSD solver does not support ROHF wave function')             

        # For FCI, CAS-like solver
        print(qcsolvers.QCsolvers)
        self._SS = 0.5*self.twoS*(0.5*self.twoS + 1)       
        self.qcsolver = qcsolvers.QCsolvers(self.solver, self.twoS, self._is_ROHF, self.e_shift, self.nroots, self.state_percent, verbose=self.verbose, memory=self.max_memory) 
        if self.solver in ['CASCI', 'CASSCF', 'SS-CASSCF', 'SS-DMRG-SCF', 'SA-CASSCF', 'SA-DMRG-SCF']:
            self.qcsolver.cas = self.cas
            self.qcsolver.molist = self.molist 
            if "SS-" in self.solver: 
                assert self.nroots > self.state_specific_, "Increasing the number of roots in the FCI solver"
            if "SA-" in self.solver: 
                self.qcsolver.nroots = len(self.state_average_) 
        if self.nevpt2_roots is not None:
            assert self.nevpt2_nroots >= len(self.nevpt2_roots), "Increasing the number of roots in the FCI solver"
            
        tprint.print_msg("Initializing ... DONE")       
                
    def kernel(self, chempot=0.0):
        '''
        This is the main kernel for DMET calculation.
        It is solving the embedding problem, then returning the total number of electrons per unit cell 
        and updating the schmidt orbitals and 1RDM.
        Args:
            chempot                    : global chemical potential to adjust the number of electrons in the unit cell
        Return:
            nelecs                     : the total number of electrons
        Update the class attributes:
            energy                          : the energy for the unit cell  
            nelec                           : the number of electrons for the unit cell    
            emb_corr_1RDM                   : correlated 1RDM for the unit cell                
        '''            
              
        # Transform the 1e/2e integrals and the JK core constribution to schmidt basis
        if self._is_new_bath == True:
            ao2eo = self.local.get_ao2eo(self.emb_orbs)
            self.emb_OEI  = self.local.get_emb_OEI(ao2eo)
            if self.use_GDF == True:
                self.emb_TEI  = self.local.get_emb_TEI(ao2eo)
            else:
                self.emb_TEI  = self.local.get_TEI(ao2eo)
            self.emb_mf_1RDM = self.local.loc_kpts_to_emb(self.loc_1RDM_kpts, self.emb_orbs)
            self.emb_JK = self.local.get_emb_JK(self.loc_1RDM_kpts, ao2eo)
            self.emb_coreJK = self.local.get_emb_coreJK(self.emb_JK, self.emb_TEI, self.emb_mf_1RDM)         
            
        #TODO: currently, the 1RDM guess is chempot independent
        #emb_guess_1RDM = self.local.get_emb_guess_1RDM(self.emb_FOCK, self.Nelec_in_emb, self.Nimp, chempot)
        emb_guess_1RDM = self.emb_mf_1RDM
        if self._cycle == 1 : 
            tprint.print_msg("   Embedding size: %2d electrons in (%2d impurities + %2d baths )" \
                                                                    % (self.Nelec_in_emb, self.Nimp, self.Nbath))
        
        self.qcsolver.initialize(self.local.e_core, self.emb_OEI, self.emb_TEI, \
                                self.emb_coreJK, emb_guess_1RDM, self.Nimp + self.Nbath, self.Nelec_in_emb, self.Nimp, chempot)
        if self.solver == 'HF':
            e_cell, e_solver, RDM1 = self.qcsolver.HF()
        elif self.solver == 'MP2':
            e_cell, e_solver, RDM1 = self.qcsolver.MP2()
        elif self.solver in ['CASCI']:
            e_cell, e_solver, RDM1 = self.qcsolver.CASCI(nevpt2_roots=self.nevpt2_roots, nevpt2_nroots=self.nevpt2_nroots)      
        elif self.solver in ['DMRG-CI']:
            e_cell, e_solver, RDM1 = self.qcsolver.CASCI(solver = 'CheMPS2', nevpt2_roots=self.nevpt2_roots, nevpt2_nroots=self.nevpt2_nroots)  
        elif self.solver in ['CASSCF']:
            e_cell, e_solver, RDM1 = self.qcsolver.CASSCF(nevpt2_roots=self.nevpt2_roots, nevpt2_nroots=self.nevpt2_nroots, nevpt2_spin=self.nevpt2_spin)     
        elif self.solver in ['DMRG-SCF']:
            e_cell, e_solver, RDM1 = self.qcsolver.CASSCF(solver = 'CheMPS2', nevpt2_roots=self.nevpt2_roots, nevpt2_nroots=self.nevpt2_nroots, nevpt2_spin=self.nevpt2_spin) 
        elif self.solver in ['SS-CASSCF']:
            e_cell, e_solver, RDM1 = self.qcsolver.CASSCF(state_specific_=self.state_specific_, nevpt2_roots=self.nevpt2_roots, nevpt2_nroots=self.nevpt2_nroots, nevpt2_spin=self.nevpt2_spin) 
        elif self.solver in ['SA-CASSCF']:
            e_cell, e_solver, RDM1 = self.qcsolver.CASSCF(state_average_=self.state_average_, state_average_mix_=self.state_average_mix_, nevpt2_roots=self.nevpt2_roots, nevpt2_nroots=self.nevpt2_nroots, nevpt2_spin=self.nevpt2_spin)  
        elif self.solver in ['SS-DMRG-SCF']:
            e_cell, e_solver, RDM1 = self.qcsolver.CASSCF(solver = 'CheMPS2', state_specific_=self.state_specific_, nevpt2_roots=self.nevpt2_roots, nevpt2_nroots=self.nevpt2_nroots)  
        elif self.solver in ['SA-DMRG-SCF']:
            e_cell, e_solver, RDM1 = self.qcsolver.CASSCF(solver = 'CheMPS2', state_average_=self.state_average_, nevpt2_roots=self.nevpt2_roots, nevpt2_nroots=self.nevpt2_nroots) 
        elif self.solver == 'FCI':
            e_cell, e_solver, RDM1 = self.qcsolver.FCI()
        elif self.solver == 'DMRG':
            e_cell, e_solver, RDM1 = self.qcsolver.DMRG()          
        elif self.solver == 'RCCSD':
            e_cell, e_solver, RDM1 = self.qcsolver.RCCSD()  
        elif self.solver == 'RCCSD_T':
            e_cell, e_solver, RDM1 = self.qcsolver.RCCSD_T()              
        elif self.solver == 'SHCI':
            e_cell, e_solver, RDM1 = self.qcsolver.SHCI()              
        
        self.emb_corr_1RDM    = RDM1
        self.loc_corr_1RDM_R0 = lib.einsum('Rim,mn,jn->Rij', self.emb_orbs, RDM1, self.emb_orbs[0].conj())
        
        if not np.isclose(self._SS, self.qcsolver.SS): 
            tprint.print_msg("           WARNING: Spin contamination. Computed <S^2>: %10.8f, Target: %10.8f" % (self.qcsolver.SS, self._SS)) 
            
        # Get the cell energy:
        if self._is_gamma:
            # The Gamma-point calculation assumes one active space, so CASCI-like formular is used to compute the energy
            if self._is_new_bath:
                if np.shape([self.core_orbs])[-1] != 0:
                    ao2core = self.local.get_ao2core(self.core_orbs)
                    lo2core = self.local.get_lo2core(self.core_orbs)
                    core_OEI = self.local.get_core_OEI(ao2core)
                    Nelec_in_core = self.Nelec_total - self.Nelec_in_emb
                    core_1RDM = self.local.get_core_mf_1RDM(lo2core, Nelec_in_core, self.loc_OEH_kpts)
                    loc_core_1RDM = lib.einsum('kim,mn,kjn->kij', lo2core, core_1RDM, lo2core.conj())
                    core_JK = self.local.get_core_JK(ao2core, loc_core_1RDM)
                    core_energy = np.sum((core_OEI + 0.5 * core_JK)* core_1RDM)
                    self.core_energy = core_energy.real
                    self.loc_core_1RDM = loc_core_1RDM.real.reshape(1, self.Norbs, self.Norbs)    
                else:
                    self.core_energy = 0.0
                    self.loc_core_1RDM = 0.0
                    E_core = 0.0
                    
            self.loc_corr_1RDM_R0 += self.loc_core_1RDM
            self.nelec_per_cell = self.Nelec_total
            if self.nevpt2_roots is not None:
                e_CAS, e_CASCI_NEVPT2, t_dm1s, dm1s = e_solver
                self.ss_CASCI = e_CASCI_NEVPT2[:,0]
                e_CASCI = e_CASCI_NEVPT2[:,1] 
                e_NEVPT2 = e_CASCI_NEVPT2[:,2]
                self.e_tot = e_CAS + self.core_energy + self.local.e_core
                self.e_emb = e_CAS 
                self.e_imp = e_cell - self.local.e_core  
                self.e_casci_tot = np.asarray(e_CASCI) + self.core_energy + self.local.e_core
                self.e_nept2_tot = np.asarray(e_NEVPT2) + self.core_energy + self.local.e_core
                self.t_dm1s = t_dm1s
                self.dm1s = dm1s
            else:
                self.e_tot = e_solver + self.core_energy + self.local.e_core
                self.e_emb = e_solver 
                self.e_imp = e_cell - self.local.e_core             
        else:
            self.nelec_per_cell = np.trace(RDM1[:self.Nimp,:self.Nimp])
            self.e_tot = e_cell   
                
        return self.nelec_per_cell
        
    def bath_contruction(self, loc_1RDM_R0, impCluster):
        '''Get the bath orbitals'''
        emb_orbs, core_orbs, Nelec, Nbath = get_bath_using_RHF_1RDM(loc_1RDM_R0, impCluster,  is_ROHF=self._is_ROHF, num_bath=self._num_bath, bath_truncation=self.bath_truncation)
        # self._num_bath is used to keep the no. of baths are the same as in the 1st cycle of SCF
        if self._num_bath is None: self._num_bath = Nbath           
        Nemb = self.Nimp + Nbath
        emb_orbs = emb_orbs.reshape(self.Nkpts, self.local.nlo, Nemb) # NR = Nkpts
        core_orbs = core_orbs.reshape(self.Nkpts, self.local.nlo, self.local.nlo - Nemb)
        Nenv = self.Norbs - Nemb
        Nelec_in_emb = Nelec

        if Nelec_in_emb > self.Nelec_total:
            Nelec_in_emb = self.Nelec_total
        elif self.Nelec_total - Nelec_in_emb > 2 * Nenv:
            Nelec_in_emb = self.Nelec_total - 2 * Nenv
        self._is_new_bath = True
        return emb_orbs, core_orbs, Nbath, Nelec_in_emb

    def check_exact(self, error = 1.e-6):
        '''
        Do one-shot DMET, only the chemical potential is optimized
        '''
        
        tprint.print_msg("--------------------------------------------------------------------")   
        
        if self.dft_CF:
            umat = df_hamiltonian.get_init_uvec(self.xc)
        else:
            umat = 0.
            
        self.loc_OEH_kpts, self.loc_1RDM_kpts, self.loc_1RDM_R0 = self.local.make_loc_1RDM(umat, self.mask4Gamma, OEH_type=self.OEH_type, dft_HF=None)      
        
        self.emb_orbs, self.core_orbs, self.Nbath, self.Nelec_in_emb = self.bath_contruction(self.loc_1RDM_R0, self._impOrbs)
        
        solver = self.solver
        self.solver = 'HF'
        nelec_cell = self.kernel(chempot=0.0)
        self.solver = solver        
        
        diff = abs(self.e_tot - self.kmf.e_tot)
        tprint.print_msg("   E(RHF)        : %12.8f" % (self.kmf.e_tot))
        tprint.print_msg("   E(RHF-DMET)   : %12.8f" % (self.e_tot))
        tprint.print_msg("   |RHF - RHF(DMET)|          : %12.8f" % (diff))
        if diff < error : 
            tprint.print_msg("   HF-in-HF embedding is exact: True")
        else:
             raise Exception('WARNING: HF-in-HF embedding is not exact')                    
        
    def one_shot(self, umat=0.0, proj_DMET=False):
        '''
        Do one-shot DMET, only the chemical potential is optimized
        this function takes umat or loc_1RDM_R0 (p-DMET algorthm)
        '''

        tprint.print_msg("-- One-shot DMET ... starting at %s" % (tunix.current_time()))    
        if self.solver == 'HF' and self.twoS == 0 and not self._is_ROHF:
            tprint.print_msg("   Bath type: %s | QC Solver: %s" % (self.bathtype, 'RHF'))
        elif (self.solver == 'HF' and self.twoS != 0) or (self.solver == 'HF' and self._is_ROHF):        
            tprint.print_msg("   Bath type: %s | QC Solver: %s | 2S = %d" % (self.bathtype, 'ROHF', self.twoS)) 
        elif self.solver == 'RCCSD': 
            tprint.print_msg("   Bath type: %s | QC Solver: %s | 2S = %d" % (self.bathtype, self.solver, self.twoS))        
        else:      
            tprint.print_msg("   Bath type: %s | QC Solver: %s | 2S = %d | Nroots: %d" % (self.bathtype, self.solver, self.twoS, self.nroots))
            
        if self.solver in ['CASCI', 'CASSCF', 'SS-CASSCF', 'SS-DMRG-SCF', 'SA-CASSCF', 'SA-DMRG-SCF']:                
            if self.qcsolver.cas is not None: tprint.print_msg("   Active space     :", self.qcsolver.cas)
            if self.qcsolver.cas is not None: tprint.print_msg("   Active space MOs :", self.qcsolver.molist)
            if "SS-" in self.solver: tprint.print_msg("   State-specific CASSCF using state id :", self.state_specific_)
            if "SA-" in self.solver: tprint.print_msg("   State-average CASSCF with weight :", self.state_average_)
            
        self._cycle = 1 
        if not proj_DMET:
            self.loc_OEH_kpts, self.loc_1RDM_kpts, self.loc_1RDM_R0 = self.local.make_loc_1RDM(umat, self.mask4Gamma, OEH_type=self.OEH_type, dft_HF=self.dft_HF)       
                    
        self.emb_orbs, self.core_orbs, self.Nbath, self.Nelec_in_emb = self.bath_contruction(self.loc_1RDM_R0, self._impOrbs)

        # Optimize the chemical potential
        if self._is_gamma:
            nelec_per_cell_from_embedding = self.kernel(chempot=0.0)
        else:
            self.chempot = optimize.newton(self.nelec_cost_func, self.chempot)
            tprint.print_msg("   No. of electrons per cell : %12.8f" % (self.nelec_per_cell))
            
        if isinstance(self.e_tot, list) or isinstance(self.e_tot, np.ndarray):
            tprint.print_msg("   Energy per cell           : %12.8f" % (self.e_tot[0])) 
            if self.state_average_ is not None:
                for i, e in enumerate(self.e_tot):
                    tprint.print_msg("      State %d weight %7.5f: E = %12.8f" % (i, self.state_average_[i], e))  
            else:
                for i, e in enumerate(self.e_tot):
                    tprint.print_msg("      State %d: E = %12.8f" % (i, e))  
        else:
            tprint.print_msg("   Energy per cell           : %12.8f" % (self.e_tot))   

        if self.nevpt2_roots is not None:
            tprint.print_msg("   NEVPT2 energies for the selected states:") 
            for i, e_nevpt2 in enumerate(self.e_nept2_tot):
                tprint.print_msg("      State %d: E(CASCI) = %12.8f   E(NEVPT2) = %12.8f   <S^2> = %8.6f" % (self.nevpt2_roots[i], self.e_casci_tot[i], e_nevpt2, self.ss_CASCI[i]))  
        
        tprint.print_msg("-- One-shot DMET ... finished at %s" % (tunix.current_time()))
        tprint.print_msg()            
        
    def self_consistent(self, get_band=False, interpolate_band=None):
        '''
        Do self-consistent pDMET
        '''    
           
        tprint.print_msg("--------------------------------------------------------------------")
        tprint.print_msg("- SELF-CONSISTENT DMET CALCULATION ... STARTING -")
        tprint.print_msg("  Convergence criteria")   
        tprint.print_msg("    Threshold :", self.SC_threshold)     
        tprint.print_msg("  Fitting 1-RDM of :", self.SC_CFtype) 
            
        if self.dft_CF: tprint.print_msg("  DF-like cost function:", self.xc)          
        if self.damping != 1.0:        
            tprint.print_msg("  Damping factor   :", self.damping)                
        if self.DIIS:
            tprint.print_msg("  DIIS start at %dth cycle and using %d previous umats" % (self.DIIS_m, self.DIIS_n))  
            
        #------------------------------------#
        #---- SELF-CONSISTENT PROCEDURE ----#      
        #------------------------------------#    
        OEH_kpts, rdm1_kpts, rdm1_R0 = self.local.make_loc_1RDM(self.umat, self.mask4Gamma, OEH_type=self.OEH_type, dft_HF=self.dft_HF)
        for cycle in range(self.SC_maxcycle):
            
            tprint.print_msg("- CYCLE %d:" % (cycle + 1))    
            umat_old = self.umat      
            rdm1_R0_old = rdm1_R0     
            energy_old = self.e_tot
            
            # Do one-shot with each uvec                  
            self.one_shot(umat=self.umat) 
            tprint.print_msg("   + Chemical potential        : %12.8f" % (self.chempot))

            # Optimize uvec to minimize the cost function
            if self.dft_CF:
                result = optimize.minimize(self.CF, self.uvec, method='L-BFGS-B', jac=None, options={'disp': False, 'gtol': 1e-4, 'eps': 1e-8}, bounds=self.bounds)
            else:
                # result = optimize.minimize(self.CF, self.uvec, method=self.SC_method, jac=self.CF_grad, options={'disp': False, 'gtol': 1e-12})
                result = optimize.minimize(self.CF, self.uvec, method = self.SC_method , options = {'disp': False, 'gtol': 1e-6}, tol=1e-4)
                
            if not result.success:         
                tprint.print_msg("     WARNING: Correlation potential is not converged")    
                  
            uvec = result.x
            self.umat = self.uvec2umat(uvec)   
            
            # Construct new global 1RDM in k-space
            global_corr_1RDM = self.local.get_1RDM_Rs(self.loc_corr_1RDM_R0)
            global_corr_1RDM = 0.5*(global_corr_1RDM.T.conj() + global_corr_1RDM) 
            if not self._is_gamma:
                loc_1RDM_R0 = global_corr_1RDM[:,:self.Nimp].reshape(self.Nkpts, self.Nimp, self.Nimp)
            else:
                loc_1RDM_R0 = global_corr_1RDM
            rdm1_R0 =  loc_1RDM_R0
            
            # Remove arbitrary chemical potential shifts
            if not self.dft_CF:
                self.umat = self.umat - np.eye(self.umat.shape[0])*np.average(np.diag(self.umat))

            if self.verbose > 0:
                tprint.print_msg("   + Correlation potential vector    : ", uvec)
                    
            umat_diff = umat_old - self.umat  
            rdm_diff  = rdm1_R0_old - rdm1_R0  
            energy_diff = self.e_tot - energy_old
            if self.state_average_ is not None:
                energy_diff = self.e_tot - energy_old
                energy_diff = np.sum(energy_diff * np.asarray(self.state_average_))
            norm_u    = np.linalg.norm(umat_diff)             
            norm_rdm  = np.linalg.norm(rdm_diff) 
            
            tprint.print_msg("   + Cost function             : %20.15f" % (result.fun))
            tprint.print_msg("   + 2-norm of umat difference : %20.15f" % (norm_u)) 
            tprint.print_msg("   + 2-norm of rdm1 difference : %20.15f" % (norm_rdm))  
            tprint.print_msg("   + Energy difference         : %20.15f" % (energy_diff))  
            
            # Export band structure at every cycle:
            if get_band:
                band = self.get_bands()
                pywannier90.save_kmf(band, str(self.solver) + '_band_cyc_' + str(cycle + 1))               
                
            # DEBUG 
            if interpolate_band is not None:
                frac_kpts = interpolate_band
                bands = self.interpolate_band(frac_kpts)          
            
            # Check convergence of 1-RDM            
            if self.dft_CF:
                if (norm_rdm <= self.SC_threshold): break
            elif (norm_u <= self.SC_threshold): 
                break
            
            if self.DIIS == True:  
                self.umat = self._diis.update(cycle, self.umat, umat_diff)            
                
            if self.damping != 1.0:                
                self.umat = (1.0 - self.damping)*umat_old + self.damping*self.umat        

            self.uvec =  self.umat2uvec(self.umat)    
            tprint.print_msg()            
            
        tprint.print_msg("- SELF-CONSISTENT DMET CALCULATION ... DONE -")
        tprint.print_msg("--------------------------------------------------------------------")            
        

    def projected_DMET(self, get_band=False):
        '''
        Do projected DMET
        '''    
           
        tprint.print_msg("--------------------------------------------------------------------")
        tprint.print_msg("- p-DMET CALCULATION ... STARTING -")
        tprint.print_msg("  Convergence criteria")   
        tprint.print_msg("    Threshold :", self.SC_threshold)    
        tprint.print_msg("  Fitting 1-RDM of :", self.SC_CFtype)         
        
        if self.damping != 1.0:        
            tprint.print_msg("  Damping factor   :", self.damping)                
        if self.DIIS:
            tprint.print_msg("  DIIS start at %dth cycle and using %d previous umats" % (self.DIIS_m, self.DIIS_n))         
                                              
        #------------------------------------#
        #---- SELF-CONSISTENT PROCEDURE ----#     
        #------------------------------------#    
        self.loc_OEH_kpts, self.loc_1RDM_kpts, self.loc_1RDM_R0 = self.local.make_loc_1RDM(0.0, self.mask4Gamma, OEH_type=self.OEH_type, dft_HF=self.dft_HF)
        global_corr_1RDM = self.local.k_to_R(self.loc_1RDM_kpts)
        for cycle in range(self.SC_maxcycle):
            tprint.print_msg("- CYCLE %d:" % (cycle + 1))  
            global_corr_1RDM_old = global_corr_1RDM
            self.one_shot(proj_DMET=True) 
           
            if not self._is_gamma:
                tprint.print_msg("   + Chemical potential        : %12.8f" % (self.chempot))
            
            # Construct new global 1RDM in k-space
            global_corr_1RDM = self.local.get_1RDM_Rs(self.loc_corr_1RDM_R0)
            global_corr_1RDM = 0.5*(global_corr_1RDM.T.conj() + global_corr_1RDM) 
            global_corr_1RDM_residual  = global_corr_1RDM_old - global_corr_1RDM             
            norm_1RDM  = np.linalg.norm(global_corr_1RDM_residual) / self.kpts.shape[0]
            tprint.print_msg("   + 2-norm of rdm1 difference : %20.15f" % (norm_1RDM))  
            
            if get_band == True:
                band = self.get_bands()
                pywannier90.save_kmf(band, str(self.solver) + '_band_cyc_' + str(cycle + 1))

            # Check convergence of 1-RDM            
            if norm_1RDM <= self.SC_threshold: 
                break
 
            if self.DIIS == True:  
                global_corr_1RDM = self._diis.update(cycle, global_corr_1RDM, global_corr_1RDM_residual)   
                
            if self.damping != 1.0:
                global_corr_1RDM = self.damping*global_corr_1RDM + (1-self.damping)*global_corr_1RDM_old
                       
            # Construct new mean-field 1-RDM from the correlated one
            eigenvals, eigenvecs = np.linalg.eigh(global_corr_1RDM)
            idx = (-eigenvals).argsort()
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:,idx]
            num_pairs = self.Nelec_total//2
            global_mf_1RDM = 2 * eigenvecs[:,:num_pairs].dot(eigenvecs[:,:num_pairs].T)
            if self._is_gamma:
                self.loc_1RDM_R0 = global_mf_1RDM.reshape(1, self.Norbs, self.Norbs) + self.loc_core_1RDM
            else:
                self.loc_1RDM_R0 = global_mf_1RDM[:,:self.Nimp].reshape(self.Nkpts, self.Nimp, self.Nimp)
                
            self.loc_1RDM_kpts = self.local.R0_to_k(self.loc_1RDM_R0)        
            tprint.print_msg()            
            
        tprint.print_msg("- p-DMET CALCULATION ... DONE -")
        tprint.print_msg("--------------------------------------------------------------------")        
        
    def nelec_cost_func(self, chempot):
        '''
        The different in the correct number of electrons (provided) and the calculated one 
        '''
        
        nelec_per_cell_from_embedding = self.kernel(chempot)
        self._is_new_bath = False
        tprint.print_msg("     Cycle %2d. Chem potential: %12.8f | Elec/cell = %12.8f | <S^2> = %12.8f" % \
                                                        (self._cycle, chempot, nelec_per_cell_from_embedding, self.qcsolver.SS))                                                                               
        self._cycle += 1
        return nelec_per_cell_from_embedding - self.Nelec_per_cell

    def cost_func(self, uvec):
        '''
        Cost function: \mathbf{CF}(u) = \mathbf{\Sigma}_{rs} (D^{mf}_{rs}(u) - D^{corr}_{rs})^2
        where D^{mf} and D^{corr} are the mean-field and correlated 1-RDM, respectively.
        and D^{mf} = \mathbf{FT}(D^{mf}(k))
        '''
        rdm_diff = self.get_rdm_diff(uvec)
        cost = np.power(rdm_diff, 2).sum()  
        return cost
        

    def cost_func_grad(self, uvec):
        '''
        Analytical derivative of the cost function,
        deriv(CF(u)) = Sum^x [Sum_{rs} (2 * rdm_diff^x_{rs}(u) * deriv(rdm_diff^x_{rs}(u))]
        ref: J. Chem. Theory Comput. 2016, 12, 2706−2719
        '''
        rdm_diff = self.get_rdm_diff(uvec)   
        rdm_diff_grad = self.rdm_diff_grad(uvec)  
        CF_grad = np.zeros(self.Nterms)
        
        for u in range(self.Nterms):
            CF_grad[u] = np.sum(2 * rdm_diff * rdm_diff_grad[u])
        return CF_grad
        
    def get_rdm_diff(self, uvec):
        '''
        Calculating the different between mf-1RDM (transformed in schmidt basis) and correlated-1RDM for the unit cell
        Args:
            uvec            : the correlation potential vector
        Return:
            error            : an array of errors for the unit cell.
        '''

        loc_OEH_kpts, loc_1RDM_kpts, loc_1RDM_R0 = self.local.make_loc_1RDM(self.uvec2umat(uvec), self.mask4Gamma, OEH_type=self.OEH_type, dft_HF=self.dft_HF)
        if self.SC_CFtype in ['F', 'diagF']:        
            mf_1RDM = self.local.loc_kpts_to_emb(loc_1RDM_kpts, self.emb_orbs[:,:,:self.Nimp])
            corr_1RDM = self.emb_corr_1RDM[:self.Nimp,:self.Nimp]              
        elif self.SC_CFtype in ['FB', 'diagFB']:  
            mf_1RDM = self.local.loc_kpts_to_emb(loc_1RDM_kpts, self.emb_orbs)
            corr_1RDM = self.emb_corr_1RDM    
            
        error = mf_1RDM - corr_1RDM
        if self.SC_CFtype in ['diagF', 'diagFB']: error = np.diag(error)      
        
        return error
        
    def rdm_diff_grad(self, uvec):
        '''
        Compute the rdm_diff gradient
        Args:
            uvec            : the correlation potential vector
        Return:
            the_gradient    : a list with the size of the number of u values in uvec
                              Each element of this list is an array of derivative corresponding to each rs.
                             
        '''
        
        RDM_deriv_kpts = self.construct_1RDM_response_kpts(uvec)
        the_gradient = []    
        for u in range(self.Nterms):
            RDM_deriv_R0 = self.local.k_to_R0(RDM_deriv_kpts[:,u,:,:])    # Transform RDM_deriv from k-space to the reference cell         
            if self.SC_CFtype in ['F','diagF']: 
                emb_error_deriv = self.local.loc_kpts_to_emb(RDM_deriv_kpts[:,u,:,:], self.emb_orbs[:,:,:self.Nimp])
            elif self.SC_CFtype in ['FB','diagFB']:
                emb_error_deriv = self.local.loc_kpts_to_emb(RDM_deriv_kpts[:,u,:,:], self.emb_orbs) 
            if self.SC_CFtype in ['diagF', 'diagFB']: emb_error_deriv = np.diag(emb_error_deriv)
            the_gradient.append(emb_error_deriv)

        return np.asarray(the_gradient)       

    def glob_cost_func(self, uvec):
        '''TODO write it 
        '''
        rdm_diff = self.get_glob_rdm_diff(uvec)
        cost = np.power(rdm_diff, 2).sum()
        return cost 
        
    def glob_cost_func_grad(self, uvec):
        '''TODO
        '''
        rdm_diff = self.get_glob_rdm_diff(uvec)   
        rdm_diff_grad = self.glob_rdm_diff_grad(uvec)  
        CF_grad = np.zeros(self.Nterms)
        
        for u in range(self.Nterms):
            CF_grad[u] = np.sum(2 * rdm_diff * rdm_diff_grad[u])
        return CF_grad
        
    def get_glob_rdm_diff(self, uvec):
        '''
        Calculating the different between mf-1RDM (transformed in schmidt basis) and correlated-1RDM for the unit cell
        Args:
            uvec            : the correlation potential vector
        Return:
            error            : an array of errors for the unit cell.
        '''                     
        loc_OEH_kpts, loc_1RDM_kpts, loc_1RDM_R0 = self.local.make_loc_1RDM(self.uvec2umat(uvec), self.mask4Gamma, OEH_type=self.OEH_type, dft_HF=self.dft_HF)
        error = loc_1RDM_R0 - self.loc_corr_1RDM_R0
        return error
        

    def glob_rdm_diff_grad(self, uvec):
        '''
        Compute the rdm_diff gradient
        Args:
            uvec            : the correlation potential vector
        Return:
            the_gradient    : a list with the size of the number of u values in uvec
                              Each element of this list is an array of derivative corresponding to each rs.
                             
        '''
        
        RDM_deriv_kpts = self.construct_1RDM_response_kpts(uvec)
        the_gradient = []    

        for u in range(self.Nterms):
            RDM_deriv_R0 = self.local.k_to_R0(RDM_deriv_kpts[:,u,:,:])    # Transform RDM_deriv from k-space to the reference cell         
            the_gradient.append(RDM_deriv_R0)
        
        return np.asarray(the_gradient)
        
    def alt_cost_func(self, uvec):
        '''
        TODO: DEBUGGING
        '''
        
        umat = self.uvec2umat(uvec)
                     
        loc_OEH_kpts, loc_1RDM_kpts, loc_1RDM_R0 = self.local.make_loc_1RDM(umat, self.mask4Gamma, OEH_type=self.OEH_type, dft_HF=self.dft_HF)
        if self.OEH_type == 'FOCK':
            OEH = self.local.loc_actFOCK_kpts #+umat
            OEH = self.local.k_to_R(OEH)
            e_fun = np.trace(OEH.dot(loc_1RDM_R0))
        else: 
            tprint.print_msg('Other type of 1e electron is not supported')
          
        rdm_diff = self.glob_rdm_diff(uvec)[0]  
        e_cstr = np.sum(umat_Rs*rdm_diff)  

        return -e_fun-e_cstr   
        
    def alt_cost_func_grad(self, uvec):
        '''
        TODO: DEBUGGING
        '''
        rdm_diff = self.glob_rdm_diff(uvec)[0]
        return -rdm_diff
        
######################################## USEFUL FUNCTION for pDMET class ######################################## 

    def make_irred_kpts(self, kpts=None):
        '''
        Make k-dependent uvec considering kmesh symmetry
        Attributes:
            kpts_irred      : a list of irreducible k-point   
            sym_id          : a list of symmetry label. k and -k should have the same label
            sym_map         : used to map the uvec (irreducible k-points) to umat (full k-points)        
        '''        
        if kpts is None: kpts = self.kpts
        kpts = np.asarray(kpts)
        
        sym_id = np.asarray(range(self.Nkpts))   
        kpts_irred, sym_counts = np.unique(sym_id, return_counts=True)                        
        sym_map = [np.where(kpts_irred == sym_id[kpt])[0][0] for kpt in range(self.Nkpts)]  
        nkpts_irred = kpts_irred.size         
        num_u = nkpts_irred * self.Nterms      
        uvec = np.zeros(num_u, dtype=np.float64)
        
        return kpts_irred, sym_counts, sym_map, uvec            
    
    def make_mask(self, is_gamma=False):
        '''
        Make a mask used to convert uvec to umat and vice versa
        '''     
        if is_gamma:
            impCluster = np.asarray(self._impOrbs)
            if self.SC_CFtype in ['F', 'FB']:
                mask = np.matrix(impCluster).T.dot(np.matrix(impCluster)) == 1
                mask[np.tril_indices(self.Norbs,-1)] = False
            else:
                mask = np.zeros([self.Norbs,self.Norbs], dtype=bool)
                mask[impCluster==1, impCluster==1] = True
        else:
            mask = np.zeros([self.Nimp, self.Nimp], dtype=bool)            
            if self.SC_CFtype in ['F', 'FB']:
                mask[np.triu_indices(self.Nimp)] = True
            else:
                np.fill_diagonal(mask, True)      
        return mask            

    def uvec2umat(self, uvec):
        '''
        Convert uvec to the umat which is will be added up to the local one-electron Hamiltonian at each k-point
        '''  
        if self.dft_CF:
            the_umat = uvec        
        elif self._is_gamma:
            the_umat = np.zeros([self.Norbs, self.Norbs], dtype=np.float64)          
            the_umat[self.mask] = uvec
            the_umat = the_umat.T
            the_umat[self.mask] = uvec 
        else:
            the_umat = np.zeros([self.Nimp, self.Nimp], dtype=np.float64)          
            the_umat[self.mask] = uvec
            the_umat = the_umat.T
            the_umat[self.mask] = uvec
            
        return np.asarray(the_umat)                

    def umat2uvec(self, umat):
        '''
        Convert umat to the uvec
        '''           
        if self.dft_CF == True:
            return umat
        else:
            return umat[self.mask]       
        
    def make_H1(self, is_gamma=False, impCluster=None):
        '''
        The H1 is the correlation potential operator, this function taking advantage of sparsity of the u matrix in calculating gradient of 1-RDM at each k-point
        Return:
            H1start: 
            H1row: 
            H1col: 
        '''
        if is_gamma == True:
            assert impCluster is not None, "In Gamma-point sampling, you need a list to define impurity orbitals"
        
        theH1 = []
        if is_gamma == True:

            imp_indices = np.where(np.asarray(impCluster) == 1)[0]
            if self.SC_CFtype in ['diagF', 'diagFB']:
                for idx in imp_indices:
                    H1 = np.zeros([self.Norbs, self.Norbs])
                    H1[idx, idx] = 1
                    theH1.append(H1)
            else:      
                for i, row in enumerate(imp_indices):
                    for col in imp_indices[i:]:
                        H1 = np.zeros([self.Norbs, self.Norbs])
                        H1[row, col] = 1
                        H1[col, row] = 1      
                        theH1.append(H1)  
        else:
            if self.SC_CFtype in ['diagF', 'diagFB']:
                for row in range(self.Nimp):
                    H1 = np.zeros([self.Nimp, self.Nimp])
                    H1[row, row] = 1
                    theH1.append(H1)
            else:        
                for row in range(self.Nimp):                                    #Fitting the whole umat
                    for col in range(row, self.Nimp):
                        H1 = np.zeros([self.Nimp, self.Nimp])
                        H1[row, col] = 1
                        H1[col, row] = 1                                
                        theH1.append(H1)    
    
        # Convert the sparse H1 to one dimension H1start, H1row, H1col arrays used in libdmet.rhf_response()
        H1start = []
        H1row   = []
        H1col   = []
        H1start.append(0)
        totalsize = 0
        for count in range(len(theH1)):
            rowco, colco = np.where(theH1[count] == 1)
            totalsize += len(rowco)
            H1start.append(totalsize)
            for count2 in range(len(rowco)):
                H1row.append(rowco[count2])
                H1col.append(colco[count2])
        H1start = np.array(H1start)
        H1row   = np.array(H1row)
        H1col   = np.array(H1col)
        
        return theH1, H1start, H1row, H1col
        
    def construct_1RDM_response_kpts(self, uvec):
        '''
        Calculate the derivative of 1RDM
        TODO: Currently the number of electron is the same at every k-point. This is not the case for
        metallic sytem. So need to consider this later
        '''
            
        rdm_deriv_kpts = []
        loc_actFOCK_kpts = self.local.loc_actFOCK_kpts + self.uvec2umat(uvec)
        Norb = loc_actFOCK_kpts.shape[-1]
        for kpt in range(self.Nkpts):
            #rdm_deriv = libdmet.rhf_response_c(Norb, self.Nterms, self.numPairs, self.H1start, self.H1row, self.H1col, loc_actFOCK_kpts[kpt])
            rdm_deriv = libdmet.rhf_response(Norb, self.Nterms, self.numPairs, self.H1start, self.H1row, self.H1col, loc_actFOCK_kpts[kpt].real)
            rdm_deriv = np.complex128(rdm_deriv)
            rdm_deriv_kpts.append(rdm_deriv)
            
        return np.asarray(rdm_deriv_kpts) 

    def construct_global_1RDM(self):
        ''' Construct the global 1RDM in the R-space'''
        
        imp_1RDM = lib.einsum('Rim,mn,jn->Rij', self.emb_orbs, self.emb_corr_1RDM, self.emb_orbs[0])
        RDM1_Rs = self.local.get_1RDM_Rs(imp_1RDM)
        RDM1_Rs = 0.5*(RDM1_Rs.T + RDM1_Rs)          # make sure the global DM is hermitian
        
        return RDM1_Rs      
    
######################################## POST pDMET ANALYSIS ######################################## 
    def get_bands(self, cell=None, dm_kpts=None, kpts=None, cost_func='glob', method='BFGS'):
        ''' Embedding 1RDM is used to construct the global 1RDM.
            The 'closest' mean-field 1RDM to the global 1RDM is found by minizing the norm(D_global - D_mf) 
        '''     
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kmf.kpts
        
        # Compute the total DM in the local basis
        if cost_func == 'FB':
            CF = self.cost_func
            CF_grad = self.cost_func_grad
            self.SC_CFtype = 'FB'
        elif cost_func == 'F':
            CF = self.cost_func
            CF_grad = self.cost_func_grad
            self.SC_CFtype = 'F'
        else:
            CF = self.glob_cost_func
            CF_grad = self.glob_cost_func_grad
       
        if self.dft_CF and self.xc == 'PBE0':
            result = optimize.minimize(self.CF, self.uvec, method='L-BFGS-B', jac=None, options={'disp': False, 'gtol': 1e-6}, bounds=self.bounds)
        else:
            result = optimize.minimize(CF, self.uvec, method=method, jac=None, options={'disp': False, 'gtol': 1e-12})
 
        uvec = result.x
        error = np.linalg.norm(self.get_glob_rdm_diff(uvec))
        if result.success == False:     
            tprint.print_msg('Band structure error: %12.8f' % (error))
            tprint.print_msg(" WARNING: Correlation potential is not converged")
        else:
            tprint.print_msg('Band structure error: %12.8f' % (error))
            
        if self.dft_CF:
            eigvals, eigvecs = self.local.make_loc_1RDM_kpts(self.uvec2umat(uvec), self.mask4Gamma, OEH_type=self.xc, get_band=True, dft_HF=self.dft_HF)
        else:
            eigvals, eigvecs = self.local.make_loc_1RDM_kpts(self.uvec2umat(uvec), self.mask4Gamma, OEH_type='FOCK', get_band=True, dft_HF=self.dft_HF)

        dmet_orbs = lib.einsum('kpq,kqr->kpr', self.local.ao2lo, eigvecs) # embedding orbs are spaned by AO instead of MLWFs here
        mo_coeff_kpts = []
        mo_energy_kpts = []
        for kpt in range(self.Nkpts):
            mo_coeff = self.kmf.mo_coeff_kpts[kpt].copy()
            mo_coeff[:, self.w90.band_included_list] = dmet_orbs[kpt]
            mo_energy = self.kmf.mo_energy_kpts[kpt].copy()
            mo_energy[self.w90.band_included_list] = eigvals[kpt]
            mo_coeff_kpts.append(mo_coeff)
            mo_energy_kpts.append(mo_energy)
            
        ovlp = self.kmf.get_ovlp()
        class fake_kmf:
            def __init__(self):
                self.kpts = kpts
                self.mo_energy_kpts = mo_energy_kpts
                self.mo_coeff_kpts = mo_coeff_kpts  
                self.get_ovlp  = lambda *arg: ovlp
                
        kmf = fake_kmf()
        
        return kmf
        
    def interpolate_band(self, frac_kpts, use_ws_distance=True, ws_search_size=[2,2,2], ws_distance_tol=1e-6):
        ''' Interpolate the band structure using the Slater-Koster scheme
            Return:
                eigenvalues and eigenvectors at the desired kpts
        '''
        OEH_kpts, eigvals, eigvecs = self.local.make_loc_1RDM_kpts(self.uvec2umat(self.uvec), self.mask4Gamma, OEH_type=self.xc, get_ham=True, dft_HF=self.dft_HF)
        eigvals, eigvecs = self.w90.interpolate_band(frac_kpts, OEH_kpts, use_ws_distance, 
                                                    ws_search_size, ws_distance_tol)
        return (eigvals, eigvecs)
    
        
    def get_supercell_Hamiltonian(self, twoS = 0):        
        '''Make mf object of the effective Hamiltonian for a molecular solver.
        '''        
        
        # 1-ERI
        Hcore_kpts = self.local.loc_actOEI_kpts 
        Hcore = self.local.k_to_R(Hcore_kpts) 
        
        # 2-ERI
        TEI = self.local.get_loc_TEI()
        
        from pyscf import gto, scf,ao2mo        
        mol = gto.Mole()
        mol.build(verbose = self.verbose)
        mol.atom.append(('He', (0, 0, 0)))
        mol.nelectron = self.Nelec_total
        mol.incore_anyway = True
        mol.spin = twoS
        mol.verbose = self.verbose
        if mol.spin == 0:        
            mf = scf.RHF(mol)    
        else:
            mf = scf.ROHF(mol)         
        mf.get_hcore = lambda *args: Hcore
        mf.get_ovlp = lambda *args: np.eye(self.Norbs)
        mf._eri = ao2mo.restore(8, TEI, self.Norbs)
        mf.scf()        
        DMloc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
        if ( mf.converged == False ):
            mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)                
        
        return mol, mf      

    def plot(self, orb = 'emb', grid = [50,50,50], path='./'):        
        '''Plot orbitals for CAS solvers
            orb = 'emb', 'mf', 'mc', 'mc_nat'
        '''            

        emb_orbs = self.emb_orbs[0]    
        if orb == 'wfs' : 
            rotate_mat = None        
        if orb == 'emb' : 
            rotate_mat = emb_orbs                  
        elif orb == 'mf'  : 
            mo = self.qcsolver.mf.mo_coeff       
            rotate_mat = emb_orbs.dot(mo)    
        elif orb == 'mc'  : 
            mo = self.qcsolver.mo
            rotate_mat = emb_orbs.dot(mo)      
        elif orb == 'nat' : 
            mo = self.qcsolver.mo_nat  
            rotate_mat = emb_orbs.dot(mo)  
        elif orb == 'nto':
            assert self.nevpt2_roots is not None, "NEVPT2 must be called to calculate the NTOs"
            pass
            
        tplot.plot_wf(self.w90, rotate_mat, path + '/' + orb, self.kmesh, grid)        


    def get_trans_dipole(self):        
        '''Calculate transition dipole
        '''     
        import scipy
        assert self.nevpt2_roots is not None, "NEVPT2 must be called to calculate the NTOs"
        charges = self.cell.atom_charges()
        coords = self.cell.atom_coords()
        nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
        self.cell.set_common_orig_(nuc_charge_center)
        dip_ints = self.cell.intor('cint1e_r_sph', comp=3) 
        ao2eo = self.local.get_ao2eo(self.emb_orbs)[0]
        def makedip(ci_id):
            t_dm1_emb = self.t_dm1s[ci_id]
            # transform density matrix from MO to AO representation
            t_dm1_ao = ao2eo @ t_dm1_emb @ ao2eo.T.conj()
            return np.einsum('xij,ji->x', dip_ints, t_dm1_ao).real
          

        for i in range(len(self.e_nept2_tot)):
            dipole = makedip(i)
            norm = np.linalg.norm(dipole)
            print('Transition dipole between |0> and |{0:d}>: {1:3.5f} {2:3.5f} {3:3.5f} | Norm: {4:3.5f}'.format(i, dipole[0], dipole[1], dipole[2], norm))
            
        print("=====================================")        
            
    def get_dipole(self):        
        '''Calculate transition dipole
        '''     
        import scipy
        assert self.nevpt2_roots is not None, "NEVPT2 must be called to calculate the NTOs"
        charges = self.cell.atom_charges()
        coords = self.cell.atom_coords()
        nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
        self.cell.set_common_orig_(nuc_charge_center)
        dip_ints = self.cell.intor('cint1e_r_sph', comp=3) 
        ao2eo = self.local.get_ao2eo(self.emb_orbs)[0]
        def makedip(ci_id):
            dm1_emb = self.dm1s[ci_id]
            # transform density matrix from MO to AO representation
            dm1_ao = ao2eo @ dm1_emb @ ao2eo.T.conj()
            return np.einsum('xij,ji->x', dip_ints, dm1_ao).real

        for i in range(len(self.e_nept2_tot)):
            dipole = makedip(i)
            norm = np.linalg.norm(dipole)
            print('Dipole moment for state |{0:d}> : {1:3.5f} {2:3.5f} {3:3.5f} | Norm: {4:3.5f}'.format(i, dipole[0], dipole[1], dipole[2], norm))    
            
        print("=====================================")
