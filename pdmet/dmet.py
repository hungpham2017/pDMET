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
from scipy import optimize
from functools import reduce
from pDMET.pdmet import localbasis, schmidtbasis, qcsolvers, diis, helper
from pDMET.tools import tchkfile, tplot, tprint, tunix
from pDMET.lib.build import libdmet

class pDMET:
    def __init__(self, cell, kmf, w90, solver = 'HF'):
        '''
        Args:
            kmf                             : a rhf wave function from pyscf/pbc
            w90                                : a converged wannier90 object
            OEH_type                        : One-electron Hamiltonian used in the bath construction, h(k) = OEH(k) + umat(k) 
            SCmethod                        : CG/SLSQP/BFGS/L-BFGS-B/LSTSQ self-consistent iteration method, defaut: BFGS
            SC_threshold                    : convergence criteria for correlation potential, default: 1e-6
            SC_maxcycle                     : maximum cycle for self-consistent iteration, default: 50
            umat                            : correlation potential
            chempot                            : global chemical potential
            emb_1RDM                        : a list of the 1RDM for each fragment
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
        solver_list   = ['HF', 'CASCI', 'CASSCF', 'DMRG-CI', 'DMRG-SCF', 'FCI', 'DMRG', 'RCCSD']
        assert solver in solver_list
        self.solver   = solver        
        self.e_shift  = None         # Use to fix spin of the wrong state with FCI, hence CASCI/CASSCF solver
        
        # Parameters
        self.kmesh_sym          = True        
        self.SC_method          = 'L-BFGS-B'        # BFGS, L-BFGS-B, CG, Newton-CG
        self.SC_threshold       = 1e-8            
        self.SC_maxcycle        = 200
        self.SC_CFtype          = 'F' # Options: ['F','diagF', 'FB','diagFB']
        self.doSCF              = False
        self.alt_CF             = False         
        self.damping            = 1.0 # 1.0 means no damping
        self.DIIS               = True           
        self.DIIS_m             = 1   
        self.DIIS_n             = 10   
        
        # DMET Output
        self.nroots         = 1
        self.state_percent  = None        
        self.twoS           = 0
        self.verbose        = 0
        self.max_memory     = 4000  # in MB      
        self.locOED_Ls      = None
        self.baths          = None
        self.emb_1RDM       = None
        self.env_orbs       = None        
        self.emb_orbs       = None
        self.e_tot         = None       # energy per unit cell     
        self.e_corr         = None
        self.core1RDM_local = None
        self.nelec_cell     = None      
        
        # Others
        self.bath_truncation = False   # if self.truncate = a threshold, then a bath truncation scheme is used          
        self.chkfile         = None    # Save integrals in the WFs basis as well as chem potential and uvec
        self.restart         = False   # Run a calculation using saved chem potential and uvec             
        self.cycle           = 1        
        
    def initialize(self, chkfile = None):
        '''
        Prepare the local integrals, correlation/chemical potential    
        '''    
        
        tprint.print_msg("Initializing ...")
        
        # -------------------------------------------------        
        # General initialized attributes 
        if self.kmf_chkfile != None:
            self.kmf = tchkfile.load_kmf(self.cell, self.kmf, self.w90.mp_grid_loc, self.kmf_chkfile, symmetrize = self.kmesh_sym, max_memory=self.max_memory)
        elif self.kmesh_sym == True:
            self.kmf = tchkfile.symmetrize_kmf(self.cell, self.kmf) 
            
        if self.w90_chkfile != None:
            self.w90 = tchkfile.load_w90(self.w90_chkfile)
        else:
            self.w90 = self.w90 
            
        assert (self.chkfile == None) or isinstance(self.chkfile,str)
        self.nkpts = self.kmf.kpts.shape[0]   
        self.local = localbasis.WF(self.cell, self.kmf, self.w90, chkfile = self.chkfile)  
        self.e_core = self.local.e_core   
        self.nImps = self.local.nactorbs

        # -------------------------------------------------        
        # The number of bath orbitals depends on whether one does Schmidt decomposition on RHF or ROHF wave function        
        if self.cell.spin == 0:
            self.bathtype = 'RHF'          
            self.nBathOrbs = self.nImps
        else:
            self.bathtype = 'ROHF'            
            self.nBathOrbs = self.nImps + self.cell.spin         
        self.Norbs = self.local.norbs
        self.Nelecs = self.local.nelec
        self.numPairs = self.local.nactelecs//2        

        # -------------------------------------------------        
        # Labeling the reference unit cell as the fragment
        # The nkpts is an odd number so the reference unit is assumed to be in the middle of the computational supercell
        self.impCluster = np.zeros((self.Norbs))
        self.impCluster[self.nImps*(self.nkpts//2):self.nImps*(self.nkpts//2 + 1)] = 1
        
        # -------------------------------------------------        
        # Correlation/chemical potential
        if self.alt_CF == True: 
            self.SC_CFtype = 'F' 
            self.CF      = self.alt_costfunction
            self.CF_grad = self.alt_costfunction_gradient            
        else:
            self.CF = self.costfunction            
            self.CF_grad = self.costfunction_gradient  
            
        self.H1start, self.H1row, self.H1col = self.make_H1()[1:4]    #Use in the calculation of 1RDM derivative
        if self.SC_CFtype in ['diagF', 'diagFB']: 
            self.Nterms = self.nImps 
        else:            
            self.Nterms = self.nImps*(self.nImps + 1) // 2   
            
        self.kNterms = self.Nterms      
        self.umat_kpt = False   
        self.mask = self.make_mask()   

         # -------------------------------------------------       
        # Load/initiate chem pot, uvec, umat    
        self.restart_success = False        
        if self.chkfile != None and self.restart == True:
            self.restart_success = True    
            self.chempot = None 
            if tunix.check_exist(self.chkfile):
                self.save_pdmet     = tchkfile.load_pdmet(self.chkfile) 
                chk_sym             = self.save_pdmet.kmesh_sym                 
                self.chempot        = self.save_pdmet.chempot
                self.uvec           = self.save_pdmet.uvec
                self.umat           = self.save_pdmet.umat                           
                if chk_sym != None: self.umat_kpt  = chk_sym

            if self.chempot == None or self.chempot == False:
                self.restart_success = False            
                tprint.print_msg("-> Cannot load the saved chemical and correlation potential") 
            else:
                tprint.print_msg("-> Load the saved chemical and correlation potential") 

        if self.restart_success == False:
            self.chempot = 0.0        
            self.uvec = np.zeros(self.Nterms, dtype=np.float64)           
            self.umat = self.uvec2umat(self.uvec)
            
        # Initializing Damping procedure and DIIS object          
        if self.DIIS == True:
            assert self.DIIS_m >= 1        
            self._diis = diis.DIIS(self.DIIS_m, self.DIIS_n, self.nkpts)   
        if self.damping != 1.0:
            assert (0 <= self.damping <= 1.0)          

        # Initializing the QC solver
        if self.nroots > 1:
            if self.solver not in ['FCI', 'DMRG']: raise Exception('Only FCI or DMRG solver supports excited state calculations')
            if self.state_percent == None: 
                self.state_percent = [1/self.nroots]*self.nroots
            else:
                assert len(self.state_percent) == self.nroots
                assert abs(sum(self.state_percent) - 1.0) < 1.e-10      # The total percent has be 1   
                
        if self.twoS != 0 and self.solver == 'RCCSD': raise Exception('RCCSD solver does not support ROHF wave function')             

        # For FCI solver
        self._SS = 0.5*self.twoS*(0.5*self.twoS + 1)       
        self.qcsolver = qcsolvers.QCsolvers(self.solver, self.twoS, self.e_shift, self.nroots, self.state_percent, verbose=self.verbose, memory=self.max_memory) 
        
        tprint.print_msg("Initializing ... DONE")       
                
    def kernel(self, chempot = 0.0):
        '''
        This is the main kernel for DMET calculation.
        It is solving the embedding problem, then returning the total number of electrons per unit cell 
        and updating the schmidt orbitals and 1RDM.
        Args:
            chempot                    : global chemical potential to adjust the number of electrons in the unit cell
        Return:
            nelecs                     : the total number of electrons
        Update the class attributes:
            energy                    : the energy for the unit cell  
            nelec                    : the number of electrons for the unit cell    
            emb_1RDM                : the 1RDM for the unit cell                
        '''            
        

        numImpOrbs = self.nImps        
        numBathOrbs, FBEorbs, envOrbs_or_core_eigenvals = self.baths
        Norb_in_imp  = numImpOrbs + numBathOrbs
        assert(Norb_in_imp <= self.Norbs)

        core_cutoff = 0.001
        for cnt in range(len(envOrbs_or_core_eigenvals)):
            if (envOrbs_or_core_eigenvals[cnt] < core_cutoff):
                envOrbs_or_core_eigenvals[cnt] = 0.0
            elif (envOrbs_or_core_eigenvals[cnt] > 2.0 - core_cutoff):
                envOrbs_or_core_eigenvals[cnt] = 2.0
            else:
                raise Exception("   Bad DMET bath orbital selection: Trying to put a bath orbital with occupation", \
                                        envOrbs_or_core_eigenvals[cnt], "into the environment")
        Nelec_in_imp = int(round(self.Nelecs - np.sum(envOrbs_or_core_eigenvals)))
        Nelec_in_environment = int(np.sum(np.abs(envOrbs_or_core_eigenvals)))                
        core1RDM_local = reduce(np.dot, (FBEorbs, np.diag(envOrbs_or_core_eigenvals), FBEorbs.T))                     
            
        # Transform the 1e/2e integrals and the JK core constribution to schmidt basis
        dmetOEI  = self.local.dmet_oei(FBEorbs, Norb_in_imp)
        dmetTEI  = self.local.dmet_tei(FBEorbs, Norb_in_imp)            
        dmetCoreJK = self.local.dmet_corejk(FBEorbs, Norb_in_imp, core1RDM_local)

        # Solving the embedding problem with high level wfs
        if self.cycle == 1 : tprint.print_msg("   Embedding size: %2d electrons in (%2d fragments + %2d baths )" % (Nelec_in_imp, numImpOrbs, numBathOrbs))                        
        DMguess = reduce(np.dot,(FBEorbs[:,:Norb_in_imp].T, self.locOED_Ls, FBEorbs[:,:Norb_in_imp]))    
        
        self.qcsolver.initialize(self.local.e_core, dmetOEI, dmetTEI, \
                                dmetCoreJK, DMguess, Norb_in_imp, Nelec_in_imp, numImpOrbs, chempot)
        if self.solver == 'HF':
            e_cell, RDM1 = self.qcsolver.HF()
        elif self.solver in ['CASCI','CASSCF']:
            e_cell, RDM1 = self.qcsolver.CAS()    
        elif self.solver in ['DMRG-CI','DMRG-SCF']:
            e_cell, RDM1 = self.qcsolver.CAS(solver = 'CheMPS2')    
        elif self.solver == 'FCI':
            e_cell, RDM1 = self.qcsolver.FCI()
        elif self.solver == 'DMRG':
            e_cell, RDM1 = self.qcsolver.DMRG()          
        elif self.solver == 'RCCSD':
            e_cell, RDM1 = self.qcsolver.RCCSD()              
         
        self.core1RDM_local = core1RDM_local
        self.emb_1RDM = RDM1
        self.emb_orbs = FBEorbs[:,:Norb_in_imp]
        self.env_orbs = FBEorbs[:,Norb_in_imp:]        
        self.nelec_cell = np.trace(RDM1[:numImpOrbs,:numImpOrbs])
    
        if not np.isclose(self._SS, self.qcsolver.SS): 
            tprint.print_msg("           WARNING: Spin contamination. Computed <S^2>: %10.8f, Target: %10.8f" % (self.qcsolver.SS, self._SS)) 
            
        # Get the cell energy:         
        self.e_tot = e_cell 
        
        return self.nelec_cell

    def check_exact(self, error = 1.e-8):
        '''
        Do one-shot DMET, only the chemical potential is optimized
        '''
        
        tprint.print_msg("--------------------------------------------------------------------")   
        
        self.locOED_Ls = self.local.construct_locOED_Ls(self.umat, 'FOCK', False, self.verbose)        
        schmidt = schmidtbasis.HF_decomposition(self.cell, self.impCluster, self.nBathOrbs, self.locOED_Ls)
        self.baths = schmidt.baths(self.bath_truncation) 
        solver = self.solver
        self.solver = 'HF'
        nelec_cell = self.kernel(chempot = 0.0)
        self.solver = solver        
        
        diff = abs(self.e_tot - self.kmf.e_tot)
        tprint.print_msg("   E(RHF)        : %12.8f" % (self.kmf.e_tot))
        tprint.print_msg("   E(RHF-DMET)   : %12.8f" % (self.e_tot))
        tprint.print_msg("   |RHF - RHF(DMET)|          : %12.8f" % (diff))
        if diff < error : 
            tprint.print_msg("   HF-in-HF embedding is exact: True")
        else:
             raise Exception('WARNING: HF-in-HF embedding is not exact')                    
        
    def one_shot(self):
        '''
        Do one-shot DMET, only the chemical potential is optimized
        '''

        tprint.print_msg("-- One-shot pDMET ... starting at %s" % (tunix.current_time()))    
        if self.solver == 'HF' and self.twoS == 0:
            tprint.print_msg("   Bath type: %s | QC Solver: %s" % (self.bathtype, 'RHF'))
        elif self.solver == 'HF' and self.twoS != 0:        
            tprint.print_msg("   Bath type: %s | QC Solver: %s | 2S = %d" % (self.bathtype, 'ROHF', self.twoS)) 
        elif self.solver == 'RCCSD': 
            tprint.print_msg("   Bath type: %s | QC Solver: %s | 2S = %d" % (self.bathtype, self.solver, self.twoS))        
        else:      
            tprint.print_msg("   Bath type: %s | QC Solver: %s | 2S = %d | Nroots: %d" % (self.bathtype, self.solver, self.twoS, self.nroots))
            
        if self.solver in ['CASCI','CASSCF','DMRG-CI','DMRG-SCF']:                
            if self.qcsolver.cas != None: tprint.print_msg("   Active space     :", self.qcsolver.cas)
            if self.qcsolver.cas != None: tprint.print_msg("   Active space MOs :", self.qcsolver.molist)            

        self.cycle = 1       
        # Optimize the chemical potential          
        self.locOED_Ls = self.local.construct_locOED_Ls(self.umat, self.OEH_type, self.doSCF, self.verbose)        # get both MO coefficients and 1-RDM in the local basis     
        schmidt = schmidtbasis.HF_decomposition(self.cell, self.impCluster, self.nBathOrbs, self.locOED_Ls)
        self.baths = schmidt.baths(self.bath_truncation) 
        self.chempot = optimize.newton(self.nelec_costfunction, self.chempot)       
      
        tprint.print_msg("   No. of electrons per cell : %12.8f" % (self.nelec_cell))
        tprint.print_msg("   Energy per cell           : %12.8f" % (self.e_tot))                          
        tprint.print_msg("-- One-shot pDMET ... finished at %s" % (tunix.current_time()))
        tprint.print_msg()            
        
    def self_consistent(self, umat_kpt = False, loose = False):
        '''
        Do self-consistent pDMET
        '''    
      
        if self.alt_CF == True: umat_kpt = False       
        if self.cell.spin !=0: raise Exception('sc-pDMET cannot be runned with ROHF bath')  
        tprint.print_msg("--------------------------------------------------------------------")
        tprint.print_msg("- SELF-CONSISTENT pDMET CALCULATION ... STARTING -")
        tprint.print_msg("  Convergence criteria")   
        tprint.print_msg("    Threshold :", self.SC_threshold) 
        tprint.print_msg("    Loose     :", loose)         
        tprint.print_msg("  Fitting 1-RDM of :", self.SC_CFtype)         
        tprint.print_msg("  k-dependent umat :", umat_kpt)           
        tprint.print_msg("  Kmesh symmetry   :", self.kmesh_sym)   
        if self.damping != 1.0:        
            tprint.print_msg("  Damping factor   :", self.damping)                
        if self.DIIS:
            tprint.print_msg("  DIIS start at %dth cycle and using %d previous umats" % (self.DIIS_m, self.DIIS_n))            
        
        self.umat_kpt = umat_kpt           
        uvec = np.zeros(self.Nterms, dtype=np.float64)
        weight = 1           
        if self.umat_kpt == True:        
            self.kpts_irred, self.sym_counts, self.sym_map, uvec = self.make_uvec(self.kmf.kpts)
            self.nkpts_irred = self.kpts_irred.size  
            self.kNterms = self.nkpts_irred*self.Nterms 
            weight  = 1/self.nkpts             

        use_saved_uvec = False            
        if self.chkfile != None and self.restart == True:     
            if self.restart_success == True and self.uvec.size == uvec.size:     
                tprint.print_msg("  Restart from the saved uvec") 
                use_saved_uvec = True
                
                
        if use_saved_uvec == False:
            self.uvec = uvec  
            self.umat = self.uvec2umat(self.uvec)            
                                          
        #------------------------------------#
        #---- SELF-CONSISTENCY PROCEDURE ----#      
        #------------------------------------#      
        u_diff = 1.0
        rdm1 = self.local.construct_locOED_kpts(self.umat, self.OEH_type, self.doSCF, self.verbose)
        for cycle in range(self.SC_maxcycle):
            
            tprint.print_msg("- CYCLE %d:" % (cycle + 1))    
            umat_old = self.umat      
            rdm1_old = rdm1                  
            # Do one-shot with each uvec                  
            self.one_shot() 
            
            tprint.print_msg("   + Chemical potential        : %12.8f" % (self.chempot))

            # Optimize uvec to minimize the cost function
            if self.SC_method in ['BFGS', 'L-BFGS-B', 'CG', 'Newton-CG']:
                result = optimize.minimize(self.CF, self.uvec, method = self.SC_method, jac = self.CF_grad, options = {'disp': False, 'gtol': 1e-12})
                # result = optimize.minimize(self.CF, self.uvec, method = self.SC_method , options = {'disp': False, 'gtol': 1e-12})                
            else:
                tprint.print_msg("     WARNING:", self.SC_method, " is not supported")
                
            if result.success == False:         
                tprint.print_msg("     WARNING: Correlation potential is not converged")    
                
            self.uvec = result.x
            self.umat = self.uvec2umat(self.uvec)   
            rdm1  = self.local.construct_locOED_kpts(self.umat, self.OEH_type, self.doSCF, self.verbose)   
            
            if self.umat_kpt == True:
                self.umat = np.asarray([self.umat[kpt] - np.eye(self.umat[kpt].shape[0])*np.average(np.diag(self.umat[kpt])) for kpt in range(self.nkpts)])            
            else:            
                self.umat = self.umat - np.eye(self.umat.shape[0])*np.average(np.diag(self.umat))            

            if self.verbose > 0 and self.umat_kpt == True:
                uvec = self.uvec.reshape(self.nkpts_irred,-1)
                tprint.print_msg("   + Correlation potential vector    : ")
                for k, kpt in enumerate(self.cell.get_scaled_kpts(self.kmf.kpts)):
                    tprint.print_msg('    %2d (%6.3f %6.3f %6.3f)   %s' % (k, kpt[0], kpt[1], kpt[2], uvec[self.sym_map[k]]))                
            elif self.verbose > 0:
                tprint.print_msg("   + Correlation potential vector    : ", self.uvec)
                    
            umat_diff = self.umat - umat_old          
            rdm_diff  = rdm1 - rdm1_old             
            norm_u    = weight * np.linalg.norm(umat_diff)             
            norm_rdm  = 1/self.nkpts * np.linalg.norm(rdm_diff) 
            
            tprint.print_msg("   + Cost function             : %20.15f" % (result.fun)) #%12.8
            tprint.print_msg("   + 2-norm of umat difference : %20.15f" % (norm_u)) 
            tprint.print_msg("   + 2-norm of rdm1 difference : %20.15f" % (norm_rdm))  
            
            # Check convergence of 1-RDM            
            if (loose == False and norm_u <= self.SC_threshold): 
                break
            elif (loose == True and norm_rdm <= self.SC_threshold): 
                break

            if self.damping != 1.0:                
                self.umat = (1.0 - self.damping)*umat_old + self.damping*self.umat               
            if self.DIIS == True:  
                self.umat = self._diis.update(cycle, self.umat, umat_diff)            
                
            self.uvec =  self.umat2uvec(self.umat)    
            
            tprint.print_msg()            
            
        tprint.print_msg("- SELF-CONSISTENT pDMET CALCULATION ... DONE -")
        tprint.print_msg("--------------------------------------------------------------------")            
        
    def nelec_costfunction(self, chempot):
        '''
        The different in the correct number of electrons (provided) and the calculated one 
        '''
        
        Nelec_dmet = self.kernel(chempot)
        Nelec_target = self.Nelecs // self.nkpts   
        print ("     Cycle %2d. Chem potential: %12.8f | Elec/cell = %12.8f | <S^2> = %12.8f" % \
                                                        (self.cycle, chempot, Nelec_dmet, self.qcsolver.SS))                                                                                
        self.cycle += 1
        if self.chkfile != None: tchkfile.save_pdmet(self, self.chkfile)

        return abs(Nelec_dmet - Nelec_target)

    def costfunction(self, uvec):
        '''
        Cost function: \mathbf{CF}(u) = \mathbf{\Sigma}_{rs} (D^{mf}_{rs}(u) - D^{corr}_{rs})^2
        where D^{mf} and D^{corr} are the mean-field and correlated 1-RDM, respectively.
        and D^{mf} = \mathbf{FT}(D^{mf}(k))
        '''
        
        uvec = uvec      
        rdm_diff = self.rdm_diff(uvec)
        cost = np.power(rdm_diff, 2).sum()  
        return cost
        
    def costfunction_gradient(self, uvec):
        '''
        Analytical derivative of the cost function,
        deriv(CF(u)) = Sum^x [Sum_{rs} (2 * rdm_diff^x_{rs}(u) * deriv(rdm_diff^x_{rs}(u))]
        ref: J. Chem. Theory Comput. 2016, 12, 2706−2719
        '''
        uvec = uvec
        rdm_diff = self.rdm_diff(uvec)   
        rdm_diff_gradient = self.rdm_diff_gradient(uvec)  
        CF_gradient = np.zeros(self.kNterms)

        for u in range(self.kNterms):
            CF_gradient[u] = np.sum(2 * rdm_diff * rdm_diff_gradient[u])
        
        return CF_gradient
        
        
    def rdm_diff(self, uvec):
        '''
        Calculating the different between mf-1RDM (transformed in schmidt basis) and correlated-1RDM for the unit cell
        Args:
            uvec            : the correlation potential vector
        Return:
            error            : an array of errors for the unit cell.
        '''
        
        locOED = self.local.construct_locOED_Ls(self.uvec2umat(uvec), self.OEH_type, self.doSCF, self.verbose)
        if self.SC_CFtype in ['F', 'diagF']:        
            mf_1RDM = reduce(np.dot, (self.emb_orbs[:,:self.nImps].T, locOED, self.emb_orbs[:,:self.nImps]))
            corr_1RDM = self.emb_1RDM[:self.nImps,:self.nImps]    
        elif self.SC_CFtype in ['FB', 'diagFB']:  
            mf_1RDM = reduce(np.dot, (self.emb_orbs.T, locOED, self.emb_orbs))
            corr_1RDM = self.emb_1RDM    
            
        error = mf_1RDM - corr_1RDM
        if self.SC_CFtype in ['diagF', 'diagFB']: error = np.diag(error)      
        
        return error

    def rdm_diff_gradient(self, uvec):
        '''
        Compute the rdm_diff gradient
        Args:
            uvec            : the correlation potential vector
        Return:
            the_gradient    : a list with the size of the number of u values in uvec
                              Each element of this list is an array of derivative corresponding to each rs.
                             
        '''
        
        the_RDM_deriv_kpts = self.construct_1RDM_response_kpts(uvec)
        the_gradient = []    
        if self.umat_kpt == True:
            for i, kpt in enumerate(self.kpts_irred):   # this is k1 index of u, RDM_deriv(k0) is non zero only when k0 = k1 = kpt
                for u in range(self.Nterms):
                    RDM_deriv_Ls = self.local.to_Ls_sparse(the_RDM_deriv_kpts[kpt,u,:,:], self.kmf.kpts[kpt])    # Transform RDM_deriv from k-space to L-space
                    if self.SC_CFtype in ['F','diagF']: 
                        error_deriv_in_schmidt_basis = reduce(np.dot, (self.emb_orbs[:,:self.nImps].T, RDM_deriv_Ls, self.emb_orbs[:,:self.nImps]))    
                    elif self.SC_CFtype in ['FB','diagFB']: 
                        error_deriv_in_schmidt_basis = reduce(np.dot, (self.emb_orbs.T, RDM_deriv_Ls, self.emb_orbs))                    
                    if self.SC_CFtype in ['diagF', 'diagFB']: error_deriv_in_schmidt_basis = np.diag(error_deriv_in_schmidt_basis)
                    the_gradient.append(self.sym_counts[i]*error_deriv_in_schmidt_basis)                    
        else:
            for u in range(self.Nterms):
                RDM_deriv_Ls = self.local.to_Ls(the_RDM_deriv_kpts[:,u,:,:])    # Transform RDM_deriv from k-space to L-space               
                if self.SC_CFtype in ['F','diagF']: 
                    error_deriv_in_schmidt_basis = reduce(np.dot, (self.emb_orbs[:,:self.nImps].T, RDM_deriv_Ls, self.emb_orbs[:,:self.nImps]))    
                elif self.SC_CFtype in ['FB','diagFB']:
                    error_deriv_in_schmidt_basis = reduce(np.dot, (self.emb_orbs.T, RDM_deriv_Ls, self.emb_orbs))     
                if self.SC_CFtype in ['diagF', 'diagFB']: error_deriv_in_schmidt_basis = np.diag(error_deriv_in_schmidt_basis)
                the_gradient.append(error_deriv_in_schmidt_basis)
        
        return the_gradient
        
    def alt_costfunction(self, uvec):
        '''
        Only works for umat_kpt = False and SC_CFtype = 'F'
        TODO: need to write
        '''
      
        umat = self.uvec2umat(uvec)
        locOED_kpts = self.local.construct_locOED_kpts(umat, self.OEH_type, self.doSCF, self.verbose)       
        if self.OEH_type == 'OEI':
            e_fun = 1./self.nkpts * np.einsum('kij,kji->', self.local.loc_actOEI_kpts, locOED_kpts).real
        elif self.OEH_type == 'FOCK':
            e_fun = 1./self.nkpts * np.einsum('kij,kji->', self.local.loc_actFOCK_kpts, locOED_kpts).real     
          
        errors = self.rdm_diff(uvec)
        dev = self.rdm_diff_gradient(uvec)    
        e_cstr = np.sum(umat*errors)  
        
        return errors.sum()       
        
    def alt_costfunction_gradient(self, uvec):
        '''
        TODO: need to write
        '''
        errors = self.rdm_diff(uvec-uvec)
        dev = self.rdm_diff_gradient(uvec)
        
        return -dev
        
######################################## USEFUL FUNCTION for pDMET class ######################################## 

    def make_uvec(self, kpts):
        '''
        Make k-dependent uvec considering kmesh symmetry
        Attributes:
            kpts_irred      : a list of irreducible k-point   
            sym_id          : a list of symmetry label. k and -k should have the same label
            sym_map         : used to map the uvec (irreducible k-points) to umat (full k-points)        
        '''        
        sym_id = np.asarray(range(self.nkpts))   
        if self.kmesh_sym == True:  
            for i in range(kpts.shape[0]-1):
                for j in range(i+1): 
                    if abs(kpts[i+1] + kpts[j]).sum() < 1.e-10: 
                        sym_id[i+1] = sym_id[j]   
                        break 
        kpts_irred, sym_counts = np.unique(sym_id, return_counts=True)                        
        sym_map = [np.where(kpts_irred == sym_id[kpt])[0][0] for kpt in range(self.nkpts)]  
        nkpts_irred = kpts_irred.size         
        num_u = nkpts_irred * self.Nterms      
        uvec = np.zeros(num_u, dtype=np.float64)
        
        return kpts_irred, sym_counts, sym_map, uvec            
    
    def make_mask(self):
        '''
        Make a mask used to convert uvec to umat and vice versa
        '''     
        mask = np.zeros([self.nImps, self.nImps], dtype=bool)            
        if self.SC_CFtype in ['F', 'FB']:
            mask[np.triu_indices(self.nImps)] = True
        else:
            np.fill_diagonal(mask, True)
        return mask            

    def uvec2umat(self, uvec):
        '''
        Convert uvec to the umat which is will be added up to the local one-electron Hamiltonian at each k-point
        '''           
         
        if self.umat_kpt == True:
            the_umat = []
            uvec = uvec.reshape(self.nkpts_irred, -1)
            for kpt in range(self.nkpts):
                umat = np.zeros([self.nImps, self.nImps], dtype=np.float64)
                umat[self.mask] = uvec[self.sym_map[kpt]]
                umat = umat.T
                umat[self.mask] = uvec[self.sym_map[kpt]]
                the_umat.append(umat)
        else:              
            the_umat = np.zeros([self.nImps, self.nImps], dtype=np.float64)          
            the_umat[self.mask] = uvec
            the_umat = the_umat.T
            the_umat[self.mask] = uvec
            
        return np.asarray(the_umat)                

    def umat2uvec(self, umat):
        '''
        Convert umat to the uvec
        '''    
        
        if self.umat_kpt == True:
            uvec = np.asarray([umat[kpt][self.mask] for kpt in self.kpts_irred])
        else:              
            uvec = umat[self.mask]          
            
        return uvec    
        
    def make_H1(self):
        '''
        The H1 is the correlation potential operator, used to calculate gradient of 1-RDM at each k-point
        Return:
            H1start: 
            H1row: 
            H1col: 
        '''
        
        theH1 = []

        if self.SC_CFtype in ['diagF', 'diagFB']:                                             #Only fitting the diagonal elements of umat
            for row in range(self.nImps):
                H1 = np.zeros([self.nImps, self.nImps])
                H1[row, row] = 1
                theH1.append(H1)
        else:        
            for row in range(self.nImps):                                    #Fitting the whole umat
                for col in range(row, self.nImps):
                    H1 = np.zeros([self.nImps, self.nImps])
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
        '''
            
        rdm_deriv_kpts = []
        loc_actFOCK_kpts = self.local.loc_actFOCK_kpts + self.uvec2umat(uvec)
        for kpt in range(self.nkpts):
            rdm_deriv = libdmet.rhf_response_c(self.nImps, self.Nterms, self.numPairs, self.H1start, self.H1row, self.H1col, loc_actFOCK_kpts[kpt])
            rdm_deriv_kpts.append(rdm_deriv)
            
        return np.asarray(rdm_deriv_kpts)        
    
######################################## POST pDMET ANALYSIS ######################################## 
    def get_bands(self, cell=None, dm_kpts=None, kpts=None, verbose=0, max_cycle=200, debug=False):
        ''' Giving a correlated 1RDM, compute the umat that produces the 'closest' mean-field 1RDM
            then make a new kmf object using this 1RDM for pyWannier90 for band plotting purpose
        '''
        tprint.print_msg('------------- Computing band structure -------------')        
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kmf.kpts
        ao2loc = self.local.CO
        
        # Run SCF for the umat optimization
        SCF_converged = True
        # activeDMloc_kpts = self.local.construct_locOED_kpts(self.umat, 'OEI', doSCF=False, verbose=verbose, max_cycle=max_cycle)
        SCF_converged, activeDMloc_kpts = self.local.construct_locOED_kpts(self.umat, 'OEI', doSCF=True, verbose=verbose, max_cycle=max_cycle)
        
        if SCF_converged == True and debug == False:
            activeDMao_kpts = np.asarray([reduce(np.dot,(ao2loc[kpt], activeDMloc_kpts[kpt],ao2loc[kpt].conj().T)) for kpt in range(self.nkpts)])
             
            # Total RDM1:
            dm_kpts = self.local.coreDM_kpts + activeDMao_kpts
            fock = self.kmf.get_hcore(cell, kpts) + self.kmf.get_veff(cell=cell, dm_kpts=dm_kpts, kpts=kpts, kpts_band=kpts)
            s1e = self.kmf.get_ovlp(cell, kpts)
            mo_energy, mo_coeff = self.kmf.eig(fock, s1e)
            kmf = self.kmf
            kmf.mo_energy_kpts = mo_energy
            kmf.mo_coeff_kpts = mo_coeff       
            tprint.print_msg('---------- Computing band structure: Done ----------') 
            return kmf
        else:
            tprint.print_msg('---------- Computing band structure: Not converged ----------') 
            return 0
            
    def get_bands_dft(self, cell=None, XC='PBE', method=1, dm_kpts=None, kpts=None, verbose=0, max_cycle=1):
        ''' Giving a correlated 1RDM, compute the umat that produces the 'closest' mean-field 1RDM
            then make a new kmf object using this 1RDM for pyWannier90 for band plotting purpose
        '''
        tprint.print_msg('------------- Computing band structure -------------')        
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kmf.kpts
        
        # Method 1:
        if method == 1:
            full_1RDM = reduce(np.dot,(self.emb_orbs, self.emb_1RDM,self.emb_orbs.conj().T))          # transform RDM(emb) --> fullRDM
            #full_1RDM += full_1RDM + self.core1RDM_local                                              # adding the core 1RDM
            print("DEBUG", full_1RDM.sum(), self.local.to_Ls(self.kmf.make_rdm1()).sum())
            full_1RDM_kpts = self.local.to_kspace(full_1RDM) # fullRDM -FT-> fullRDM(k) in local basis
            ao2loc = self.local.CO
            activeDMao_kpts = np.asarray([reduce(np.dot,(ao2loc[kpt], full_1RDM_kpts[kpt],ao2loc[kpt].conj().T)) for kpt in range(self.nkpts)])
        elif method == 2:        
            ao2loc = self.local.CO
            activeDMloc_kpts = self.local.construct_locOED_kpts(self.umat, self.OEH_type, self.doSCF)
            activeDMao_kpts = np.asarray([reduce(np.dot,(ao2loc[kpt], activeDMloc_kpts[kpt],ao2loc[kpt].conj().T)) for kpt in range(self.nkpts)])
             
        # Total RDM1:
        # from pyscf.pbc import scf
        # dm_kpts = self.local.coreDM_kpts + activeDMao_kpts
        # kmf = scf.KRHF(cell, kpts, exxdiv=None)
        # kmf.xc = XC
        # kmf.verbose = self.verbose
        # kmf.max_cycle = max_cycle
        # kmf.run(dm_kpts)   
        # tprint.print_msg('---------- Computing band structure: Done ----------') 
        
        return 1 # kmf

            
    def get_bands_(self, kpts_band, cell=None, dm_kpts=None, kpts=None):
        '''Get energy bands at the given (arbitrary) 'band' k-points.

        Returns:
            mo_energy : (nmo,) ndarray or a list of (nmo,) ndarray
            Bands energies E_n(k)
            mo_coeff : (nao, nmo) ndarray or a list of (nao,nmo) ndarray
            Band orbitals psi_n(k)
            TODO: the function works only for exxdiv='vcut_sph'. Other options the band structure will have critical points
        '''
        
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kmf.kpts
        frozenDMao_kpts = self.local.frozenDMao_kpts
        activeDMloc_kpts = self.local.construct_locOED_kpts(self.uvec2umat(self.uvec), self.OEH_type)
        CO = self.local.CO
        activeDMao_kpts = np.einsum('kua,kab,bvk ->kuv', CO, activeDMloc_kpts, CO.T.conj(), optimize=True)
        # Total RDM1:
        dm_kpts = frozenDMao_kpts + activeDMao_kpts
        
        kpts_band = np.asarray(kpts_band)
        single_kpt_band = (kpts_band.ndim == 1)
        kpts_band = kpts_band.reshape(-1,3)
        fock = self.kmf.get_hcore(cell, kpts_band)
        fock = fock + self.kmf.get_veff(cell, dm_kpts, kpts, kpts_band)
        s1e = self.kmf.get_ovlp(cell, kpts_band)
        mo_energy, mo_coeff = self.kmf.eig(fock, s1e)
        if single_kpt_band:
            mo_energy = mo_energy[0]
            mo_coeff = mo_coeff[0]
        
        return mo_energy, mo_coeff
        
    def effHamiltonian(self, twoS = 0):        
        '''Make mf object of the effective Hamiltonian for a molecular solver.
        '''        
        Nimp = self.Nelecs
        
        FOCK = self.local.loc_actOEI_Ls.copy()
        
        from pyscf import gto, scf,ao2mo        
        mol = gto.Mole()
        mol.build(verbose = self.verbose)
        mol.atom.append(('He', (0, 0, 0)))
        mol.nelectron = self.Nelecs
        mol.incore_anyway = True
        mol.spin = twoS
        mol.verbose = self.verbose
        if mol.spin == 0:        
            mf = scf.RHF(mol)    
        else:
            mf = scf.ROHF(mol)         
        mf.get_hcore = lambda *args: FOCK
        mf.get_ovlp = lambda *args: np.eye(self.Norbs)
        mf._eri = ao2mo.restore(8, self.local.loc_actTEI_Ls, self.Norbs)
        mf.scf()        
        DMloc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
        if ( mf.converged == False ):
            mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)                
        
        return mol, mf      

    def plot(self, orb = 'emb', grid = [50,50,50]):        
        '''Plot orbitals for CAS solvers
            orb = 'emb', 'mf', 'mc', 'mc_nat'
        '''    
        
        if self.chkfile != None and self.restart == True:
            if self.restart_success != True:
                raise Exception('Need to run at least one cycle to generate orbitals')
        else:
            raise Exception('Need to restart from a chkfile')                    

        emb_orbs = self.save_pdmet.emb_orbs    
        if orb == 'wfs' : rotate_mat = None        
        if orb == 'emb' : rotate_mat = emb_orbs        
        if orb == 'env' : rotate_mat = self.save_pdmet.env_orbs            
        if orb == 'mf'  : rotate_mat = emb_orbs.dot(self.save_pdmet.mf_mo)                
        if orb == 'mc'  : rotate_mat = emb_orbs.dot(self.save_pdmet.mc_mo)    
        if orb == 'nat' : rotate_mat = emb_orbs.dot(self.save_pdmet.mc_mo_nat)
        
        tplot.plot_wf(self.w90, rotate_mat, orb, self.w90.mp_grid_loc, grid)                         