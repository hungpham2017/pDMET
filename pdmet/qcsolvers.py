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
import sys, os, ctypes
from functools import reduce
from pyscf import lib, gto, ao2mo, scf, cc, fci, mcscf, mrpt
from pyscf.mcscf import addons
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm

class QCsolvers:
    def __init__(self, solver, twoS=0, is_KROHF=False, e_shift=None, nroots=1, state_percent=None, verbose=0, memory=4000):

        self.solver = solver       
        self.state_percent = state_percent
        self.SS =  0.5*twoS*(0.5*twoS + 1)      
        self.twoS = twoS 
        self.e_shift = e_shift

        self._is_KROHF = is_KROHF
        self.mol = gto.Mole()
        self.mol.build(verbose = 0)
        self.mol.atom.append(('S', (0, 0, 0)))
        self.mol.nelectron = 2 + self.twoS
        self.mol.incore_anyway = True
        self.mol.max_memory = memory 
        self.mol.spin = self.twoS    
        
        
        if self.mol.spin == 0 and not self._is_KROHF:        
            self.mf = scf.RHF(self.mol)    
        else:     
            self.mf = scf.ROHF(self.mol)      

        # Replace FCI solver by DMRG solver in CheMPS2 or BLOCK
        if self.solver is 'CASCI':
            self.cas    = None
            self.molist = None   
            self.mo     = None  
            self.mo_nat = None   
            self.mc = mcscf.CASCI(self.mf, 2, 2)
            self.nroots = nroots   
            self.mc.verbose = verbose 
            self.mc.max_memory = memory
            self.mc.natorb = True
        elif self.solver is 'DMRG-CI':
            from pyscf import dmrgscf  
            self.cas    = None
            self.molist = None   
            self.mo     = None  
            self.mo_nat = None       
            self.mc = mcscf.CASCI(self.mf, 2, 2)
            self.nroots = nroots   
            self.mc.verbose = verbose 
            self.mc.max_memory = memory 
            self.mc.natorb = True            
        elif self.solver in ['CASSCF','SS-CASSCF','SA-CASSCF']:
            self.cas    = None
            self.molist = None   
            self.mo     = None 
            self.mo_nat = None              
            self.mc = mcscf.CASSCF(self.mf, 2, 2)
            self.nroots = nroots  
            self.mc.verbose = verbose
            self.mc.max_memory = memory 
            self.mc.natorb = True   
            self.chkfile = None    
        elif self.solver in ['DMRG-SCF', 'SS-DMRG-SCF','SA-DMRG-SCF']:
            from pyscf import dmrgscf 
            self.cas    = None
            self.molist = None   
            self.mo     = None 
            self.mo_nat = None              
            self.mc = mcscf.CASSCF(self.mf, 2, 2)
            self.nroots = nroots  
            self.mc.verbose = verbose 
            self.mc.max_memory = memory 
            self.mc.natorb = True 
        elif self.solver == 'FCI':          
            self.fs = None
            self.fs_conv_tol            = 1e-10   
            self.fs_conv_tol_residual   = None  
            self.ci = None  
            self.verbose = verbose
        elif self.solver == 'SHCI':   
            from pyscf.shciscf import shci 
            self.mch = None
            # self.fs_conv_tol            = 1e-10   
            # self.fs_conv_tol_residual   = None  
            self.ci = None  
            self.mo_coeff = None
            self.verbose = verbose
        elif self.solver == 'DMRG':
            from pyscf import PyCheMPS2
            self.CheMPS2print   = False
            self._D             = [200,500,1000,1000]
            self._convergence   = [1.e-4,1.e-5,1.e-6,1.e-8]
            self.noise          = 0.03
            self.max_sweep      = 100
            self._davidson_tol  = [1.e-3,1.e-4,1.e-5,1.e-6]
            if self.mol.verbose > 0: 
                self.CheMPS2print = True
            else:
                self.CheMPS2print = False                                
        elif self.solver == 'MP2': 
            self.mp2 = None
        elif self.solver == 'RCCSD': 
            self.cc = cc.CCSD(self.mf)
            self.t1 = None
            self.t2 = None
        elif self.solver == 'RCCSD_T': 
            self.cc = cc.CCSD(self.mf)
            self.t1 = None
            self.t2 = None
            self.verbose = verbose
         
    def initialize(self, kmf_ecore, OEI, TEI, JK, DMguess, Norb, Nel, Nimp, chempot=0.0):
        self.kmf_ecore      = kmf_ecore       
        self.OEI            = OEI
        self.TEI            = TEI
        self.FOCK           = OEI + JK
        self.DMguess        = DMguess
        self.Norb           = Norb
        self.Nel            = Nel
        self.Nimp           = Nimp
        chempot_array = np.zeros(Norb)
        chempot_array[:Nimp] = chempot
        self.chempot         = np.diag(chempot_array)
        

#####################################        
########## RHF/ROHF solver ##########
#####################################        
    def HF(self):
        '''
        Restricted open/close-shell Hartree-Fock (RHF/ROHF)
        '''        
        
        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy() - self.chempot 
     
        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: FOCKcopy
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
        self.mf.scf(self.DMguess)       
        DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        
        if (self.mf.converged == False):
            self.mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        
        ERHF = self.mf.e_tot
        RDM1 = self.mf.make_rdm1()
        JK   = self.mf.get_veff(None, dm=RDM1) 
        # To calculate the impurity energy, rescale the JK matrix with a factor 0.5 to avoid double counting: 0.5 * ( OEI + FOCK ) = OEI + 0.5 * JK
        if self.mol.spin == 0 and not self._is_KROHF:        
            ImpurityEnergy = 0.5*lib.einsum('ij,ij->', RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                            + 0.5*lib.einsum('ij,ij->', RDM1[:Nimp,:], JK[:Nimp,:])                                                  
        else:         
            ImpurityEnergy_a = 0.5*lib.einsum('ij,ij->', RDM1[0][:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                            + 0.5*lib.einsum('ij,ij->', RDM1[0][:Nimp,:], JK[0][:Nimp,:])        
            ImpurityEnergy_b = 0.5*lib.einsum('ij,ij->', RDM1[1][:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                            + 0.5*lib.einsum('ij,ij->', RDM1[1][:Nimp,:], JK[1][:Nimp,:])
            ImpurityEnergy =  ImpurityEnergy_a + ImpurityEnergy_b     
            RDM1 = RDM1.sum(axis=0)
                
        # Compute total energy        
        e_cell = self.kmf_ecore + ImpurityEnergy  

        return (e_cell, ERHF, RDM1) 
        
        
##################################
########## RCCSD solver ########## 
##################################           
    def MP2(self):
        '''
        MP2 solver
        '''        

        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy() - self.chempot
                
        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: FOCKcopy
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
        self.mf.scf(self.DMguess)       
        DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        if (self.mf.converged == False):
            self.mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)

        # MP2 calculation
        self.mp2 = self.mf.MP2()
        ecorr, t2 = self.mp2.kernel()
        EMP2 = self.mf.e_tot + ecorr
        RDM1_mo = self.mp2.make_rdm1(t2=t2)
        RDM2_mo = self.mp2.make_rdm2(t2=t2)  

        # Transform RDM1 , RDM2 to local basis
        RDM1 = lib.einsum('ap,pq->aq', self.mf.mo_coeff, RDM1_mo)
        RDM1 = lib.einsum('bq,aq->ab', self.mf.mo_coeff, RDM1)     
        RDM2 = lib.einsum('ap,pqrs->aqrs', self.mf.mo_coeff, RDM2_mo)
        RDM2 = lib.einsum('bq,aqrs->abrs', self.mf.mo_coeff, RDM2)
        RDM2 = lib.einsum('cr,abrs->abcs', self.mf.mo_coeff, RDM2)
        RDM2 = lib.einsum('ds,abcs->abcd', self.mf.mo_coeff, RDM2)        

        # Compute the impurity energy        
        ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])                                                

        # Compute total energy    
        e_cell = self.kmf_ecore + ImpurityEnergy 

        return (e_cell, EMP2, RDM1)  
        
        
##################################
########## RCCSD solver ########## 
##################################           
    def RCCSD(self):
        '''
        Couple-cluster Single-Double 
        '''        

        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy() - self.chempot
                
        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: FOCKcopy
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
        self.mf.scf(self.DMguess)       
        DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        if (self.mf.converged == False):
            self.mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)

        # Modify CC object using the correct mol, mf objects
        self.cc.mol  = self.mol         
        self.cc._scf = self.mf 
        self.cc.mo_coeff = self.mf.mo_coeff
        self.cc.mo_occ = self.mf.mo_occ
        self.cc._nocc = self.Nel//2
        self.cc._nmo = self.Norb
        self.cc.chkfile = self.mf.chkfile
        
        # Run RCCSD and get RDMs    
        if self.t1 is not None and self.t1.shape[0] == self.cc._nocc:            
            t1_0 = self.t1
            t2_0 = self.t2
        else:
            t1_0 = None
            t2_0 = None
                       
        Ecorr, t1, t2 = self.cc.kernel(t1=t1_0, t2=t2_0)
        ECCSD = Ecorr + self.mf.e_tot
        self.t1 = t1
        self.t2 = t2
        if not self.cc.converged: print('           WARNING: The solver is not converged')        
        RDM1_mo = self.cc.make_rdm1(t1=t1, t2=t2)
        RDM2_mo = self.cc.make_rdm2(t1=t1, t2=t2)  

        # Transform RDM1 , RDM2 to local basis
        RDM1 = lib.einsum('ap,pq->aq', self.mf.mo_coeff, RDM1_mo)
        RDM1 = lib.einsum('bq,aq->ab', self.mf.mo_coeff, RDM1)     
        RDM2 = lib.einsum('ap,pqrs->aqrs', self.mf.mo_coeff, RDM2_mo)
        RDM2 = lib.einsum('bq,aqrs->abrs', self.mf.mo_coeff, RDM2)
        RDM2 = lib.einsum('cr,abrs->abcs', self.mf.mo_coeff, RDM2)
        RDM2 = lib.einsum('ds,abcs->abcd', self.mf.mo_coeff, RDM2)        

        # Compute the impurity energy        
        ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])                                                

        # Compute total energy    
        e_cell = self.kmf_ecore + ImpurityEnergy 

        return (e_cell, ECCSD, RDM1)            


##################################
########## RCCSD(T) solver ########## 
##################################           
    def RCCSD_T(self):
        '''
        Couple-cluster Single-Double (T) with CCSD RDM 
        '''        

        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy() - self.chempot
                
        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: FOCKcopy
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
        self.mf.scf(self.DMguess)       
        DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        if (self.mf.converged == False):
            self.mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)

        # Modify CC object using the correct mol, mf objects
        self.cc.mol  = self.mol         
        self.cc._scf = self.mf 
        self.cc.mo_coeff = self.mf.mo_coeff
        self.cc.mo_occ = self.mf.mo_occ
        self.cc._nocc = self.Nel//2
        self.cc._nmo = self.Norb
        self.cc.chkfile = self.mf.chkfile
        
        # Run RCCSD and get RDMs    
        if self.t1 is not None and self.t1.shape[0] == self.cc._nocc:            
            t1_0 = self.t1
            t2_0 = self.t2
        else:
            t1_0 = None 
            t2_0 = None
                       
        Ecorr, t1, t2 = self.cc.kernel(t1=t1_0, t2=t2_0)
        ET = self.cc.ccsd_t()
        ECCSD_T = Ecorr + ET + self.mf.e_tot
        self.t1 = t1
        self.t2 = t2
        if not self.cc.converged: print('           WARNING: The solver is not converged')   
        
        # Get CCSD rdm
        RDM1_mo = self.cc.make_rdm1(t1=t1, t2=t2)
        RDM2_mo = self.cc.make_rdm2(t1=t1, t2=t2)  

        # Transform RDM1 , RDM2 to local basis
        RDM1 = lib.einsum('ap,pq->aq', self.mf.mo_coeff, RDM1_mo)
        RDM1 = lib.einsum('bq,aq->ab', self.mf.mo_coeff, RDM1)     
        RDM2 = lib.einsum('ap,pqrs->aqrs', self.mf.mo_coeff, RDM2_mo)
        RDM2 = lib.einsum('bq,aqrs->abrs', self.mf.mo_coeff, RDM2)
        RDM2 = lib.einsum('cr,abrs->abcs', self.mf.mo_coeff, RDM2)
        RDM2 = lib.einsum('ds,abcs->abcd', self.mf.mo_coeff, RDM2)        

        # Compute the impurity energy        
        ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])                                                

        # Compute total energy    
        e_cell = self.kmf_ecore + ImpurityEnergy 

        return (e_cell, ECCSD_T, RDM1)            
        
##################################
########## RCCSD(T) solver ########## 
##################################           
    def RCCSD_T_slow(self):
        '''
        Couple-cluster Single-Double (T) with full CCSD(T) rdm, very very expensive
        '''        

        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy() - self.chempot
                
        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: FOCKcopy
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
        self.mf.scf(self.DMguess)       
        DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        if (self.mf.converged == False):
            self.mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)

        # Modify CC object using the correct mol, mf objects
        self.cc.mol  = self.mol         
        self.cc._scf = self.mf 
        self.cc.mo_coeff = self.mf.mo_coeff
        self.cc.mo_occ = self.mf.mo_occ
        self.cc._nocc = self.Nel//2
        self.cc._nmo = self.Norb
        self.cc.chkfile = self.mf.chkfile
        
        # Run RCCSD and get RDMs    
        if self.t1 is not None and self.t1.shape[0] == self.cc._nocc:            
            t1_0 = self.t1
            t2_0 = self.t2
        else:
            t1_0 = None 
            t2_0 = None
                       
        Ecorr, t1, t2 = self.cc.kernel(t1=t1_0, t2=t2_0)
        ET = self.cc.ccsd_t()
        ECCSD_T = Ecorr + ET + self.mf.e_tot
        self.t1 = t1
        self.t2 = t2
        if not self.cc.converged: print('           WARNING: The solver is not converged')   
        
        # Get CCSD(T) rdm
        eris = self.cc.ao2mo()      # Consume too much memory, need to be fixed!
        l1, l2 = ccsd_t_lambda.kernel(self.cc, eris, t1, t2, verbose=self.verbose)[1:]
        RDM1_mo = ccsd_t_rdm.make_rdm1(self.cc, t1=t1, t2=t2, l1=l1, l2=l2, eris=eris)
        RDM2_mo = ccsd_t_rdm.make_rdm2(self.cc, t1=t1, t2=t2, l1=l1, l2=l2, eris=eris)

        # Transform RDM1 , RDM2 to local basis
        RDM1 = lib.einsum('ap,pq->aq', self.mf.mo_coeff, RDM1_mo)
        RDM1 = lib.einsum('bq,aq->ab', self.mf.mo_coeff, RDM1)     
        RDM2 = lib.einsum('ap,pqrs->aqrs', self.mf.mo_coeff, RDM2_mo)
        RDM2 = lib.einsum('bq,aqrs->abrs', self.mf.mo_coeff, RDM2)
        RDM2 = lib.einsum('cr,abrs->abcs', self.mf.mo_coeff, RDM2)
        RDM2 = lib.einsum('ds,abcs->abcd', self.mf.mo_coeff, RDM2)        

        # Compute the impurity energy        
        ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])                                                

        # Compute total energy    
        e_cell = self.kmf_ecore + ImpurityEnergy 

        return (e_cell, ECCSD_T, RDM1) 
        
#################################           
########## DMRG solver ########## 
#################################           
    @property
    def D(self):
        return self._D
    @D.setter
    def D(self, value):    
        assert isinstance(value, int)
        Dvec = [value//4, value//2, value, value] 
        print("from soler", Dvec)                
        self._D = Dvec
    @property
    def convergence(self):    
        return self._convergence
    @convergence.setter
    def convergence(self, value):    
        converg = [value*1.e4, value*1.e3, value*1.e2, value] 
        self._convergence = converg     
    @property
    def davidson_tol(self):    
        return self._davidson_tol
    @davidson_tol.setter
    def davidson_tol(self, value):    
        dav_tol = [value*1.e3, value*1.e2, value*1.e1, value] 
        self._davidson_tol = dav_tol    
        
    def DMRG(self):
        '''
        Density Matrix Renormalization Group using CheMPS2 library     
        '''    
        
        Norb = self.Norb
        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy() - self.chempot                                
                
        # CheMPS2 calculation                
        Initializer = PyCheMPS2.PyInitialize()
        Initializer.Init()
        Group = 0
        orbirreps = np.zeros([Norb], dtype=ctypes.c_int)
        HamCheMPS2 = PyCheMPS2.PyHamiltonian(Norb, Group, orbirreps)
        
        #Feed the 1e and 2e integral (T and V)
        for orb1 in range(Norb):
            for orb2 in range(Norb):
                HamCheMPS2.setTmat(orb1, orb2, FOCKcopy[orb1, orb2])
                for orb3 in range(Norb):
                    for orb4 in range(Norb):
                        HamCheMPS2.setVmat(orb1, orb2, orb3, orb4, self.TEI[orb1, orb3, orb2, orb4]) #From chemist to physics notation        

        assert(self.Nel % 2 == 0)
        TwoS  = self.twoS     
        Irrep = 0
       
        if self.CheMPS2print == False:
            sys.stdout.flush()
            old_stdout = sys.stdout.fileno()
            new_stdout = os.dup(old_stdout)
            devnull = os.open('/dev/null', os.O_WRONLY)
            os.dup2(devnull, old_stdout)
            os.close(devnull)        
        
        Prob  = PyCheMPS2.PyProblem(HamCheMPS2, TwoS, self.Nel, Irrep)
        OptScheme = PyCheMPS2.PyConvergenceScheme(4) # 3 instructions      
        #OptScheme.setInstruction(instruction, reduced virtual dimension D, energy convergence, maxSweeps, noisePrefactor, Davidson residual tolerance)
        OptScheme.set_instruction(0, self._D[0], self._convergence[0], 5  , self.noise,  self._davidson_tol[0])        
        OptScheme.set_instruction(1, self._D[1], self._convergence[1], 5  , self.noise,  self._davidson_tol[1])
        OptScheme.set_instruction(2, self._D[2], self._convergence[2], 5  , self.noise,  self._davidson_tol[2])
        OptScheme.set_instruction(3, self._D[3], self._convergence[3], self.max_sweep, 0.00,  self._davidson_tol[3]) # Last instruction a few iterations without noise

        theDMRG = PyCheMPS2.PyDMRG(Prob, OptScheme)
        EDMRG0 = theDMRG.Solve()  
        theDMRG.calc2DMandCorrelations()
        RDM2 = np.zeros([Norb, Norb, Norb, Norb], dtype=ctypes.c_double)
        for orb1 in range(Norb):
            for orb2 in range(Norb):
                for orb3 in range(Norb):
                    for orb4 in range(Norb):
                        RDM2[orb1, orb3, orb2, orb4] = theDMRG.get2DMA(orb1, orb2, orb3, orb4) #From physics to chemistry notation

        RDM1 = lib.einsum('ijkk->ij', RDM2)/(self.Nel - 1)
        
        # Excited state:
        if self.nroots > 1 :      
            theDMRG.activateExcitations(self.nroots - 1)
            EDMRG  = [EDMRG0]            
            RDM1s  = [RDM1]           
            RDM2s  = [RDM2]            
            for state in range(self.nroots - 1): 
                theDMRG.newExcitation(np.abs(EDMRG0));
                EDMRG.append(theDMRG.Solve())   
                theDMRG.calc2DMandCorrelations()  
                rdm2 = np.zeros([Norb, Norb, Norb, Norb], dtype=ctypes.c_double)
                for orb1 in range(Norb):
                    for orb2 in range(Norb):
                        for orb3 in range(Norb):
                            for orb4 in range(Norb):
                                rdm2[orb1, orb3, orb2, orb4] = theDMRG.get2DMA(orb1, orb2, orb3, orb4) #From physics to chemistry notation
                                
                rdm1 = lib.einsum('ijkk->ij', rdm2)/(self.Nel - 1)
                RDM1s.append(rdm1)                
                RDM2s.append(rdm2)    

        # theDMRG.deleteStoredMPS()
        theDMRG.deleteStoredOperators()
        del(theDMRG)
        del(OptScheme)
        del(Prob)
        del(HamCheMPS2)
        del(Initializer)    

        if self.CheMPS2print == False:        
            sys.stdout.flush()
            os.dup2(new_stdout, old_stdout)
            os.close(new_stdout)
            
        # Compute energy and RDM1      
        if self.nroots == 1:
            ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
            e_cell = self.kmf_ecore + ImpurityEnergy          
        else: 
            e_cell = []              
            for i in range(self.nroots):                    
                Imp_Energy_state = 0.50  * lib.einsum('ij,ij->',     RDM1s[i][:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', RDM2s[i][:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', RDM2s[i][:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', RDM2s[i][:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', RDM2s[i][:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
                Imp_e = self.kmf_ecore + Imp_Energy_state 
                print('       Root %d: E(Solver) = %12.8f  E(Imp) = %12.8f  <S^2> = %8.6f' % (i, EDMRG[i], Imp_e, self.SS))                                    
                e_cell.append(Imp_e)                
            RDM1 = lib.einsum('i,ijk->jk',self.state_percent, RDM1s) 
            e_cell = lib.einsum('i,i->',self.state_percent, e_cell)                     
            
        return (e_cell, EDMRG0, RDM1)     

########## FCI solver (not spin-adapted) ##########          
    def FCI(self):
        '''
        FCI solver from PySCF
        '''        

        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy() - self.chempot 
                
        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: FOCKcopy
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
        self.mf.scf(self.DMguess)       
        DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        if (self.mf.converged == False):
            self.mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)        
            
        # Create and solve the fci object      
        if self.e_shift == None:        
            self.fs = fci.FCI(self.mf, self.mf.mo_coeff)                
        else:                 
            self.fs = fci.addons.fix_spin_(fci.FCI(self.mf, self.mf.mo_coeff), self.e_shift)
            
        self.fs.verbose = self.verbose
        self.fs.conv_tol       = self.fs_conv_tol  
        self.fs.conv_tol_residual = self.fs_conv_tol_residual             
        self.fs.nroots = self.nroots 
        if self.ci is not None: 
            ci0 = self.ci
        else:
            ci0 = None
        EFCI, fcivec = self.fs.kernel(ci0=ci0)         
        self.ci = fcivec

        # Compute energy and RDM1      
        if self.nroots == 1:
            if not  self.fs.converged: print('           WARNING: The solver is not converged')
            self.SS = self.fs.spin_square(fcivec, self.Norb, self.mol.nelec)[0]
            RDM1_mo , RDM2_mo = self.fs.make_rdm12(fcivec, self.Norb, self.mol.nelec)
            # Transform RDM1 , RDM2 to local basis
            RDM1 = lib.einsum('ap,pq->aq', self.mf.mo_coeff, RDM1_mo)
            RDM1 = lib.einsum('bq,aq->ab', self.mf.mo_coeff, RDM1)     
            RDM2 = lib.einsum('ap,pqrs->aqrs', self.mf.mo_coeff, RDM2_mo)
            RDM2 = lib.einsum('bq,aqrs->abrs', self.mf.mo_coeff, RDM2)
            RDM2 = lib.einsum('cr,abrs->abcs', self.mf.mo_coeff, RDM2)
            RDM2 = lib.einsum('ds,abcs->abcd', self.mf.mo_coeff, RDM2)                
            
            ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
            e_cell = self.kmf_ecore + ImpurityEnergy         
        else:
            if not self.fs.converged.any(): print('           WARNING: The solver is not converged')
            tot_SS = 0 
            RDM1 = []  
            e_cell = []           
            for i, vec in enumerate(fcivec):
                SS = self.fs.spin_square(vec, self.Norb, self.mol.nelec)[0]   
                rdm1_mo , rdm2_mo = self.fs.make_rdm12(vec, self.Norb, self.mol.nelec)
                # Transform rdm1 , rdm2 to local basis
                rdm1 = lib.einsum('ap,pq->aq', self.mf.mo_coeff, rdm1_mo)
                rdm1 = lib.einsum('bq,aq->ab', self.mf.mo_coeff, rdm1)     
                rdm2 = lib.einsum('ap,pqrs->aqrs', self.mf.mo_coeff, rdm2_mo)
                rdm2 = lib.einsum('bq,aqrs->abrs', self.mf.mo_coeff, rdm2)
                rdm2 = lib.einsum('cr,abrs->abcs', self.mf.mo_coeff, rdm2)
                rdm2 = lib.einsum('ds,abcs->abcd', self.mf.mo_coeff, rdm2)                    
                Imp_Energy_state = 0.50  * lib.einsum('ij,ij->',     rdm1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
                Imp_e = self.kmf_ecore + Imp_Energy_state               
                print('       Root %d: E(Solver) = %12.8f  E(Imp) = %12.8f  <S^2> = %8.6f' % (i, EFCI[i], Imp_e, SS))                 
                tot_SS += SS                              
                RDM1.append(rdm1) 
                e_cell.append(Imp_e)              
            RDM1 = lib.einsum('i,ijk->jk',self.state_percent, RDM1) 
            e_cell = lib.einsum('i,i->',self.state_percent, e_cell)                
            self.SS = tot_SS/self.nroots  
               
        return (e_cell, EFCI, RDM1)
        
########## SHCI solver ##########          
    def SHCI(self):
        '''
        SHCI solver from PySCF
        '''        

        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy()  
                
        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: FOCKcopy
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
        self.mf.scf(self.DMguess)       
        DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        if (self.mf.converged == False):
            self.mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)        
            
        # Create and solve the SHCI object      
        mch = shci.SHCISCF(self.mf, self.Norb, self.mol.nelectron)
        mch.fcisolver.mpiprefix = ''
        mch.fcisolver.nPTiter = 0 # Turn off perturbative calc.
        mch.fcisolver.sweep_iter = [ 0, 3 ]
        # Setting large epsilon1 thresholds highlights improvement from perturbation.
        mch.fcisolver.sweep_epsilon = [ 1e-3, 0.5e-3 ]

        # Run a single SHCI iteration with perturbative correction.
        # mch.fcisolver.stochastic = False # Turns on deterministic PT calc.
        # mch.fcisolver.epsilon2 = 1e-8
        # shci.writeSHCIConfFile( mch.fcisolver, [self.mol.nelectron/2,self.mol.nelectron/2] , False )
        # shci.executeSHCI( mch.fcisolver )

        # Open and get the energy from the binary energy file shci.e.
        # file1 = open(os.path.join(mch.fcisolver.runtimeDir, "%s/shci.e"%(mch.fcisolver.prefix)), "rb")
        # format = ['d']*1
        # format = ''.join(format)
        # e_PT = struct.unpack(format, file1.read())

        if self.ci is not None: 
            ci0 = self.ci
            mo_coeff = self.mo_coeff
        else:
            ci0 = None
            mo_coeff = None
        e_noPT, e_cas, fcivec, mo_coeff = mch.mc1step(mo_coeff=mo_coeff, ci0=ci0)[:4] 
        ESHCI = e_noPT #TODO: this is not correct, will be modified later
        self.ci = fcivec
        self.mo_coeff = mo_coeff
        
        # Compute energy and RDM1      
        if self.nroots == 1:
            if mch.converged == False: print('           WARNING: The solver is not converged')
            self.SS = mch.fcisolver.spin_square(fcivec, self.Norb, self.mol.nelec)[0]  
            RDM1_mo , RDM2_mo = mch.fcisolver.make_rdm12(fcivec, self.Norb, self.mol.nelec)
            # Transform RDM1 , RDM2 to local basis
            RDM1 = lib.einsum('ap,pq->aq', self.mf.mo_coeff, RDM1_mo)
            RDM1 = lib.einsum('bq,aq->ab', self.mf.mo_coeff, RDM1)     
            RDM2 = lib.einsum('ap,pqrs->aqrs', self.mf.mo_coeff, RDM2_mo)
            RDM2 = lib.einsum('bq,aqrs->abrs', self.mf.mo_coeff, RDM2)
            RDM2 = lib.einsum('cr,abrs->abcs', self.mf.mo_coeff, RDM2)
            RDM2 = lib.einsum('ds,abcs->abcd', self.mf.mo_coeff, RDM2)                
            
            ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
            e_cell = self.kmf_ecore + ImpurityEnergy         
        else:
            if mch.converged.any() == False: print('           WARNING: The solver is not converged')
            tot_SS = 0 
            RDM1 = []  
            e_cell = []           
            for i, vec in enumerate(fcivec):
                SS = mch.fcisolver.spin_square(fcivec, self.Norb, self.mol.nelec)[0]   
                rdm1_mo , rdm2_mo = mch.fcisolver.make_rdm12(fcivec, self.Norb, self.mol.nelec)
                # Transform rdm1 , rdm2 to local basis
                rdm1 = lib.einsum('ap,pq->aq', self.mf.mo_coeff, rdm1_mo)
                rdm1 = lib.einsum('bq,aq->ab', self.mf.mo_coeff, rdm1)     
                rdm2 = lib.einsum('ap,pqrs->aqrs', self.mf.mo_coeff, rdm2_mo)
                rdm2 = lib.einsum('bq,aqrs->abrs', self.mf.mo_coeff, rdm2)
                rdm2 = lib.einsum('cr,abrs->abcs', self.mf.mo_coeff, rdm2)
                rdm2 = lib.einsum('ds,abcs->abcd', self.mf.mo_coeff, rdm2)                    
                Imp_Energy_state = 0.50  * lib.einsum('ij,ij->',     rdm1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
                Imp_e = self.kmf_ecore + Imp_Energy_state               
                print('       Root %d: E(Solver) = %12.8f  E(Imp) = %12.8f  <S^2> = %8.6f' % (i, e[i], Imp_e, SS))                 
                tot_SS += SS                              
                RDM1.append(rdm1) 
                e_cell.append(Imp_e)              
            RDM1 = lib.einsum('i,ijk->jk',self.state_percent, RDM1) 
            e_cell = lib.einsum('i,i->',self.state_percent, e_cell)                
            self.SS = tot_SS/self.nroots  
               
        return (e_cell, ESHCI, RDM1)
              

#########################################        
##########     CASCI solver    ##########
#########################################          
    def CASCI(self, solver = 'FCI', nevpt2_roots=None, nevpt2_nroots=10):
        '''
        CASCI with FCI or DMRG solver for a multiple roots calculation
        '''        

        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy() - self.chempot
                
        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: FOCKcopy
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
        self.mf.scf(self.DMguess)       
        DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        if (self.mf.converged == False):
            self.mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
    
        if self.cas == None:
            cas_nelec = self.Nel
            cas_norb = self.Norb
        else:
            cas_nelec = self.cas[0]
            cas_norb = self.cas[1]

        # Updating mc object from the new mf, mol objects     
        self.mc.mol = self.mol   
        self.mc._scf = self.mf 
        self.mc.ncas = cas_norb
        nelecb = (cas_nelec - self.mol.spin)//2
        neleca = cas_nelec - nelecb
        self.mc.nelecas = (neleca, nelecb)
        ncorelec = self.mol.nelectron - (self.mc.nelecas[0] + self.mc.nelecas[1])     
        assert(ncorelec % 2 == 0)
        self.mc.ncore = ncorelec // 2       
        self.mc.mo_coeff = self.mf.mo_coeff
        self.mc.mo_energy = self.mf.mo_energy
            
        # Define FCI solver
        if solver == 'CheMPS2':      
            self.mc.fcisolver = dmrgscf.CheMPS2(self.mol)
        elif solver == 'FCI' and self.e_shift != None:         
            target_SS = 0.5*self.twoS*(0.5*self.twoS + 1)
            self.mc.fix_spin_(shift = self.e_shift, ss = target_SS)                  
    
        self.mc.fcisolver.nroots = self.nroots 
        if self.mo is not None: 
            mo = self.mo
        elif self.molist is not None: 
            mo = mcscf.sort_mo(self.mc, self.mc.mo_coeff, self.molist, 0)
        else: 
            mo = self.mc.mo_coeff
        e_tot, e_cas, fcivec = self.mc.kernel(mo)[:3] 
        if not self.mc.converged: print('           WARNING: The solver is not converged')
        
        # Save mo for the next iterations
        self.mo_nat     = self.mc.mo_coeff           
   
        # Compute energy and RDM1      
        if self.nroots == 1:
            civec = fcivec
            self.SS = self.mc.fcisolver.spin_square(civec, self.Norb, self.mol.nelec)[0]
            RDM1_mo , RDM2_mo = self.mc.fcisolver.make_rdm12(civec, self.Norb, self.mol.nelec)
            
            ###### Get RDM1 + RDM2 #####
            core_norb = self.mc.ncore    
            core_MO = self.mc.mo_coeff[:,:core_norb]
            active_MO = self.mc.mo_coeff[:,core_norb:core_norb+cas_norb] 
            casdm1_mo, casdm2_mo = self.mc.fcisolver.make_rdm12(self.mc.ci, cas_norb, self.mc.nelecas) #in CAS(MO) space    

            # Transform the casdm1_mo to local basis
            casdm1 = lib.einsum('ap,pq->aq', active_MO, casdm1_mo)
            casdm1 = lib.einsum('bq,aq->ab', active_MO, casdm1)
            coredm1 = np.dot(core_MO, core_MO.T) * 2 #in local basis
            RDM1 = coredm1 + casdm1   
            
            # Transform the casdm2_mo to local basis
            casdm2 = lib.einsum('ap,pqrs->aqrs', active_MO, casdm2_mo)
            casdm2 = lib.einsum('bq,aqrs->abrs', active_MO, casdm2)
            casdm2 = lib.einsum('cr,abrs->abcs', active_MO, casdm2)
            casdm2 = lib.einsum('ds,abcs->abcd', active_MO, casdm2)    
        
            coredm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb])
            coredm2 += lib.einsum('pq,rs-> pqrs',coredm1,coredm1)
            coredm2 -= 0.5*lib.einsum('ps,rq-> pqrs',coredm1,coredm1)

            effdm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb])
            effdm2 += 2*lib.einsum('pq,rs-> pqrs',casdm1,coredm1)
            effdm2 -= lib.einsum('ps,rq-> pqrs',casdm1,coredm1)                
                        
            RDM2 = coredm2 + casdm2 + effdm2               
            
            ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
            e_cell = self.kmf_ecore + ImpurityEnergy         
        else:
            tot_SS = 0 
            RDM1 = []  
            e_cell = []           
            for i, civec in enumerate(fcivec):
                SS = self.mc.fcisolver.spin_square(civec, cas_norb, self.mc.nelecas)[0]
                
                ###### Get RDM1 + RDM2 #####
                core_norb = self.mc.ncore     
                core_MO = self.mc.mo_coeff[:,:core_norb]
                active_MO = self.mc.mo_coeff[:,core_norb:core_norb+cas_norb] 
                casdm1_mo, casdm2_mo = self.mc.fcisolver.make_rdm12(civec, cas_norb, self.mc.nelecas) #in CAS(MO) space    

                # Transform the casdm1_mo to local basis
                casdm1 = lib.einsum('ap,pq->aq', active_MO, casdm1_mo)
                casdm1 = lib.einsum('bq,aq->ab', active_MO, casdm1)
                coredm1 = np.dot(core_MO, core_MO.T) * 2 #in local basis
                rdm1 = coredm1 + casdm1   
                
                # Transform the casdm2_mo to local basis
                casdm2 = lib.einsum('ap,pqrs->aqrs', active_MO, casdm2_mo)
                casdm2 = lib.einsum('bq,aqrs->abrs', active_MO, casdm2)
                casdm2 = lib.einsum('cr,abrs->abcs', active_MO, casdm2)
                casdm2 = lib.einsum('ds,abcs->abcd', active_MO, casdm2)    
            
                coredm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb])
                coredm2 += lib.einsum('pq,rs-> pqrs',coredm1,coredm1)
                coredm2 -= 0.5*lib.einsum('ps,rq-> pqrs',coredm1,coredm1)

                effdm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb])
                effdm2 += 2*lib.einsum('pq,rs-> pqrs',casdm1,coredm1)
                effdm2 -= lib.einsum('ps,rq-> pqrs',casdm1,coredm1)                
                            
                rdm2 = coredm2 + casdm2 + effdm2         
                
                Imp_Energy_state = 0.50  * lib.einsum('ij,ij->',     rdm1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
                Imp_e = self.kmf_ecore + Imp_Energy_state               
                print('       Root %d: E(Solver) = %12.8f  E(Imp) = %12.8f  <S^2> = %8.6f' % (i, e_tot[i], Imp_e, SS))                 
                tot_SS += SS                              
                RDM1.append(rdm1) 
                e_cell.append(Imp_e)              
            RDM1 = lib.einsum('i,ijk->jk',self.state_percent, RDM1) 
            e_cell = lib.einsum('i,i->',self.state_percent, e_cell)                
            self.SS = tot_SS/self.nroots  
        
        if nevpt2_roots is not None:
            # Run a CASCI for an excited-state wfn
            if solver == 'FCI' and self.e_shift is not None: 
                mc_CASCI = mcscf.CASCI(self.mf, cas_norb, cas_nelec)
                mc_CASCI = mc_CASCI.fix_spin_(shift=self.e_shift, ss=target_SS) 
            else:
                mc_CASCI = mcscf.CASCI(self.mf, cas_norb, cas_nelec)
                
            mc_CASCI.fcisolver.nroots = nevpt2_nroots
            fcivec = mc_CASCI.kernel(self.mc.mo_coeff)[2]

            # Run NEVPT2
            e_casci_nevpt = []
            for root in nevpt2_roots:
                SS = mc_CASCI.fcisolver.spin_square(fcivec[root], cas_norb, self.mc.nelecas)[0]
                e_corr = mrpt.NEVPT(mc_CASCI, root).kernel()
                if not isinstance(mc_CASCI.e_tot, np.ndarray):
                    e_CASCI = mc_CASCI.e_tot
                    e_nevpt = e_CASCI + e_corr
                else:
                    e_CASCI = mc_CASCI.e_tot[root]
                    e_nevpt = e_CASCI + e_corr
                e_casci_nevpt.append([SS, e_CASCI, e_nevpt])
                
            #Pack E_CASSCF and E_NEVPT2 into a tuple of e_tot
            e_casci_nevpt = np.asarray(e_casci_nevpt)
            e_tot = (e_tot, e_casci_nevpt)
                
        return (e_cell, e_tot, RDM1)
        
#########################################        
##########     CASSCF solver    ##########
#########################################  

    def CASSCF(self, solver='FCI', state_specific_=None, state_average_=None, state_average_mix_=None, nevpt2_roots=None, nevpt2_nroots=10, nevpt2_spin=0.0):
        '''
        CASSCF with FCI or DMRG solver:
            - Ground state
            - State-specfic
            - State-average
        state_specific_ is used to pass the state_id to the solver
        state_average_ is used to pass the weights to the solver
        '''        

        Nimp = self.Nimp
        FOCKcopy = self.FOCK.copy() - self.chempot
        self.mol.nelectron = self.Nel
        self.mf.__init__(self.mol)
        self.mf.get_hcore = lambda *args: FOCKcopy
        self.mf.get_ovlp = lambda *args: np.eye(self.Norb)
        self.mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
        self.mf.scf(self.DMguess)       
        DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)
        if (self.mf.converged == False):
            self.mf.newton().kernel(dm0=DMloc)
            DMloc = np.dot(np.dot(self.mf.mo_coeff, np.diag(self.mf.mo_occ)), self.mf.mo_coeff.T)

        if self.cas is None:
            cas_nelec = self.Nel
            cas_norb = self.Norb
        else:
            cas_nelec = self.cas[0]
            cas_norb = self.cas[1]

        # Updating mc object from the new mf, mol objects    
        if state_specific_ is not None and state_average_ is None:
            state_id = state_specific_
            if not 'FakeCISolver' in str(self.mc.fcisolver):
                self.mc = self.mc.state_specific_(state_id)
        elif state_specific_ is None and state_average_ is not None and state_average_mix_ is None:
            weights = state_average_
            if not 'FakeCISolver' in str(self.mc.fcisolver):
                self.mc = self.mc.state_average_(weights)
        elif state_average_mix_ is not None:
            solver1, solver2, weights = state_average_mix_
            mcscf.state_average_mix_(self.mc, [solver1, solver2], weights)
        else:
            state_id = 0
            self.nroots = 1
            self.mc.fcisolver.nroots = self.nroots 
        
        self.mc.mol = self.mol   
        self.mc._scf = self.mf 
        self.mc.ncas = cas_norb
        nelecb = (cas_nelec - self.mol.spin)//2
        neleca = cas_nelec - nelecb
        self.mc.nelecas = (neleca, nelecb)
        ncorelec = self.mol.nelectron - (self.mc.nelecas[0] + self.mc.nelecas[1])     
        assert(ncorelec % 2 == 0)
        self.mc.ncore = ncorelec // 2       
        self.mc.mo_coeff = self.mf.mo_coeff
        self.mc.mo_energy = self.mf.mo_energy
        
        # Define FCI solver
        if solver == 'CheMPS2':      
            self.mc.fcisolver = dmrgscf.CheMPS2(self.mol)
        elif solver == 'FCI' and self.e_shift is not None and state_average_mix_ is None:         
            target_SS = 0.5*self.twoS*(0.5*self.twoS + 1)
            self.mc.fix_spin_(shift=self.e_shift, ss=target_SS)                  
            
        if self.mo is not None: 
            mo = self.mo
        elif self.molist is not None: 
            if self.chkfile is not None:
                mo = lib.chkfile.load(self.chkfile, 'mcscf/mo_coeff')
            else:
                mo = self.mc.mo_coeff
            mo = mcscf.sort_mo(self.mc, mo, self.molist, base=0)
        else: 
            mo = self.mc.mo_coeff

        e_tot, e_cas, fcivec = self.mc.kernel(mo)[:3] 
        if state_specific_ is None and state_average_ is not None: 
            e_tot = np.asarray(self.mc.e_states)
            
        if not self.mc.converged: print('           WARNING: The solver is not converged')
        
        # Save mo for the next iterations
        self.mo_nat = self.mc.mo_coeff           
        self.mo = self.mc.mo_coeff  
        
    
        # Compute energy and RDM1      
        if self.nroots == 1 or state_specific_ is not None:
            civec = fcivec
            self.SS, spin_multiplicity = mcscf.spin_square(self.mc)
            
            ###### Get RDM1 + RDM2 #####
            core_norb = self.mc.ncore    
            core_MO = self.mc.mo_coeff[:,:core_norb]
            active_MO = self.mc.mo_coeff[:,core_norb:core_norb+cas_norb] 
            casdm1_mo, casdm2_mo = self.mc.fcisolver.make_rdm12(civec, cas_norb, self.mc.nelecas) #in CAS(MO) space    

            # Transform the casdm1_mo to local basis
            casdm1 = lib.einsum('ap,pq->aq', active_MO, casdm1_mo)
            casdm1 = lib.einsum('bq,aq->ab', active_MO, casdm1)
            coredm1 = np.dot(core_MO, core_MO.T) * 2 #in local basis
            RDM1 = coredm1 + casdm1   
            
            # Transform the casdm2_mo to local basis
            casdm2 = lib.einsum('ap,pqrs->aqrs', active_MO, casdm2_mo)
            casdm2 = lib.einsum('bq,aqrs->abrs', active_MO, casdm2)
            casdm2 = lib.einsum('cr,abrs->abcs', active_MO, casdm2)
            casdm2 = lib.einsum('ds,abcs->abcd', active_MO, casdm2)    
        
            coredm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb])
            coredm2 += lib.einsum('pq,rs-> pqrs',coredm1,coredm1)
            coredm2 -= 0.5*lib.einsum('ps,rq-> pqrs',coredm1,coredm1)

            effdm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb])
            effdm2 += 2*lib.einsum('pq,rs-> pqrs',casdm1,coredm1)
            effdm2 -= lib.einsum('ps,rq-> pqrs',casdm1,coredm1)                
                        
            RDM2 = coredm2 + casdm2 + effdm2               
            
            ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
            e_cell = self.kmf_ecore + ImpurityEnergy         
            print('       State %d: E(Solver) = %12.8f  E(Imp) = %12.8f  <S^2> = %8.6f' % (state_id, e_tot, ImpurityEnergy, self.SS))  
        elif state_average_ is not None:
            tot_SS = 0 
            RDM1 = []  
            e_cell = []           
            rdm1s, rdm2s = self.mc.fcisolver.states_make_rdm12(fcivec, cas_norb, self.mc.nelecas)
            SSs, spin_multiplicities = self.mc.fcisolver.states_spin_square(fcivec, cas_norb, self.mc.nelecas) 
            
            for i in range(len(weights)):
                SS, spin_multiplicity = SSs[i], spin_multiplicities[i]

                ###### Get RDM1 + RDM2 #####
                core_norb = self.mc.ncore    
                core_MO = self.mc.mo_coeff[:,:core_norb]
                active_MO = self.mc.mo_coeff[:,core_norb:core_norb+cas_norb] 
                casdm1_mo, casdm2_mo = rdm1s[i], rdm2s[i]  

                # Transform the casdm1_mo to local basis
                casdm1 = lib.einsum('ap,pq->aq', active_MO, casdm1_mo)
                casdm1 = lib.einsum('bq,aq->ab', active_MO, casdm1)
                coredm1 = np.dot(core_MO, core_MO.T) * 2 #in local basis
                rdm1 = coredm1 + casdm1   
                
                # Transform the casdm2_mo to local basis
                casdm2 = lib.einsum('ap,pqrs->aqrs', active_MO, casdm2_mo)
                casdm2 = lib.einsum('bq,aqrs->abrs', active_MO, casdm2)
                casdm2 = lib.einsum('cr,abrs->abcs', active_MO, casdm2)
                casdm2 = lib.einsum('ds,abcs->abcd', active_MO, casdm2)    
            
                coredm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb])
                coredm2 += lib.einsum('pq,rs-> pqrs',coredm1,coredm1)
                coredm2 -= 0.5*lib.einsum('ps,rq-> pqrs',coredm1,coredm1)

                effdm2 = np.zeros([self.Norb, self.Norb, self.Norb, self.Norb])
                effdm2 += 2*lib.einsum('pq,rs-> pqrs',casdm1,coredm1)
                effdm2 -= lib.einsum('ps,rq-> pqrs',casdm1,coredm1)                
                            
                rdm2 = coredm2 + casdm2 + effdm2         
                
                Imp_Energy_state = 0.50  * lib.einsum('ij,ij->',     rdm1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                              + 0.125 * lib.einsum('ijkl,ijkl->', rdm2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])
                Imp_e = self.kmf_ecore + Imp_Energy_state  
                if state_average_ is not None:
                    print('       State %d (%5.3f): E(Solver) = %12.8f  E(Imp) = %12.8f  <S^2> = %8.6f' % (i, weights[i], e_tot[i], Imp_e, SS))  

                tot_SS += SS                              
                RDM1.append(rdm1) 
                e_cell.append(Imp_e)    

            RDM1 = lib.einsum('i,ijk->jk',state_average_, RDM1) 
            e_cell = lib.einsum('i,i->',state_average_, e_cell) 
            # self.SS = tot_SS/self.nroots  
            
        if nevpt2_roots is not None:

            # Run a CASCI for an excited-state wfn
            # if solver == 'FCI' and self.e_shift is not None: 
                # mc_CASCI = mcscf.CASCI(self.mf, cas_norb, cas_nelec)
                # mc_CASCI = mc_CASCI.fix_spin_(shift=self.e_shift, ss=target_SS) 
            # else:
                # mc_CASCI = mcscf.CASCI(self.mf, cas_norb, cas_nelec)
                
            self.mf.spin = nevpt2_spin
            nelecb = (cas_nelec - self.mf.spin)//2
            neleca = cas_nelec - nelecb
            nelecas = (neleca, nelecb)
            mc_CASCI = mcscf.CASCI(self.mf, cas_norb, (neleca, nelecb))
            mc_CASCI.fcisolver.nroots = nevpt2_nroots
            fcivec = mc_CASCI.kernel(self.mc.mo_coeff)[2]

            # Run NEVPT2
            e_casci_nevpt = []
            from pyscf.fci import cistring
            print("=====================================")
            if len(nevpt2_roots) > len(fcivec): nevpt2_roots = np.arange(len(fcivec))
            for root in nevpt2_roots:
                ci = fcivec[root]
                SS = mc_CASCI.fcisolver.spin_square(ci, cas_norb, nelecas)[0]
                e_corr = mrpt.NEVPT(mc_CASCI, root).kernel()
                if not isinstance(mc_CASCI.e_tot, np.ndarray):
                    e_CASCI = mc_CASCI.e_tot
                    e_nevpt = e_CASCI + e_corr
                else:
                    e_CASCI = mc_CASCI.e_tot[root]
                    e_nevpt = e_CASCI + e_corr
                e_casci_nevpt.append([SS, e_CASCI, e_nevpt])
                
                ''' TODO: NEED TO BE GENERALIZED LATER '''
                rdm1 = mc_CASCI.fcisolver.make_rdm12(ci, cas_norb, nelecas)[0]
                e, v = np.linalg.eig(rdm1)
                # Find the two SDs with most contribution 
                strsa = np.asarray(cistring.make_strings(range(cas_norb), neleca))
                strsb = np.asarray(cistring.make_strings(range(cas_norb), nelecb))    
                na = len(strsa)
                nb = len(strsb)
                
                idx_1st_max = abs(ci).argmax()
                c1 = ci.flatten()[idx_1st_max]
                stra_1st = strsa[idx_1st_max // nb]
                strb_1st = strsb[idx_1st_max % nb ]
                
                abs_fcivec = abs(ci).flatten()
                abs_fcivec[idx_1st_max] = 0.0
                idx_2nd_max = abs_fcivec.argmax()
                c2 = ci.flatten()[idx_2nd_max]
                stra_2nd = strsa[idx_2nd_max // nb]
                strb_2nd = strsb[idx_2nd_max % nb ]
                
                abs_fcivec[idx_2nd_max] = 0.0
                idx_3rd_max = abs_fcivec.argmax()
                c3 = ci.flatten()[idx_3rd_max]
                stra_3rd = strsa[idx_3rd_max // nb]
                strb_3rd = strsb[idx_3rd_max % nb ]

                abs_fcivec[idx_3rd_max] = 0.0
                idx_4th_max = abs_fcivec.argmax()
                c4 = ci.flatten()[idx_4th_max]
                stra_4th = strsa[idx_4th_max // nb]
                strb_4th = strsb[idx_4th_max % nb ]
                
                print("== State {0:d}: {1:2.4f}|{2:s},{3:s}> + {4:2.4f}|{5:s},{6:s}> + {7:2.4f}|{8:s},{9:s}> + {10:2.4f}|{11:s},{12:s}>".format(root, c1, bin(stra_1st)[2:], bin(strb_1st)[2:], c2, bin(stra_2nd)[2:], bin(strb_2nd)[2:], c3, bin(stra_3rd)[2:], bin(strb_3rd)[2:], c4, bin(stra_4th)[2:], bin(strb_4th)[2:]))
                print("   Occupancy:", e)
                
                ''' TODO: NEED TO BE GENERALIZED LATER '''
            print("=====================================") 
                
            #Pack E_CASSCF and E_NEVPT2 into a tuple of e_tot
            e_casci_nevpt = np.asarray(e_casci_nevpt)
            e_tot = (e_tot, e_casci_nevpt)
                
        return (e_cell, e_tot, RDM1)  
        
        