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
import PyCheMPS2
import pyscf
from pyscf import lib, gto, ao2mo, scf, cc, fci, mcscf
# from pyscf import dmrgscf 
# from pyscf.shciscf import shci 


class QCsolvers:
    def __init__(self, solver, twoS = 0, e_shift = None, nroots = 1, state_percent = None, verbose = 0, memory = 4000):

        self.solver = solver  
        self.nroots = nroots        
        self.state_percent = state_percent
        self.SS =  0.5*twoS*(0.5*twoS + 1)      
        self.twoS = twoS 
        self.e_shift = e_shift

        self.mol = gto.Mole()
        self.mol.build(verbose = 0)
        self.mol.atom.append(('S', (0, 0, 0)))
        self.mol.nelectron = 2 + self.twoS
        self.mol.incore_anyway = True
        self.mol.max_memory = memory 
        self.mol.spin = self.twoS    
        
        if self.mol.spin == 0:        
            self.mf = scf.RHF(self.mol)    
        else:     
            self.mf = scf.ROHF(self.mol)      
            
        # Replace FCI solver by DMRG solver in CheMPS2 or BLOCK
        if self.solver in ['CASCI','DMRG-CI']:
            self.cas    = None
            self.molist = None   
            self.mo     = None  
            self.mo_nat = None             
            self.mc = mcscf.CASCI(self.mf, 2, 2)
            self.mc.verbose = verbose 
            self.mc.max_memory = memory          
        elif self.solver in ['CASSCF','DMRG-SCF']:
            self.cas    = None
            self.molist = None   
            self.mo     = None 
            self.mo_nat = None              
            self.mc = mcscf.CASSCF(self.mf, 2, 2)
            self.mc.verbose = verbose                 
        elif self.solver == 'FCI':          
            self.fs = None
            self.fs_conv_tol            = 1e-10   
            self.fs_conv_tol_residual   = None  
            self.ci = None  
            self.verbose = verbose
        elif self.solver == 'SHCI':          
            self.mch = None
            # self.fs_conv_tol            = 1e-10   
            # self.fs_conv_tol_residual   = None  
            self.ci = None  
            self.mo_coeff = None
            self.verbose = verbose
        elif self.solver == 'DMRG':
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

        elif self.solver == 'RCCSD': 
            self.cc = cc.CCSD(self.mf)
            self.t1 = None
            self.t2 = None
         
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
        if self.mol.spin == 0:        
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
        self.cc._scf = self.mf
        self.cc.mo_coeff = self.mf.mo_coeff
        self.cc.mo_occ = self.mf.mo_occ        
        self.cc.chkfile = self.mf.chkfile
        
        # Run RCCSD and get RDMs    
        if self.t1 is not None and self.t2 is not None:
            t1_0 = self.t1
            t2_0 = self.t2
        else:
            t1_0 = self.t1
            t2_0 = self.t2
        Ecorr, t1, t2 = self.cc.kernel(t1=t1_0, t2=t2_0)
        ECCSD = Ecorr + self.mf.e_tot
        self.t1 = t1
        self.t2 = t2
        if self.cc.converged == False: print('           WARNING: The solver is not converged')        
        RDM1_mo = self.cc.make_rdm1()
        RDM2_mo = self.cc.make_rdm2()  

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
                print('       Root %d: E(Solver) = %12.8f  E(Imp) = %12.8f  <S^2> = %12.8f' % (i, EDMRG[i], Imp_e, self.SS))                                    
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
                print('       Root %d: E(Solver) = %12.8f  E(Imp) = %12.8f  <S^2> = %12.8f' % (i, EFCI[i], Imp_e, SS))                 
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
                print('       Root %d: E(Solver) = %12.8f  E(Imp) = %12.8f  <S^2> = %12.8f' % (i, e[i], Imp_e, SS))                 
                tot_SS += SS                              
                RDM1.append(rdm1) 
                e_cell.append(Imp_e)              
            RDM1 = lib.einsum('i,ijk->jk',self.state_percent, RDM1) 
            e_cell = lib.einsum('i,i->',self.state_percent, e_cell)                
            self.SS = tot_SS/self.nroots  
               
        return (e_cell, ESHCI, RDM1)
        
#########################################        
########## CASSCF/CASCI solver ##########
#########################################          
    def CAS(self, solver = 'FCI'):
        '''
        CASCI/CASSCF with FCI or DMRG solver
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
    
        if self.mo is not None and self.solver in ['CASSCF', 'DMRG-SCF']: 
            e_tot, e_cas, civec = self.mc.kernel(self.mo)[:3]
        elif self.molist is not None: 
            mo = self.mc.sort_mo(self.molist)
            e_tot, e_cas, civec = self.mc.kernel(mo)[:3]            
        else: 
            e_tot, e_cas, civec = self.mc.kernel()[:3]
    
        if self.mc.converged == False: print('           WARNING: The solver is not converged')
        
        if solver not in ['CheMPS2', 'Block']:      
            self.SS = self.mc.fcisolver.spin_square(civec, cas_norb, self.mc.nelecas)[0]   
      
        # Save mo for the next iterations
        self.mo     = self.mc.mo_coeff           
        self.mo_nat = self.mc.cas_natorb()[0]
        
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

        # Compute the impurity energy             
        ImpurityEnergy = 0.50  * lib.einsum('ij,ij->',     RDM1[:Nimp,:], self.FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * lib.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])                                                 

        # Compute total energy    
        e_cell = self.kmf_ecore + ImpurityEnergy               

        return (e_cell, e_tot, RDM1)                