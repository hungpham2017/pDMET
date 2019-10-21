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
from pyscf import gto, ao2mo, scf
      
        
class DMRG:
    '''
    Density Matrix Renormalization Group using CheMPS2 library 	
    '''	
    def __init__(self, mf, nroots):    
    
        self.norb   = mf.mo_coeff.shape[1] 
        self.nelec  = mf.mol.nelectron  
        self.spin   = mf.mol.spin
        self.nroots = nroots
        self.h1e    = reduce(np.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
        self.g2e    = ao2mo.incore.general(mf._eri, (mf.mo_coeff,)*4, compact=False).reshape(self.norb,self.norb,self.norb,self.norb)           
        self._D             = [200,500,1000,1000]
        self._convergence   = [1.e-4,1.e-5,1.e-6,1.e-8]
        self.noise          = 0.03
        self.max_sweep      = 100
        self._davidson_tol  = [1.e-3,1.e-4,1.e-5,1.e-6]
        self.CheMPS2print   = True             
            
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
        
    def run(self):
    
        # CheMPS2 calculation                
        Initializer = PyCheMPS2.PyInitialize()
        Initializer.Init()
        Group = 0
        orbirreps = np.zeros([self.norb], dtype=ctypes.c_int)
        HamCheMPS2 = PyCheMPS2.PyHamiltonian(self.norb, Group, orbirreps)
        #Feed the 1e and 2e integral (T and V)
        for orb1 in range(self.norb):
            for orb2 in range(self.norb):
                HamCheMPS2.setTmat(orb1, orb2, self.h1e[orb1, orb2])
                for orb3 in range(self.norb):
                    for orb4 in range(self.norb):
                        HamCheMPS2.setVmat(orb1, orb2, orb3, orb4, self.g2e[orb1, orb3, orb2, orb4]) #From chemist to physics notation        

        assert(self.nelec % 2 == 0)
        TwoS  = self.spin   
        Irrep = 0    
 
        if self.CheMPS2print == False:
            sys.stdout.flush()
            old_stdout = sys.stdout.fileno()
            new_stdout = os.dup(old_stdout)
            devnull = os.open('/dev/null', os.O_WRONLY)
            os.dup2(devnull, old_stdout)
            os.close(devnull)        
        
        Prob  = PyCheMPS2.PyProblem(HamCheMPS2, TwoS, self.nelec, Irrep)
        OptScheme = PyCheMPS2.PyConvergenceScheme(4) # 3 instructions
        #OptScheme.setInstruction(instruction, reduced virtual dimension D, energy convergence, maxSweeps, noisePrefactor, Davidson residual tolerance)
        OptScheme.set_instruction(0, self._D[0], self._convergence[0], 5  , self.noise,  self._davidson_tol[0])        
        OptScheme.set_instruction(1, self._D[1], self._convergence[1], 5  , self.noise,  self._davidson_tol[1])
        OptScheme.set_instruction(2, self._D[2], self._convergence[2], 5  , self.noise,  self._davidson_tol[2])
        OptScheme.set_instruction(3, self._D[3], self._convergence[3], self.max_sweep, 0.00,  self._davidson_tol[3]) # Last instruction a few iterations without noise

        theDMRG = PyCheMPS2.PyDMRG(Prob, OptScheme)
        EDMRG0 = theDMRG.Solve()  
        theDMRG.calc2DMandCorrelations()
        RDM2 = np.zeros([self.norb, self.norb, self.norb, self.norb], dtype=ctypes.c_double)
        for orb1 in range(self.norb):
            for orb2 in range(self.norb):
                for orb3 in range(self.norb):
                    for orb4 in range(self.norb):
                        RDM2[orb1, orb3, orb2, orb4] = theDMRG.get2DMA(orb1, orb2, orb3, orb4) #From physics to chemistry notation

        RDM1 = lib.einsum('ijkk->ij', RDM2)/(self.nelec - 1)
        
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
                rdm2 = np.zeros([self.norb, self.norb, self.norb, self.norb], dtype=ctypes.c_double)
                for orb1 in range(self.norb):
                    for orb2 in range(self.norb):
                        for orb3 in range(self.norb):
                            for orb4 in range(self.norb):
                                rdm2[orb1, orb3, orb2, orb4] = theDMRG.get2DMA(orb1, orb2, orb3, orb4) #From physics to chemistry notation
                                
                rdm1 = lib.einsum('ijkk->ij', rdm2)/(self.nelec - 1)
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
            
        if self.nroots > 1:
            return EDMRG , RDM1s, RDM2s
        else:
            return EDMRG0, RDM1 , RDM2       
        