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
import time
from pDMET.tools import tunix
     
        
class DIIS:
    '''
    Direct inversion in the iterative subspace method for correlation potential	
    Attributes:
        m       : start DIIS procedure at m th cycle
        n       : using n vector to construct the new vector
        
    Return:
        
    
    '''	
    def __init__(self, m, n, nkpts):     
        self.start_at = m
        self.nvector  = n 
        self.nkpts    = nkpts  
       
        # These are not variables:
        self._umat          = []
        self._err_umat      = []   
        self._errors        = [] 
        self._norm_errors   = [] 
        self._c             = None         
        self._max_idx       = None         
        
        # These are not variables:
        out = open('diis.temp', 'w')
        out.write('DIIS: %s\n' % (tunix.current_time()))        
        
    def get_errors(self, err_umat):
        ''' Get errors list and their norms
        '''
        
        errors = []
        norms  = []        
        for i in range(len(err_umat)):
            e  = err_umat[i]
            norm  = np.linalg.norm(e)
            errors.append(e) 
            norms.append(norm) 
            
        return errors, norms
        
    def extrapolate(self):
        ''' Condtruct the new vector using previous vectors following DIIS procedure
            ref: http://vergil.chemistry.gatech.edu/notes/diis/diis.pdf
        ''' 
      
        dim = self.nvector + 1    
        new_umats = []       
        umat   = np.asarray(self._umat)
        errors = np.asarray(self._errors)        
        B = np.einsum('ikab,jkba->ij', errors.transpose(0,1,3,2).conj(), errors).real  
        bigB = np.zeros([dim ,dim])
        bigB[:self.nvector,:self.nvector] = B
        bigB[:self.nvector,-1] = bigB[-1, :self.nvector] = -1        
        E = np.zeros(dim)
        E[-1] = -1 
        self._c = np.linalg.solve(bigB, E)[:-1] 
         
        if umat.ndim ==3:
            new_umat = np.einsum('iab,i->ab', umat, self._c)
        else:
            new_umat = np.einsum('ikab,i->kab', umat, self._c)
        
        return new_umat
        
    def update(self, cycle, umat, err_umat, projected=True):
        '''
        Return a new approximate vector or the same umat if cycle < m + n - 1
        '''	
        
        # Save the umat at the current cycle
        if cycle >= self.start_at - 1:
            if projected==False and err_umat.ndim == 2:         
                err_umat = err_umat + np.zeros([self.nkpts,err_umat.shape[0],err_umat.shape[0]])
            self._umat.append(umat)
            self._err_umat.append(err_umat)                    
            self._errors, self._norm_errors = self.get_errors(self._err_umat)
            
            # if the new umat has the highest error, 
            # then the umat with the second highest error will be removed from the list 
            if len(self._errors) == (self.nvector + 1):
                max_err_idx = np.argmax(self._norm_errors) 
                if max_err_idx == self.nvector: 
                    max_err_idx = np.argmax(self._norm_errors[:-1])   
                self._umat.pop(max_err_idx)
                self._err_umat.pop(max_err_idx)                
                self._errors.pop(max_err_idx)                 
                self._norm_errors.pop(max_err_idx) 
            
            # Print:
            nvector = len(self._err_umat)
            out = open('diis.temp', 'a')
            out.write('Cycle ' + str(cycle+1) + ':\n')
            if cycle < self.start_at + self.nvector - 1:
                out.write('  #          Error\n')
                out.write('  --------------------\n')                
                for i in range(nvector):
                    out.write(' %2d       %15.12f\n' % (i, self._norm_errors[i]))
                
        # Return umat
        if cycle >= self.start_at + self.nvector - 1:   
            new_umat = self.extrapolate()
            
            # Print
            out.write('  #          Error                  c\n')
            out.write('  --------------------------------------\n')  
            for i in range(nvector):
                out.write(' %2d       %15.12f          %8.5f\n' % (i, self._norm_errors[i], self._c[i]))
        else:
            new_umat = umat
    
        return new_umat 
        
        