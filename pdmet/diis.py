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
from pyscf import lib
import time
from pdmet.tools import tunix
     
        
class DIIS:
    '''
    Direct inversion in the iterative subspace method for correlation potential	
    Attributes:
        m       : start DIIS procedure at m th cycle
        n       : using n vector to construct the new vector
        
    Return:
        
    
    '''	
    def __init__(self, m, n):     
        self.start_at = m - 1
        self.nvector  = n  
       
        # These are not variables:
        self._obj_mat           = []
        self._residual_mat      = []   
        self._errors        = [] 
        self._norm_errors   = [] 
        self._c             = None         
        self._max_idx       = None         
        
        # These are not variables:
        out = open('diis.temp', 'w')
        out.write('DIIS: %s\n' % (tunix.current_time()))        
        
    def get_errors(self, residual_mat):
        ''' Get errors list and their norms
        '''
        
        errors = []
        norms  = []        
        for i in range(len(residual_mat)):
            e  = residual_mat[i]
            norm  = np.linalg.norm(e)
            errors.append(e) 
            norms.append(norm) 
            
        return errors, norms
        
    def extrapolate(self):
        ''' Condtruct the new vector using previous vectors following DIIS procedure
            ref: http://vergil.chemistry.gatech.edu/notes/diis/diis.pdf
        ''' 
      
        dim = self.nvector + 1     
        obj_mat   = np.asarray(self._obj_mat)
        errors = np.asarray(self._errors)        
        B = lib.einsum('iab,jab->ij', errors, errors.conj())  
        bigB = np.zeros([dim ,dim])
        bigB[:self.nvector,:self.nvector] = B
        bigB[:self.nvector,-1] = bigB[-1, :self.nvector] = -1        
        E = np.zeros(dim)
        E[-1] = -1 
        self._c = np.linalg.solve(bigB, E)[:-1] 
        new_obj_mat = lib.einsum('iab,i->ab', obj_mat, self._c)
        return new_obj_mat
        
    def update(self, cycle, obj_mat, residual_mat):
        '''
        Return a new approximate vector or the same umat if cycle < m + n - 1
        '''	
        
        # Save the umat at the current cycle
        if cycle >= self.start_at - 1:
            self._obj_mat.append(obj_mat)
            self._residual_mat.append(residual_mat)                    
            self._errors, self._norm_errors = self.get_errors(self._residual_mat)
            
            # if the new umat has the highest error, 
            # then the umat with the second highest error will be removed from the list 
            if len(self._errors) == (self.nvector + 1):
                max_err_idx = np.argmax(self._norm_errors) 
                if max_err_idx == self.nvector: 
                    max_err_idx = np.argmax(self._norm_errors[:-1])   
                self._obj_mat.pop(max_err_idx)
                self._residual_mat.pop(max_err_idx)                
                self._errors.pop(max_err_idx)                 
                self._norm_errors.pop(max_err_idx) 
            
            # Print:
            nvector = len(self._residual_mat)
            out = open('diis.temp', 'a')
            out.write('Cycle ' + str(cycle+1) + ':\n')
            if cycle < self.start_at + self.nvector - 1:
                out.write('  #          Error\n')
                out.write('  --------------------\n')                
                for i in range(nvector):
                    out.write(' %2d       %15.12f\n' % (i, self._norm_errors[i]))
                
        # Return new boject matrix
        if cycle >= self.start_at + self.nvector - 1:   
            new_obj_mat = self.extrapolate()
            
            # Print
            out.write('  #          Error                  c\n')
            out.write('  --------------------------------------\n')  
            for i in range(nvector):
                out.write(' %2d       %15.12f          %8.5f\n' % (i, self._norm_errors[i], self._c[i]))
        else:
            new_obj_mat = obj_mat
    
        return new_obj_mat 
        
        