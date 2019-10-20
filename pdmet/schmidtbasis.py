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
from pDMET.lib.build import libdmet
    
class HF_decomposition:
    def __init__(self, cell, impOrbs, numBathOrbs, locOED_Ls):    
        self.cell = cell
        self.impOrbs = impOrbs
        self.numBathOrbs = numBathOrbs        
        self.locOED_Ls = locOED_Ls
        
    def baths(self, threshold):
        '''
        This function is used to call the Schmidt basis using 1-rdm
        '''                  
        return self.UsingOED(self.numBathOrbs, threshold = threshold)
            
    def UsingOED(self, numBathOrbs, threshold = 1e-8):
        '''
        Construct the RHF bath using one-electron density matrix (OED)
        This function is a modified version of qcdmethelper/constructbath funtion 
        in the QC-DMET <Copyright (C) 2015 Sebastian Wouters>
        ref: 
            J. Chem. Theory Comput. 2016, 12, 2706−2719
            
        This should work for an KROHF wfs too, resulting in an ROHF bath  with the number of bath orbitals: num_impurity + 2S          
        '''    
        
        OneDM = self.locOED_Ls    
        impurityOrbs = np.asarray(self.impOrbs)
        embeddingOrbs = 1 - impurityOrbs
        if (embeddingOrbs.shape[0] > 1):
            embeddingOrbs = embeddingOrbs.T        
        embeddingOrbs = np.matrix(embeddingOrbs)                 #Converse embeddingOrbs to a matrix (1, x)
        isEmbedding = np.dot(embeddingOrbs.T , embeddingOrbs) == 1
        numEmbedOrbs = np.sum(embeddingOrbs, dtype = np.int64)
        embedding1RDM = np.reshape(OneDM[isEmbedding], (numEmbedOrbs, numEmbedOrbs))   
        numImpOrbs   = np.sum(impurityOrbs, dtype = np.int64)
        numTotalOrbs = len(impurityOrbs)
        eigenvals, eigenvecs = np.linalg.eigh(embedding1RDM, UPLO='U')      # 0 <= eigenvals <= 2        
        idx = np.maximum(-eigenvals, eigenvals - 2.0).argsort() # Occupation numbers closest to 1 come first

        if threshold == False:
            tokeep = numBathOrbs
        else:
            tokeep = np.sum(-np.maximum(-eigenvals, eigenvals - 2.0)[idx] > threshold)
            if tokeep < numBathOrbs:
                print ("   Bath construction: Throw out", numBathOrbs - tokeep, "orbitals using the threshold", threshold)
                
        numBathOrbs = min(np.sum(tokeep), numBathOrbs)
        
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:,idx]
        pureEnvals = eigenvals[numBathOrbs:]
        pureEnorbs = eigenvecs[:,numBathOrbs:]
        idx = (-pureEnvals).argsort()
        eigenvecs[:,numBathOrbs:] = pureEnorbs[:,idx]
        pureEnvals = pureEnvals[idx]
        coreOccupations = np.hstack((np.zeros([numImpOrbs + numBathOrbs]), pureEnvals)) #Use to calculate the 1e rdm of core orbitals

        # Reconstruct the fragment orbitals so that the density matrix has a trivial form:
        embeddingOrbs_frag = np.matrix(impurityOrbs)                 #Converse embeddingOrbs to a matrix (1, x)
        isEmbedding_frag = np.dot(embeddingOrbs_frag.T , embeddingOrbs_frag) == 1
        numEmbedOrbs_frag = np.sum(embeddingOrbs_frag, dtype = np.int32)
        embedding1RDM_frag = np.reshape(OneDM[isEmbedding_frag], (numEmbedOrbs_frag, numEmbedOrbs_frag))
        eigenvals_frag, eigenvecs_frag = np.linalg.eigh(embedding1RDM_frag)      # 0 <= eigenvals <= 2

        #Debug: rotate the fragment orbitals among themselves
        if False: eigenvecs_frag = np.eye(eigenvecs_frag.shape[0],eigenvecs_frag.shape[0])
        
        #Fragment orbitals: stack columns with zeros in the end
        #Embedding orbitals: stack columns with zeros in the beginning    
        eigenvecs_frag = np.hstack((eigenvecs_frag, np.zeros((numImpOrbs, numEmbedOrbs)))) 
        eigenvecs = np.hstack((np.zeros((numEmbedOrbs, numImpOrbs)), eigenvecs))
        row = 0
        for ao in range(0, numTotalOrbs):
            if impurityOrbs[ao]:
                eigenvecs = np.insert(eigenvecs, ao, eigenvecs_frag[row], axis=0)
                row += 1
    
        # Orthonormality is guaranteed due to (1) stacking with zeros and (2) orthonormality eigenvecs for symmetric matrix
        assert(np.linalg.norm(np.dot(eigenvecs.T, eigenvecs) - np.identity(numTotalOrbs)) < 1e-12 )
    
        return (numBathOrbs, eigenvecs, coreOccupations)

        
