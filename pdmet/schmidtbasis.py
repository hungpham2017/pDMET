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
    
        
            
def get_bath_using_RHF_1RDM(supercell_1RDM, imp_indices=None):
    '''
    Construct the RHF bath using the 1RDM for reference unit cell
    ref: 
        J. Chem. Theory Comput. 2s016, 12, 2706−2719
        
    This should work for an KROHF wfs too, resulting in an ROHF bath  with the number of bath orbitals: num_impurity + 2S          
    
    Attributes:
        supercell_1RDM_{0,L}        : the 1-RDM of the reference unit cell
    '''    
    NR, Nimp, Nimp = supercell_1RDM.shape
    Nlo = NR * Nimp
    if imp_indices is None:
        supercell_1RDM = supercell_1RDM.reshape(Nlo, Nimp)
        emb_1RDM = supercell_1RDM[Nimp:,:]      
    else:
        imp_indices = np.asarray(imp_indices)
        env_indices = np.matrix(1 - imp_indices)   
        frag_env_mask = imp_indices.T.dot(env_indices) == 1
        Nimp = np.int32(imp_indices.sum())
        Nenv = Nlo - Nimp
        emb_1RDM = supercell_1RDM[0][frag_env_block].reshape(Nimp, Nenv) 
    
    U, sigma, Vh = np.linalg.svd(emb_1RDM, full_matrices=False)
    distance_from_1 = np.abs(np.sqrt(np.abs(1-sigma**2)))
    idx = (distance_from_1).argsort()
    sigma = sigma[idx]
    V = Vh.T[:,idx]
    
    # Assemble the embedding orbitals
    emb_orbs = np.zeros([Nlo,2*Nimp])
    emb_orbs[:Nimp,:Nimp] = V           # impurity orbitals
    emb_orbs[Nimp:,Nimp:] = U           # bath orbitals
    
    assert(np.linalg.norm(np.dot(emb_orbs.T, emb_orbs) - np.identity(2*Nimp)) < 1e-12 ), "WARNING: The embedding orbitals is not orthogonal"

    return emb_orbs
