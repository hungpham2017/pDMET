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
       
            
def get_bath_using_RHF_1RDM(supercell_1RDM, imp_indices=None, num_bath=None, threshold=1.e-10):
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
        imp_indices = np.zeros([Nlo])
        imp_indices[:Nimp] = 1        
    else:
        imp_indices = np.asarray(imp_indices)
        env_indices = np.matrix(1 - imp_indices) 
        env_frag_mask = env_indices.T.dot(np.matrix(imp_indices) ) == 1
        Nimp = np.int32(imp_indices.sum())
        Nenv = Nlo - Nimp
        emb_1RDM = supercell_1RDM[0][env_frag_mask].reshape(Nenv, Nimp) 
    
    U, sigma, Vh = np.linalg.svd(emb_1RDM, full_matrices=True)
    distance_from_1 = np.abs(np.sqrt(np.abs(1-sigma**2)))
    idx = (distance_from_1).argsort()
    distance_from_1 = distance_from_1[idx]
    sigma = sigma[idx]
    U[:,:Nimp] = U[:,:Nimp][:,idx]
    V = Vh.T
    
    # Eliminate unentangled bath using a threshold:
    #Nbath = (np.abs(distance_from_1 - 1) > threshold).sum()
    if num_bath is not None: 
        Nbath = num_bath
    elif 2*Nimp <= Nlo: 
        Nbath = Nimp         # Avoid zero bath situation
    else:
        Nbath = Nlo - Nimp 

    # Assemble the embedding + core orbitals
    Nemb = Nimp + Nbath
    emb_orbs = np.zeros([Nlo, Nemb])
    emb_orbs[imp_indices==1,:Nimp] = V                      # impurity orbitals
    emb_orbs[imp_indices==0,Nimp:] = U[:,:Nbath]            # bath orbitals
    core_orbs = np.zeros([Nlo, Nlo - Nemb])
    core_orbs[imp_indices==0,:] = U[:,Nbath:]
    
    emb_core_orbs = np.hstack([emb_orbs, core_orbs])
    assert(np.linalg.norm(np.dot(emb_core_orbs.T, emb_core_orbs) - np.identity(Nlo)) < 1e-12 ), "WARNING: The embedding orbitals is not orthogonal"

    return emb_orbs, core_orbs, Nbath
    
    
def get_bath_using_gamma_RHF_1RDM(supercell_1RDM, imp_indices=None, threshold=1.e-10):
    '''
    Construct the RHF bath using the 1RDM for reference unit cell
    ref: 
        J. Chem. Theory Comput. 2s016, 12, 2706−2719
        
    This should work for an KROHF wfs too, resulting in an ROHF bath  with the number of bath orbitals: num_impurity + 2S          
    
    Attributes:
        supercell_1RDM_{0,L}        : the 1-RDM of the reference unit cell
        
    TODO: this was used to debug only, will be removed permanently
    '''    

    NR, Nimp, Nimp = supercell_1RDM.shape
    Nlo = NR * Nimp
    imp_indices = np.asarray(imp_indices)
    env_indices = np.matrix(1 - imp_indices) 
    env_mask = env_indices.T.dot(env_indices) == 1
    Nimp = np.int32(imp_indices.sum())
    Nenv = Nlo - Nimp
    emb_1RDM = supercell_1RDM[0][env_mask].reshape(Nenv, Nenv) 
        
    sigma, U = np.linalg.eigh(emb_1RDM)
    distance_from_1 = np.abs(sigma - 1)
    idx = (distance_from_1).argsort()
    distance_from_1 = distance_from_1[idx]
    sigma = sigma[idx]
    U = U[:,idx]
    
    # Eliminate unentangled bath using a threshold:
    Nbath = (np.abs(distance_from_1 - 1) > threshold).sum()

    # For the impurity:
    frag_frag_mask = np.matrix(imp_indices).T.dot(np.matrix(imp_indices)) == 1
    imp_1RDM = supercell_1RDM[0][frag_frag_mask].reshape(Nimp, Nimp) 
    sigma1, U1 = np.linalg.eigh(imp_1RDM)
    distance_from_1 = np.abs(sigma1 - 1)
    idx = (distance_from_1).argsort()
    distance_from_1 = distance_from_1[idx]
    sigma1 = sigma1[idx]
    U1 = U1[:,idx]
    
    # Assemble the embedding orbitals
    Nemb = Nimp + Nbath
    emb_orbs = np.zeros([Nlo, Nemb])
    emb_orbs[imp_indices==1,:Nimp] = U1           # impurity orbitals
    emb_orbs[imp_indices==0,Nimp:] = U[:,:Nbath]           # bath orbitals
    
    
    # Assemble the core orbitals
    eigvals_env = sigma[Nbath:]
    idx = (-eigvals_env).argsort()
    eigvals_env = eigvals_env[idx]
    eigvecs_env = U[:,Nbath:][:,idx]
    env_orbs = np.zeros([Nlo, Nlo - Nemb])
    env_orbs[imp_indices==0,:] = eigvecs_env
    env_occ = np.zeros(Nlo)
    env_occ[Nemb:] = eigvals_env
    
    assert(np.linalg.norm(np.dot(emb_orbs.T, emb_orbs) - np.identity(Nemb)) < 1e-12 ), "WARNING: The embedding orbitals is not orthogonal"

    return emb_orbs, env_orbs, Nbath
    