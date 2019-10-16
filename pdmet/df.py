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

The Gaussian density fitting is modified based on the eri_transform_gdf_rhf.py module provided by:
    Zhihao Cui <zcui@caltech.edu>
    Tianyu Zhu <tyzhu@caltech.edu>
    ref: https://arxiv.org/abs/1909.08596 and https://arxiv.org/abs/1909.08592
'''

import numpy as np
import scipy.linalg as la
from pyscf import lib, ao2mo
einsum = lib.einsum
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper



def _pack_tril(Lij):
    Lij_pack = lib.pack_tril(Lij)
    return Lij_pack
    
def _Lij_to_Lmn(Lij, ao2eo, ki, kj):
    ncells, nlo, nemb = ao2eo.shape
    nL = Lij.shape[-3]
    
    Lmn = np.zeros((nL, nemb, nemb), dtype=np.complex128)
    Li_n = np.empty((nL*nlo, nemb), dtype=np.complex128)
    Ln_m = np.empty((nL*nemb, nemb), dtype=np.complex128) 

    lib.dot(Lij.reshape((nL*nlo, nlo)), ao2eo[kj], c=Li_n) 
    lib.dot(np.ascontiguousarray(np.swapaxes(Li_n.reshape((nL, nlo, nemb)), 1, 2).reshape((nL*nemb, nlo))), \
            ao2eo[ki].conj(), c=Ln_m) 
    Lmn = np.swapaxes(Ln_m.reshape((nL, nemb, nemb)), 1, 2) 
    return Lmn

def _Lij_s4_to_eri(Lij_s4, eri, nkpts, extra_Lij_factor=1.0):
    if len(Lij_s4.shape) == 2:
        Lij_s4 = Lij_s4[np.newaxis, ...]
    spin, nL, nemb_pair = Lij_s4.shape
    Lij_s4 /= (np.sqrt(nkpts)*extra_Lij_factor)
    if spin == 1:
        lib.dot(Lij_s4[0].conj().T, Lij_s4[0], 1, eri[0], 1) 
    else:
        lib.dot(Lij_s4[0].conj().T, Lij_s4[0], 1, eri[0], 1) 
        lib.dot(Lij_s4[0].conj().T, Lij_s4[1], 1, eri[1], 1) 
        lib.d
        
def get_emb_ERIs(cell, mydf, ao2eo=None, feri=None, symmetry=1, max_memory=2000, kconserv_tol=1e-12):
    '''
    Fast routine to compute embedding space ERI on the fly
    ao2lo: (nkpts, nao, nlo), transform matrix from AO to LO basis in k-space
    '''
    # gdf variables
    nao = cell.nao_nr()
    kpts = mydf.kpts
    nkpts = len(kpts)
    if mydf._cderi is None:
        if feri is not None:
            mydf._cderi = feri

    kscaled = cell.get_scaled_kpts(kpts)
    nemb = ao2eo[0].shape[1]
    nemb_pair = nemb*(nemb+1) // 2
    
    # ERI construction
    eri = np.zeros((nemb_pair, nemb_pair), dtype=np.complex128) 
    for kL in range(nkpts):
        Lij_emb = 0.0
        for i, kpti in enumerate(kpts):
            for j, kptj in enumerate(kpts):
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                is_kconserv = la.norm(np.round(kconserv) - kconserv) < kconserv_tol 
                if is_kconserv:
                    Lij_emb_pq = []
                    # Loop over L chunks
                    for LpqR, LpqI, sign in mydf.sr_loop([kpti, kptj], max_memory=max_memory, compact=False):
                        Lpq = (LpqR + LpqI*1.0j).reshape(-1, nao, nao)
                        Lij_emb_pq.append(_Lij_to_Lmn(Lpq, ao2eo, i, j))
                    Lij_emb_pq = np.vstack(Lij_emb_pq).reshape(-1, nemb, nemb)
                    #Lij_emb_pq = Lij_emb_pq.transpose(1, 0, 2, 3)
                    Lij_emb += Lij_emb_pq
        #Lij_s4 = _pack_tril(Lij_emb)
        #_Lij_s4_to_eri(Lij_s4, eri, nkpts)
        Lij_s4 = lib.pack_tril(Lij_emb)
        lib.dot(Lij_s4.conj().T, Lij_s4, 1, eri, 1) 
    eri_imag_norm = np.max(np.abs(eri.imag))
    assert(eri_imag_norm < 1e-10)
    eri = ao2mo.restore(symmetry, eri.real, nemb)
    return eri

