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

NOTE:
This module is modified from eri_transform_gdf_rhf.py by:
    Zhihao Cui <zcui@caltech.edu>
    Tianyu Zhu <tyzhu@caltech.edu>
'''

import numpy as np
from functools import reduce
from pyscf.pbc.tools import pywannier90
from pyscf import lib, ao2mo
from pDMET.tools import tchkfile, tunix
from pDMET.pdmet import helper
from pDMET.lib.build import libdmet





def get_emb_ERIs(cell, mydf, C_ao_lo=None, basis=None, feri=None, \
        kscaled_center=None, symmetry=1, max_memory=2000, kconserv_tol=1e-12):
    '''
    Fast routine to compute embedding space ERI on the fly
    C_ao_lo: (nkpts, nao, nlo), transform matrix from AO to LO basis in k-space
    basis: (spin, ncells, nlo, nemb), embedding basis
    '''
    # gdf variables
    nao = cell.nao_nr()
    kpts = mydf.kpts
    nkpts = len(kpts)
    if mydf._cderi is None:
        if feri is not None:
            mydf._cderi = feri
    
    # If C_ao_lo and basis not given, this routine is k2gamma AO transformation
    if C_ao_lo is None:
        C_ao_lo = np.zeros((nkpts, nao, nao), dtype=np.complex128)
        C_ao_lo[:, range(nao), range(nao)] = 1.0 + 0.0j # identity matrix for each k
    C_ao_lo = C_ao_lo[np.newaxis, ...]

    # possible kpts shift
    kscaled = cell.get_scaled_kpts(kpts)
    if kscaled_center is not None:
        kscaled -= kscaled_center
    
    # basis related
    if basis is None:
        basis = np.eye(nao*nkpts).reshape(1, nkpts, nao, nao*nkpts)
    spin = basis.shape[0]
    nemb = basis.shape[-1]
    nemb_pair = nemb*(nemb+1) // 2
    scell, phase = get_phase(cell, kpts)
    basis_k = _get_basis_k(basis, phase) # FT transformed basis 
    C_ao_emb = multiply_basis(C_ao_lo, basis_k)
    
    # ERI construction
    eri = np.zeros((spin*(spin+1)//2, nemb_pair, nemb_pair), dtype=np.complex128) 
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
                        #Lij = _Lpq_to_Lij(Lpq, C_ao_lo, i, j)[0]
                        Lij_emb_pq.append(_Lij_to_Lmn(Lpq, C_ao_emb, i, j).transpose(1, 0, 2, 3))
                    # Lij_emb_pq: (spin, naux, nemb, nemb)
                    Lij_emb_pq = np.vstack(Lij_emb_pq).reshape(-1, spin, nemb, nemb)
                    Lij_emb_pq = Lij_emb_pq.transpose(1, 0, 2, 3)
                    Lij_emb += Lij_emb_pq
        Lij_s4 = _pack_tril(Lij_emb)
        _Lij_s4_to_eri(Lij_s4, eri, nkpts)
    eri_imag_norm = np.max(np.abs(eri.imag))
    assert(eri_imag_norm < 1e-10)
    eri = ao2mo.restore(symmetry, eri[0].real, nemb)[np.newaxis, ...] 
    return eri

