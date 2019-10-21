#!/usr/bin/env python -u 
'''
This Gaussian density fitting is provided by:
    Zhihao Cui <zcui@caltech.edu>
    Tianyu Zhu <tyzhu@caltech.edu>
    ref: https://arxiv.org/abs/1909.08596 and https://arxiv.org/abs/1909.08592
'''

import numpy as np
import scipy.linalg as la
from pyscf import lib, ao2mo
einsum = lib.einsum

def _pack_tril(Lij):
    if len(Lij.shape) == 3:
        Lij_pack = lib.pack_tril(Lij)
    else:
        spin, nL, _, nao = Lij.shape
        nao_pair = nao*(nao+1) // 2
        Lij_pack = np.empty((spin, nL, nao_pair), dtype=Lij.dtype)
        for s in range(spin):
            Lij_pack[s] = lib.pack_tril(Lij[s])
    return Lij_pack
    
def _Lij_to_Lmn(Lij, basis, ki, kj):
    if len(basis.shape) == 3:
        basis = basis[np.newaxis, ...]
    if len(Lij.shape) == 3:
        Lij = Lij[np.newaxis, ...]
    spin, ncells, nlo, nemb = basis.shape
    nL = Lij.shape[-3]
    
    Lmn = np.zeros((spin, nL, nemb, nemb), dtype=np.complex128)
    Li_n = np.empty((nL*nlo, nemb), dtype=np.complex128)
    Ln_m = np.empty((nL*nemb, nemb), dtype=np.complex128) 
    for s in range(spin):
        lib.dot(Lij[s].reshape((nL*nlo, nlo)), basis[s, kj], c=Li_n) 
        lib.dot(np.ascontiguousarray(np.swapaxes(Li_n.reshape((nL, nlo, nemb)), 1, 2).reshape((nL*nemb, nlo))), \
                basis[s, ki].conj(), c=Ln_m) 
        Lmn[s] = np.swapaxes(Ln_m.reshape((nL, nemb, nemb)), 1, 2) 
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
        lib.dot(Lij_s4[1].conj().T, Lij_s4[1], 1, eri[2], 1) 
    return
        
def get_emb_eri_fast(cell, mydf, ao2eo, feri=None, \
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
    
    ao2eo = ao2eo[np.newaxis, ...]

    # possible kpts shift
    kscaled = cell.get_scaled_kpts(kpts)
    if kscaled_center is not None:
        kscaled -= kscaled_center
    
    spin = ao2eo.shape[0]
    nemb = ao2eo.shape[-1]
    nemb_pair = nemb*(nemb+1) // 2
    
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
                        Lij_emb_pq.append(_Lij_to_Lmn(Lpq, ao2eo, i, j).transpose(1, 0, 2, 3))
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

