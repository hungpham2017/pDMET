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


def get_occ_r(nelec, mo_energy_kpts):
    ''' Get occupation numbers at k-point for a KRHF wf, modified from pbc/scf/krhf.py'''
    
    nkpts = len(mo_energy_kpts)
    nocc = nelec // 2
    mo_energy = np.sort(np.hstack(mo_energy_kpts))
    fermi = mo_energy[nocc-1]
    mo_occ_kpts = []
    for mo_e in mo_energy_kpts:
        mo_occ_kpts.append((mo_e <= fermi).astype(np.double) * 2)
    return np.asarray(mo_occ_kpts)

def get_occ_ro(nelec, mo_energy_kpts):
    ''' Get occupation numbers at k-point for a KROHF wf, modified from pbc/scf/krohf.py'''

    if hasattr(mo_energy_kpts[0], 'mo_ea'):
        mo_ea_kpts = [x.mo_ea for x in mo_energy_kpts]
        mo_eb_kpts = [x.mo_eb for x in mo_energy_kpts]
    else:
        mo_ea_kpts = mo_eb_kpts = mo_energy_kpts

    nkpts = len(mo_energy_kpts)
    nocc_a = nelec[0] * nkpts
    nocc_b = nelec[1] * nkpts

    mo_energy_kpts1 = np.hstack(mo_energy_kpts)
    mo_energy = np.sort(mo_energy_kpts1)
    if nocc_b > 0:
        core_level = mo_energy[nocc_b-1]
    else:
        core_level = -1e9
    if nocc_a == nocc_b:
        fermi = core_level
    else:
        mo_ea_kpts1 = np.hstack(mo_ea_kpts)
        mo_ea = np.sort(mo_ea_kpts1[mo_energy_kpts1 > core_level])
        fermi = mo_ea[nocc_a - nocc_b - 1]

    mo_occ_kpts = []
    for k, mo_e in enumerate(mo_energy_kpts):
        occ = np.zeros_like(mo_e)
        occ[mo_e <= core_level] = 2
        if nocc_a != nocc_b:
            occ[(mo_e > core_level) & (mo_ea_kpts[k] <= fermi)] = 1
        mo_occ_kpts.append(occ)
        
    return mo_occ_kpts
    
def irred_kmesh(kpts):
    '''
    Giving any kpts, return irreducible kpts
    Attributes:
        kpts_irred      : a list of irreducible k-point   
        sym_id          : a list of symmetry label. k and -k should have the same label
        sym_map         : used to map the uvec (irreducible k-points) to umat (full k-points)        
    '''        
    nkpts = kpts.shape[0]
    sym_id = np.asarray(range(nkpts))   
    for i in range(kpts.shape[0]-1):
        for j in range(i+1): 
            if abs(kpts[i+1] + kpts[j]).sum() < 1.e-10: 
                sym_id[i+1] = sym_id[j]   
                break 
    kpts_irred, sym_counts = np.unique(sym_id, return_counts=True)                        
    nkpts_irred = kpts_irred.size 
    sym_map = [np.where(kpts_irred == sym_id[kpt])[0][0] for kpt in range(nkpts)]        
    
    return kpts_irred, sym_counts, sym_map
    
def KRHF(cell, OEI, TEI, nelectron, kpts, DMguess, verbose=0, max_cycle=10):
    '''KRHF wrapper to solve for 1RDM with a certain umat'''

    from pyscf.pbc import gto, scf   
    import warnings
    nkpts = kpts.shape[0]
    def get_veff(cell=None, dm_kpts=None, *args):
        '''Function to compute veff from ERI'''
        delta = np.eye(nkpts) 
        weight = 1/nkpts
        vj = weight * np.einsum('ijkpqrs,ksr,ij->ipq', TEI,dm_kpts,delta,optimize = True)
        vk = weight * np.einsum('ijkpqrs,jqr,jk->ips', TEI,dm_kpts,delta,optimize = True)       
        veff = vj - 0.5*vk
        return veff

    nao = OEI.shape[1]
    cell.atom = [['He', (0.5, 0.5, 0.5)]]
    cell.incore_anyway = True
    cell.nelectron = nelectron
    kmf = scf.KRHF(cell, kpts, exxdiv=None)
    kmf.get_hcore = lambda *args: OEI
    kmf.get_ovlp = lambda *args: np.array([np.eye(nao)]*nkpts)
    kmf.get_veff = get_veff
    kmf.max_cycle = max_cycle
    kmf.run(DMguess)   
    dms = kmf.make_rdm1()
    
    return dms
    
def KRKS(cell, XC, OEI, TEI, nelectron, kpts, DMguess, verbose=0, max_cycle=1):
    '''KRKS wrapper to solve for 1RDM with a certain umat'''

    from pyscf.pbc import gto, scf   
    import warnings
    nkpts = kpts.shape[0]
    def get_veff(dm_kpts=None, *args):
        '''Function to compute veff from ERI'''
        delta = np.eye(nkpts) 
        weight = 1/nkpts
        vj = weight * np.einsum('ijkpqrs,ksr,ij->ipq', TEI,dm_kpts,delta,optimize = True)
        vk = weight * np.einsum('ijkpqrs,jqr,jk->ips', TEI,dm_kpts,delta,optimize = True)       
        veff = vj - 0.5*vk
        return veff

    nao = OEI.shape[1]
    cell.atom = [['He', (0.5, 0.5, 0.5)]]
    cell.incore_anyway = True
    cell.nelectron = nelectron
    kmf = scf.KRKS(cell, kpts, exxdiv=None)
    kmf.xc = XC
    kmf.get_hcore = lambda *args: OEI
    kmf.get_ovlp = lambda *args: np.array([np.eye(nao)]*nkpts)
    kmf.get_veff = get_veff
    kmf.max_cycle = max_cycle
    kmf.run(DMguess)   
    dms = kmf.make_rdm1()
    
    return kmf.converged, dms
