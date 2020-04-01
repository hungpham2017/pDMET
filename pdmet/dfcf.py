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
import scipy.optimize as optimize


def get_init_uvec(xc_type='PBE0'):
    '''Initialize uvec corresponding to each type of the DF-like cost function'''
    if xc_type == 'PBE0':
        uvec = [1. , 0., 0.] 
    elif xc_type == 'B3LYP':
        uvec = [1. , 0., 0. , 0., 0.]  
    elif xc_type == 'CAMB3LYP':
        uvec = [1. , .0000001, 1. , 0., 0., 0., 0.]  
    elif xc_type == 'RSH-PBE0':
        pass
        
    return uvec

def get_bounds(xc_type='PBE0'):
    '''Prodive the bounds to optimize the DF-like cost function'''
    if xc_type == 'PBE0':
        bounds = optimize.Bounds([-1.0,-1.0,-1.0], [1.0,1.0,1.0])
    elif xc_type == 'B3LYP':
        bounds = optimize.Bounds([-1.0,.0,.0,.0,.0], [1.0,1.,1.,1.,1.])
    elif xc_type == 'CAMB3LYP':
        bounds = optimize.Bounds([0.0,.0000001,.0,.0,.0,.0,.0], [1.,1.0,1.,1.,1.,1.,1.])
    elif xc_type == 'RSH-PBE0':
        bounds = optimize.Bounds([-1.,-1.0], [1.,1.0])
    return bounds
      
    
def get_OEH_kpts(local, umat, xc_type='PBE0'):
    '''Construct the mf Hamiltonian'''
    umat = np.asarray(umat)
    if xc_type == 'PBE0':
        a, b, c = umat
        xc =  "{:.12f}*HF + {:.12f}*PBE, {:.12f}*PBE".format(a, b, c)
        n, exc, vxc = local.kks._numint.nr_rks(local.cell, local.kks.grids, xc, local.dm, 0, local.kpts, None)
        veff = vxc + local.vj - a * 0.5 * local.vk  
        OEH_kpts = local.h_core + veff
        OEH_kpts = local.ao_2_loc(OEH_kpts, local.ao2lo)
        
    elif xc_type == 'B3LYP':
        a, b, c, d, e = umat
        xc =  "{:.12f}*HF + {:.12f}*LDA + {:.12f}*B88, {:.12f}*LYP + {:.12f}*VWN".format(a, b, c, d, e)
        n, exc, vxc = local.kks._numint.nr_rks(local.cell, local.kks.grids, xc, local.dm, 0, local.kpts, None)
        veff = vxc + local.vj - a * 0.5 * local.vk  
        OEH_kpts = local.h_core + veff
        OEH_kpts = local.ao_2_loc(OEH_kpts, local.ao2lo) 

    elif xc_type == 'CAMB3LYP':
        # TODO: need to debug
        a, b, c, d, e, f, g = umat
        xc =  '{:.12f}*SR_HF({:.12f}) + {:.12f}*LR_HF({:.12f}) + {:.12f}*ITYH + {:.12f}*B88, {:.12f}*VWN5 + {:.12f}*LYP'.format(a, b, c, b, d, e, f, g)
        n, exc, vxc = local.kks._numint.nr_rks(local.cell, local.kks.grids, xc, local.dm, 0, local.kpts, None)
        veff = vxc + local.vj - a * 0.5 * local.vk  
        OEH_kpts = local.h_core + veff
        OEH_kpts = local.ao_2_loc(OEH_kpts, local.ao2lo) 
        
    elif xc_type == 'RSH-PBE0':
        a, b = umat
        xc =  "{:.12f}*SR_HF({:.12f})+ {:.12f}*PBE, {:.12f}*PBE".format(a, 0.11, 1-a, b)
        n, exc, vxc = local.kks._numint.nr_rks(local.cell, local.kks.grids, xc, local.dm, 0, local.kpts, None)
        vklr = - a * local.vklr
        veff = vxc + local.vj - 0.5 * (a * local.vk + vklr)
        OEH_kpts = local.h_core + veff
        OEH_kpts = local.ao_2_loc(OEH_kpts, local.ao2lo)
        
    elif xc_type == 'manyGGA':
        pass
        
    print("DEBUG xc", xc) 
        
    return OEH_kpts
      