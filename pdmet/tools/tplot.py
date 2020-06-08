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
import os, datetime
from pyscf import lib
import pyscf.data.nist as param
from pyscf.lib.chkfile import save, load
             

def get_kpts(cell, path_info, density=10):
    '''Giving a path (in text), this function produce the absolute kpoint list for band structure calculations
    '''
    
    points = path_info.split()
    num_klist = (len(points)-1)//2
    path = points[-1].split('-')
    num_path = len(path) -1 
    klist = [points[2*kpt] for kpt in range(num_klist)]
    kcoor = np.asarray([[np.float64(points[2*kpt+1].split(',')[i]) for i in range(3)] for kpt in range(num_klist)])
    num_kpts = (density-1)*num_path + 1    
    path_coor = np.asarray([kcoor[klist.index(kpt)] for kpt in path])

    scaled_kpts = np.empty([num_kpts,3]) 
    for line in range(num_path):
        delta = path_coor[line+1] - path_coor[line]
        if line ==0:
            scaled_kpts[line*density:(line+1)*density,:] = \
                np.asarray([i*delta/(density-1) for i in np.arange(density)]) + np.asarray(path_coor[line])
        else:
            scaled_kpts[line*density-line:(line+1)*density-line,:] = \
                np.asarray([i*delta/(density-1) for i in np.arange(density)]) + np.asarray(path_coor[line])

            
    kpts = cell.get_abs_kpts(scaled_kpts) 
    kpts_copy = kpts.copy()
    kpts_copy[1:,:] = kpts[:-1,:]
    x = np.cumsum(np.sqrt(((kpts-kpts_copy)**2).sum(axis=1)))     
    
    return kpts, x
    
def save_bands(bands, x, chkfile):
    '''
        Save bandstructure for plotting
    '''
    band_structure = np.hstack((x.reshape(-1,1),bands))
    bands_dic = { 'bands'           : band_structure} 
    save(chkfile, 'bands', bands_dic)
    
def load_bands(chkfile):
    '''
        load bandstructure for plotting
    '''
    save_bands = load(chkfile, 'bands')
    band_structure = save_bands['bands']
    return band_structure     
    
def orb_analysis(cell, mo_coeff, mo = None):
    '''
        Oribital decomposition
    '''
    mo_coeff = np.asarray(mo_coeff)
    assert (cell.nao_nr() == mo_coeff.shape[1])
    num_kpts = mo_coeff.shape[0]
    percentile_kpts = []
    for kpt in range(num_kpts):
        c2 = (mo_coeff[kpt].conj()*mo_coeff[kpt]).real
        sum = c2.sum(axis=0)
        percentile_kpts.append(c2/sum)
    percentile_kpts = np.asarray(percentile_kpts)
    percentile = 1/num_kpts*percentile_kpts.sum(axis=0)
    print()
    for ao in range(cell.nao_nr()):
        print(('%15s' + '%3.2f  '*cell.nao_nr()) % (tuple([cell.ao_labels()[ao]]) + tuple(percentile[ao,:])))
    
def to_cartesian(lattice_mat, frac_coors):
    '''
        Convert fractional coordination to cartesian
    '''        
    coors = frac_coors.split() 
    natom = len(coors) // 4
    lattice = np.sqrt((np.sum(lattice_mat**2, axis = 1)))
    atoms_cartesian = []
    for atom in range(natom):
        symbol, = coors[4*atom]
        x = np.float64(coors[4*atom + 1])
        y = np.float64(coors[4*atom + 2])
        z = np.float64(coors[4*atom + 3])
        x, y, z = np.asarray([x,y,z]) * lattice
        atoms_cartesian.append([symbol, [x,y,z]])
    return atoms_cartesian   

def make_xsf(cell, struc = 'structure'):  
    '''
        Export xsf structure from cell object
    '''    
    import datetime
    import pyscf.data.nist as param
    lattice = cell.lattice_vectors() * param.BOHR
    atom_symbols = [atom[0] for atom in cell._atom]
    atoms_cart = np.asarray([(np.asarray(atom[1])* param.BOHR).tolist() for atom in cell._atom])

    with open(struc + '.xsf', 'w') as f:
        f.write(str(datetime.datetime.now()))
        f.write('\n\n')        
        f.write('CRYSTAL\n')
        f.write('PRIMVEC\n')    
        for row in range(3):
            f.write('%10.7f  %10.7f  %10.7f\n' % (lattice[row,0], lattice[row,1], lattice[row,2]))    
        f.write('CONVVEC\n')
        for row in range(3):
            f.write('%10.7f  %10.7f  %10.7f\n' % (lattice[row,0], lattice[row,1], lattice[row,2]))    
        f.write('PRIMCOORD\n')
        f.write('%3d %3d\n' % (cell.natm, 1))
        for atom in range(len(atom_symbols)):
            f.write('%s  %7.7f  %7.7f  %7.7f\n' % (atom_symbols[atom], atoms_cart[atom][0], \
            atoms_cart[atom][1], atoms_cart[atom][2]))                
        f.write('\n\n')    
    
def get_wannier(w90, supercell = [1,1,1], grid = [50,50,50]):
    '''
    Evaluate the MLWF using a periodic grid
    '''	
    import sys    
    sys.path.append('/home/gagliard/phamx494/CPPlib/pyWannier90')
    import libwannier90
    sys.path.append('/panfs/roc/groups/6/gagliard/phamx494/CPPlib/pyWannier90')
    import pywannier90
    from pyscf.pbc.dft import gen_grid, numint    
    
    grids_coor, weights = pywannier90.periodic_grid(w90.cell, grid, supercell = [1,1,1], order = 'C')	
    kpts = w90.cell.get_abs_kpts(w90.kpt_latt_loc)          
    ao_kpts = np.asarray([numint.eval_ao(w90.cell, grids_coor, kpt = kpt) for kpt in kpts]) 
    
    u_mo  = []            
    for k_id in range(w90.num_kpts_loc):
        mo_included = w90.mo_coeff_kpts[k_id][:,w90.band_included_list]
        mo_in_window = w90.lwindow[k_id]
        C_opt = mo_included[:,mo_in_window].dot(w90.U_matrix_opt[k_id].T)          
        C_tildle = C_opt.dot(w90.U_matrix[k_id].T)  
        kpt = kpts[k_id]
        ao = numint.eval_ao(w90.cell, grids_coor, kpt = kpt)            
        u_ao = np.einsum('x,xi->xi', np.exp(-1j*np.dot(grids_coor, kpt)), ao, optimize = True)   
        u_mo.append(np.einsum('xi,in->xn', u_ao, C_tildle, optimize = True))      
    
    u_mo = np.asarray(u_mo)
    
    nimgs = [kpt//2 for kpt in w90.mp_grid_loc]	
    Ts = lib.cartesian_prod((np.arange(-nimgs[0],nimgs[0]+1),
                             np.arange(-nimgs[1],nimgs[1]+1),
                             np.arange(-nimgs[2],nimgs[2]+1))) 
                             
    Ts = np.asarray(Ts, order='C')      #lib.cartesian_prod store array in Fortran order in memory                           
    WFs = libwannier90.get_WFs(w90.kpt_latt_loc.shape[0],w90.kpt_latt_loc, Ts.shape[0], Ts, supercell, grid, u_mo)    
       
    return WFs
        
def plot_wf(w90, rotate_mat = None, outfile = 'MLWF', supercell = [1,1,1], grid = [50,50,50]):
    '''
    Export Wannier function at cell R
    xsf format: http://web.mit.edu/xcrysden_v1.5.60/www/XCRYSDEN/doc/XSF.html
    Attributes:
        wf_list		: a list of MLWFs to plot
        supercell	: a supercell used for plotting
    This function is modified from pywannier90 to rotate WFs before plotting        
    '''	
  
    
    grid = np.asarray(grid)
    origin = np.asarray([-(grid[i]*(supercell[i]//2) + 1)/grid[i] for i in range(3)]).dot(w90.cell.lattice_vectors().T)* param.BOHR            
    real_lattice_loc = (grid*supercell-1)/grid * w90.cell.lattice_vectors() * param.BOHR	
    nx, ny, nz = grid*supercell    
    WFs = get_wannier(w90, supercell, grid)
    if rotate_mat is not None: WFs = WFs.dot(rotate_mat)
    num_wfs = WFs.shape[-1]
    
    for wf_id in list(range(num_wfs)):
        WF = WFs[:,wf_id].reshape(nx,ny,nz).real

                            
        with open(outfile + '-' + str(wf_id) + '.xsf', 'w') as f:
            f.write('Generated by the pyWannier90\n\n')		
            f.write('CRYSTAL\n')
            f.write('PRIMVEC\n')	
            for row in range(3):
                f.write('%10.7f  %10.7f  %10.7f\n' % (w90.real_lattice_loc[row,0], w90.real_lattice_loc[row,1], \
                w90.real_lattice_loc[row,2]))	
            f.write('CONVVEC\n')
            for row in range(3):
                f.write('%10.7f  %10.7f  %10.7f\n' % (w90.real_lattice_loc[row,0], w90.real_lattice_loc[row,1], \
                w90.real_lattice_loc[row,2]))	
            f.write('PRIMCOORD\n')
            f.write('%3d %3d\n' % (w90.num_atoms_loc, 1))
            for atom in range(len(w90.atom_symbols_loc)):
                f.write('%s  %7.7f  %7.7f  %7.7f\n' % (w90.atom_symbols_loc[atom], w90.atoms_cart_loc[atom][0], \
                 w90.atoms_cart_loc[atom][1], w90.atoms_cart_loc[atom][2]))				
            f.write('\n\n')			
            f.write('BEGIN_BLOCK_DATAGRID_3D\n3D_field\nBEGIN_DATAGRID_3D_UNKNOWN\n')	
            f.write('   %5d	 %5d  %5d\n' % (nx, ny, nz))		
            f.write('   %10.7f  %10.7f  %10.7f\n' % (origin[0],origin[1],origin[2]))
            for row in range(3):
                f.write('   %10.7f  %10.7f  %10.7f\n' % (real_lattice_loc[row,0], real_lattice_loc[row,1], \
                real_lattice_loc[row,2]))	
                
            fmt = ' %13.5e' * nx + '\n'
            for iz in range(nz):
                for iy in range(ny):
                    f.write(fmt % tuple(WF[:,iy,iz].tolist()))										
            f.write('END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D')	
            
