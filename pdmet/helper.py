'''
Multipurpose Density Matrix Embedding theory (mp-DMET)
Hung Q. Pham
email: pqh3.14@gmail.com
'''

'''
Available tools:
    - Save/load kmf object for a pywannier90
    - Save/load kmf object for a pDMET
    - Save/load kmf object for a band structure calculation       
'''

import numpy as np
from pyscf.scf import hf
from pyscf import lib
from pyscf.lib.chkfile import save, load


def eig(h_kpts, s_kpts):
    nkpts = len(h_kpts)
    eig_kpts = []
    mo_coeff_kpts = []

    for k in range(nkpts):
        e, c = hf.eig(h_kpts[k], s_kpts[k])
        eig_kpts.append(e)
        mo_coeff_kpts.append(c)
    return eig_kpts, mo_coeff_kpts

def get_ovlp(cell, kpts):
    '''Get the overlap AO matrices at sampled k-points.

    Args:
        kpts : (nkpts, 3) ndarray

    Returns:
        ovlp_kpts : (nkpts, nao, nao) ndarray
    '''
    
# Avoid pbcopt's prescreening in the lattice sum, for better accuracy
    s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts,
                       pbcopt=lib.c_null_ptr())
    cond = np.max(lib.cond(s))
    if cond * cell.precision > 1e2:
        prec = 1e2 / cond
        rmin = max([cell.bas_rcut(ib, prec) for ib in range(cell.nbas)])
        if cell.rcut < rmin:
            logger.warn(cell, 'Singularity detected in overlap matrix.  '
                        'Integral accuracy may be not enough.\n      '
                        'You can adjust  cell.precision  or  cell.rcut  to '
                        'improve accuracy.  Recommended values are\n      '
                        'cell.precision = %.2g  or smaller.\n      '
                        'cell.rcut = %.4g  or larger.', prec, rmin)
    return lib.asarray(s)

def get_hcore(kmf, cell, kpts):
    '''Get the core Hamiltonian AO matrices at sampled k-points.

    Args:
        kpts : (nkpts, 3) ndarray

    Returns:
        hcore : (nkpts, nao, nao) ndarray
    '''
    
    if cell.pseudo:
        nuc = lib.asarray(kmf.with_df.get_pp(kpts))
    else:
        nuc = lib.asarray(kmf.with_df.get_nuc(kpts))
     
    if len(cell._ecpbas) > 0:
        nuc += lib.asarray(ecp.ecp_int(cell, kpts))
    t = lib.asarray(cell.pbc_intor('int1e_kin', 1, 1, kpts))
    return nuc + t

def get_veff(kmf, cell, dm_kpts, kpts, kpts_band=None, hermi=1):

    vj, vk = kmf.with_df.get_jk(dm_kpts, hermi, kpts, kpts_band,
                                 exxdiv=kmf.exxdiv)
    return vj - vk * .5
        
def get_bands(self, cell, kpts_band, dm_kpts=None, kpts=None):
    '''Get energy bands at the given (arbitrary) 'band' k-points.

    Returns:
        mo_energy : (nmo,) ndarray or a list of (nmo,) ndarray
            Bands energies E_n(k)
        mo_coeff : (nao, nmo) ndarray or a list of (nao,nmo) ndarray
            Band orbitals psi_n(k)
    '''
    if dm_kpts is None: dm_kpts = self.make_rdm1()
    if kpts is None: kpts = self.kpts

    kpts_band = np.asarray(kpts_band)
    single_kpt_band = (kpts_band.ndim == 1)
    kpts_band = kpts_band.reshape(-1,3)

    fock = self.get_hcore(cell, kpts_band)
    fock = fock + self.get_veff(cell, dm_kpts, kpts=kpts, kpts_band=kpts_band)
    s1e = self.get_ovlp(cell, kpts_band)
    mo_energy, mo_coeff = self.eig(fock, s1e)
    if single_kpt_band:
        mo_energy = mo_energy[0]
        mo_coeff = mo_coeff[0]
    return mo_energy, mo_coeff
            
def save_kmf(kmf, chkfile):
    e_tot = kmf.e_tot
    kpts = kmf.kpts
    mo_occ_kpts = kmf.mo_occ_kpts
    mo_energy_kpts = kmf.mo_energy_kpts
    mo_coeff_kpts = kmf.mo_coeff_kpts
    get_fock = kmf.get_fock()
    make_rdm1 = kmf.make_rdm1()    
    
    scf_dic = { 'e_tot'         : e_tot, 
                'kpts'          : kpts,
                'mo_occ_kpts'   : mo_occ_kpts,
                'mo_energy_kpts': mo_energy_kpts,
                'mo_coeff_kpts' : mo_coeff_kpts,                
                'get_fock'      : get_fock,
                'make_rdm1'     : make_rdm1}
                
    save(chkfile, 'scf', scf_dic)
	
def load_kmf(kmf, chkfile):
    '''
        One need a initialized kmf object, does not have to rerun the SCF
    '''
    
    save_kmf = load(chkfile, 'scf')
    class fake_kmf:
        def __init__(self, save_kmf):
            self.e_tot = save_kmf['e_tot']
            self.kpts  = save_kmf['kpts']
            self.mo_occ_kpts = save_kmf['mo_occ_kpts']
            self.mo_energy_kpts = save_kmf['mo_energy_kpts']
            self.mo_coeff_kpts = save_kmf['mo_coeff_kpts']
            self.get_fock  = lambda *arg: save_kmf['get_fock']
            self.make_rdm1  = lambda *arg: save_kmf['make_rdm1']            
            self.eig       = lambda h_kpts, s_kpts: eig(h_kpts, s_kpts)
            self.get_ovlp  = lambda cell, kpts: get_ovlp(cell, kpts)
            self.get_hcore = lambda cell, kpts: get_hcore(kmf, cell, kpts)
            self.get_veff  = lambda cell, dm_kpts, kpts, kpts_band: get_veff(kmf, cell, dm_kpts, kpts, kpts_band, hermi=1)
            self.get_bands  = lambda cell, kpts_band, dm_kpts, kpts: get_bands(self, cell, kpts_band, dm_kpts, kpts)

    final_kmf = fake_kmf(save_kmf)
    return final_kmf
    
def save_w90(w90, chkfile):
    mp_grid_loc        = w90.mp_grid_loc
    exclude_bands      = w90.exclude_bands
    mp_grid_loc        = w90.mp_grid_loc
    mo_coeff_kpts      = w90.mo_coeff_kpts
    band_included_list = w90.band_included_list
    lwindow            = w90.lwindow
    U_matrix_opt       = w90.U_matrix_opt
    U_matrix           = w90.U_matrix
    w90_dic = { 'mp_grid_loc'           : mp_grid_loc, 
                'exclude_bands'         : exclude_bands,
                'mp_grid_loc'           : mp_grid_loc,
                'mo_coeff_kpts'         : mo_coeff_kpts,
                'band_included_list'    : band_included_list,
                'lwindow'               : lwindow,
                'U_matrix_opt'          : U_matrix_opt,
                'U_matrix'              : U_matrix}
                
    save(chkfile, 'w90', w90_dic)
    
def load_w90(chkfile):
    save_w90 = load(chkfile, 'w90')
    class fake_w90:
        def __init__(self, save_w90):
            self.mp_grid_loc = save_w90['mp_grid_loc']
            self.exclude_bands  = save_w90['exclude_bands']
            self.mp_grid_loc  = save_w90['mp_grid_loc']
            self.mo_coeff_kpts  = save_w90['mo_coeff_kpts']
            self.band_included_list  = save_w90['band_included_list']
            self.lwindow  = save_w90['lwindow']
            self.U_matrix_opt  = save_w90['U_matrix_opt']
            self.U_matrix  = save_w90['U_matrix']   
            
    w90 = fake_w90(save_w90)
    return w90    

def get_kpts(cell, path_info, density=10):
    '''
        Giving a path (in text), this function produce the absolute kpoint list for band structure calculations
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
