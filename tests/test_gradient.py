import numpy as np
from scipy.misc import derivative
from pyscf.pbc import gto, scf, df
import sys
sys.path.append('/panfs/roc/groups/6/gagliard/phamx494/CPPlib/pyWannier90')
import pywannier90
from pDMET.pdmet import dmet
from pDMET.lib.build import libdmet

'''
To avoid warning, run 
    pytest -p no:warnings
'''


def make_cell(d, kmesh):
    a = np.eye(3)
    a[0,0] = 10
    a[1,1] = 10
    a[2,2] = 2*d
    atoms = [['H',[5,5,0.25*a[2,2]+0.5]],['H',[5,5,0.75*a[2,2]-0.5]]]
    cell = gto.Cell()
    cell.atom = atoms
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = a
    cell.verbose = 5
    cell.build()

    kpts = cell.make_kpts(kmesh,wrap_around=True)
    krhf = scf.KRHF(cell, kpts).density_fit()
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = 'gdf.h5'
    gdf.build()
    krhf.with_df._cderi = 'gdf.h5'
    krhf.exxdiv = None
    krhf.run()

    num_wann =2
    keywords = \
    '''
    begin projections
    H:s
    end projections
    num_iter = 100 
    '''
    w90 = pywannier90.W90(krhf, cell, kmesh, num_wann, other_keywords = keywords)
    w90.kernel()
    w90.plot_wf(grid=[20,20,20], supercell=kmesh)
    return cell, krhf, w90

    
    
def multi_derivative(func, x0, dx=1e-8):
    '''Numerical derivative for a multivariable function'''
    the_deriv = []
    for var in range(len(x0)):
        args = x0
        def func_wrapper(x):
            '''Compute the func value at an arbitrary x'''
            args[var] = x
            return func(args)
            
        deriv = derivative(func_wrapper, x0[var], dx=dx)
        the_deriv.append(deriv)
    return np.asarray(the_deriv)
    

d = 2.0
kmesh = [1,1,5]
cell, kmf, w90 = make_cell(d, kmesh)   
   
def test_cost_function():
    pdmet = dmet.pDMET(cell, kmf, w90, solver='FCI')
    pdmet.initialize()
    pdmet.one_shot() 
    pdmet.SC_CFtype = 'FB'
    
    uvec0 = np.random.rand(pdmet.Nterms)
    umat0 = pdmet.uvec2umat(uvec0)
    Nkpts = pdmet.Nkpts
    OEH = pdmet.local.loc_actFOCK_kpts + umat0
    
    '''Test dmet.construct_1RDM_response_kpts function'''
    numerical_deriv = []
    for kpt in range(Nkpts):
        def RDM1_func(uvec):
            umat = pdmet.uvec2umat(uvec)
            loc_OEH_kpts = pdmet.get_loc_OEH_kpts(umat)
            return pdmet.local.make_loc_1RDM(loc_OEH_kpts)[0][kpt]
        num_deriv = multi_derivative(RDM1_func, uvec0)
        numerical_deriv.append(num_deriv)
    
    numerical_deriv  = np.asarray(numerical_deriv)
    analytical_deriv  = pdmet.construct_1RDM_response_kpts(uvec0)
    assert np.linalg.norm(numerical_deriv - analytical_deriv) < 5.e-5


    '''Test dmet.rdm_diff_grad function'''		
    loc_OEH_kpts = pdmet.get_loc_OEH_kpts(umat0)
    loc_1RDM_kpts, loc_1RDM_R0 = pdmet.local.make_loc_1RDM(loc_OEH_kpts)  
    analytical_rdm_diff_grad  = pdmet.rdm_diff_grad(uvec0)   
    def RDM1_diff(uvec):
        return pdmet.get_rdm_diff(uvec)
        
    numerical_rdm_diff_grad = multi_derivative(RDM1_diff, uvec0)
    assert np.linalg.norm(numerical_rdm_diff_grad - analytical_rdm_diff_grad) < 5.e-5

    '''Test cost_func_grad fucntion'''		
    analytical_CF_grad = pdmet.cost_func_grad(uvec0) 
    numerical_CF_grad = multi_derivative(pdmet.cost_func, uvec0)
    assert np.linalg.norm(numerical_CF_grad - analytical_CF_grad) < 5.e-5	


def test_glob_cost_function():
    pdmet = dmet.pDMET(cell, kmf, w90, solver='FCI')
    pdmet.initialize()
    pdmet.one_shot() 
    
    uvec0 = np.random.rand(pdmet.Nterms)
    umat0 = pdmet.uvec2umat(uvec0)
    Nkpts = pdmet.Nkpts
    OEH = pdmet.local.loc_actFOCK_kpts + umat0


    '''Test dmet.glob_rdm_diff_grad function'''		
    loc_OEH_kpts = pdmet.get_loc_OEH_kpts(umat0)
    loc_1RDM_kpts, loc_1RDM_R0 = pdmet.local.make_loc_1RDM(loc_OEH_kpts)  
    analytical_rdm_diff_grad  = pdmet.glob_rdm_diff_grad(uvec0)   
    def RDM1_diff(uvec):
        return pdmet.get_glob_rdm_diff(uvec)
        
    numerical_rdm_diff_grad = multi_derivative(RDM1_diff, uvec0)
    print(numerical_rdm_diff_grad)
    print("-----------------------")
    print(analytical_rdm_diff_grad)
    assert np.linalg.norm(numerical_rdm_diff_grad - analytical_rdm_diff_grad) < 5.e-5

    '''Test cost_func_grad fucntion'''		
    analytical_CF_grad = pdmet.glob_cost_func_grad(uvec0) 
    numerical_CF_grad = multi_derivative(pdmet.glob_cost_func, uvec0)
    assert np.linalg.norm(numerical_CF_grad - analytical_CF_grad) < 5.e-5	