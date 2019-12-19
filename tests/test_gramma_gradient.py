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


def make_cell():
    a = np.eye(3)
    a[0,0] = 10
    a[1,1] = 10
    a[2,2] = 15
    atoms = [['Li',[5,5,1.5]],['H',[5,5,3.5]],['H',[5,5,6.5]],['H',[5,5,8.5]],['H',[5,5,11.5]],['H',[5,5,13.5]]]
    cell = gto.Cell()
    cell.atom = atoms
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = a
    cell.verbose = 5
    cell.build()
    
    kmesh = [1, 1, 1]
    kpts = cell.make_kpts(kmesh)
    krhf = scf.KRHF(cell, kpts).density_fit()
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = 'gdf_gamma.h5'
    gdf.build()
    krhf.with_df._cderi = 'gdf_gamma.h5'
    krhf.exxdiv = None
    krhf.run()

    num_wann =7
    keywords = \
    '''
    begin projections
    Li:sp
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
    
cell, kmf, w90 = make_cell()   
   
def test_cost_function():
    pdmet = dmet.pDMET(cell, kmf, w90, solver='FCI')
    pdmet.impCluster = [1,1,0,0,0,0,0]
    pdmet.initialize()
    pdmet.one_shot() 
    # pdmet.SC_CFtype = 'FB'
    
    uvec0 = np.random.rand(pdmet.Nterms)
    umat0 = pdmet.uvec2umat(uvec0)
    OEH = pdmet.local.loc_actFOCK_kpts + umat0
    
    '''Test dmet.construct_1RDM_response_kpts function'''
    def RDM1_func(uvec):
        umat = pdmet.uvec2umat(uvec)
        loc_OEH_kpts = pdmet.get_loc_OEH_kpts(umat)
        return pdmet.local.make_loc_1RDM(loc_OEH_kpts)[0][0]
    numerical_deriv = multi_derivative(RDM1_func, uvec0)
    
    numerical_deriv  = np.asarray(numerical_deriv)
    analytical_deriv  = pdmet.construct_1RDM_response_kpts(uvec0)[0]
    np.set_printoptions(6, suppress=True)   
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
