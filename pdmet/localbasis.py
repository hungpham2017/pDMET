'''
Multipurpose Density Matrix Embedding theory (mp-DMET)
Hung Q. Pham
email: pqh3.14@gmail.com
'''

import numpy as np
import scipy as scipy
from functools import reduce
from pyscf.pbc.tools import pywannier90
from pyscf import lib, ao2mo
from pdmet.lib.build import libdmet

class main:
	def __init__(self, cell, kmf, w90, ERIinMEM = True):
		'''
		Prepare the Wannier functions, transform OEI and TEI to the real-space representation
		Args:
			kmf		: a k-dependent mean-field wf
			w90		: a converged wannier90 object
		'''		
		
		self.ERIinMEM  = ERIinMEM
		# Collect cell and kmf object information
		self.e_tot = kmf.e_tot
		self.w90 = w90
		self.kmf = kmf		
		self.kpts = kmf.kpts
		self.nkpts = kmf.kpts.shape[0]	
		self.nao = cell.nao_nr()
		
		# The k-point number has to be odd, since the fragment is assumed to be in the middle of the supercell
		assert ([kpt%2 == 0 for kpt in w90.mp_grid_loc] == [False]*3)
		nimgs = [kpt//2 for kpt in w90.mp_grid_loc]	
		self.Ls = cell.get_lattice_Ls(nimgs)
		self.nLs = self.Ls.shape[0]
		assert self.nLs == self.nkpts
		
		
		# Active part info
		self.active = np.zeros([cell.nao_nr()], dtype=int)
		for orb in range(cell.nao_nr()):
			if (orb+1) not in w90.exclude_bands: self.active[orb] = 1
		self.nActorbs = np.sum(self.active)	
		self.norbs = self.nkpts * self.nActorbs
		self.nActelecs = np.int32(cell.nelectron - np.sum(kmf.mo_occ_kpts[0][self.active==0]))		
		self.nelec = self.nkpts * self.nActelecs
			
		self.CO, WFs = self.make_WFs(self.w90)	# WFs basis in k- and L- space
		
		#-------------------------------------------------------------
		# Construct the effective Hamiltonian due to the frozen core  | 
		#-------------------------------------------------------------
		fullH1eao_kpts = kmf.get_hcore(cell, self.kpts)
		mo_kpts = kmf.mo_coeff_kpts
		self.frozenDMao_kpts = []
		for kpt in range(self.nkpts):
			frozenDMmo  = kmf.mo_occ_kpts[kpt].copy()
			frozenDMmo[self.active==1] = 0
			frozenDMao = reduce(np.dot, (mo_kpts[kpt], np.diag(frozenDMmo), mo_kpts[kpt].T.conj()))
			self.frozenDMao_kpts.append(frozenDMao)
		
		self.frozenDMao_kpts = np.asarray(self.frozenDMao_kpts, dtype=np.complex128)
		self.frozenJKao_kpts = kmf.get_veff(cell, self.frozenDMao_kpts, self.kpts, None)
		
		# Core energy from the frozen orbitals
		self.coreEnergy = cell.energy_nuc() + 1./self.nkpts *np.einsum( 'kij,kij->', fullH1eao_kpts + 0.5*self.frozenJKao_kpts, self.frozenDMao_kpts )
   
		supcell = self.make_supcell(cell, w90.mp_grid_loc)	# This supercell is used to generate 1eint and 2eint in L-space
		# One-electron integral for the active part
		ActOEI_kpts = fullH1eao_kpts + self.frozenJKao_kpts 
		self.loc_ActOEI_kpts = self.to_localbas(ActOEI_kpts, self.CO)
		frozenJKao_Ls = libdmet.iFTWFs(self.nkpts, self.kpts, self.nLs, self.Ls, self.frozenJKao_kpts).real
		
		H1e_Ls = self.get_oei_Ls(supcell)
		self.loc_ActOEI_Ls = reduce(np.dot, (WFs.T, H1e_Ls + frozenJKao_Ls, WFs))

		# Two-electron integral for the active part
		if self.ERIinMEM == True:
			self.loc_ActTEI_Ls = ao2mo.outcore.full_iofree(supcell, WFs, compact=False).reshape(self.norbs, self.norbs, self.norbs, self.norbs)
		else:
			pass
			#TODO: ERI on-a-fly
			
		# Fock for the active part		
		fock_kpts = kmf.get_fock()
		self.loc_ActFOCK_kpts = self.to_localbas(fock_kpts, self.CO)	
		self.loc_ActFOCK_Ls = libdmet.iFTWFs(self.nkpts, self.kpts, self.nLs, self.Ls, self.loc_ActFOCK_kpts).real	
		
	def construct_locOED_kpts(self, umat, OEH_type, doSCF = False):
		'''
		Construct MOs/one-electron density matrix at each k-point in the local basis
		with a certain k-independent correlation potential umat
		'''	

		#Two choices for the one-electron Hamiltonian
		if OEH_type == 'OEI':
			OEH_kpts = self.loc_ActOEI_kpts + umat
		elif OEH_type == 'FOCK':
			OEH_kpts = self.loc_ActFOCK_kpts + umat
		else:
			raise Exception('the current one-electron Hamiltonian type is not supported')
	
		if doSCF == False:
			eigvals_kpts, eigvecs_kpts = np.linalg.eigh(OEH_kpts)
			idx_kpts = eigvals_kpts.argsort()
			eigvals_kpts = np.asarray([eigvals_kpts[kpt][idx_kpts[kpt]] for kpt in range(self.nkpts)], dtype=np.float64)
			eigvecs_kpts = np.asarray([eigvecs_kpts[kpt][:,idx_kpts[kpt]] for kpt in range(self.nkpts)], dtype=np.complex128)
			nelec_pairs = self.nActelecs // 2 
			loc_OED_kpts = np.asarray([2*np.dot(eigvecs_kpts[kpt][:,:nelec_pairs], eigvecs_kpts[kpt][:,:nelec_pairs].T.conj()) for kpt in range(self.nkpts)], dtype=np.complex128)
		else:
			pass
			# TODO
			# rerun a SCF using the new Hamiltonian

		return (eigvecs_kpts, loc_OED_kpts)
		
	def construct_locOED_Ls(self, umat, OEH_type, doSCF = False):
		'''
		Construct MOs/one-electron density matrix dm_{pq}^{0L} at each lattice vector
		with a certain k-independent correlation potential umat
		'''	
	
		eigvecs_kpts, loc_OED_kpts = self.construct_locOED_kpts(umat, OEH_type, doSCF)
		loc_OED_Ls = libdmet.iFTWFs(self.nkpts, self.kpts, self.nLs, self.Ls, loc_OED_kpts).real
		eigenvals_Ls = libdmet.iFTWFs(self.nkpts, self.kpts, self.nLs, self.Ls, eigvecs_kpts).real

		return (eigenvals_Ls, loc_OED_Ls)
		
	def dmet_oei(self, FBEorbs, Norb_in_imp):
		oei = reduce(np.dot,(FBEorbs[:,:Norb_in_imp].T, self.loc_ActOEI_Ls, FBEorbs[:,:Norb_in_imp]))		
		return oei

	def dmet_tei(self, FBEorbs, Norb_in_imp):
		tei = ao2mo.incore.full(ao2mo.restore(8, self.loc_ActTEI_Ls, self.norbs), FBEorbs[:,:Norb_in_imp], compact=False)
		tei = tei.reshape(Norb_in_imp, Norb_in_imp, Norb_in_imp, Norb_in_imp)
		return tei		

	def dmet_corejk(self, FBEorbs, Norb_in_imp, core1RDM_loc):
		J = np.einsum('pqrs,rs->pq', self.loc_ActTEI_Ls, core1RDM_loc)
		K = np.einsum('prqs,rs', self.loc_ActTEI_Ls, core1RDM_loc)	
		JK = reduce(np.dot,(FBEorbs[:,:Norb_in_imp].T, J -0.5*K, FBEorbs[:,:Norb_in_imp]))		
		return JK
	
	def make_supcell(self, cell, nimgs):
		'''
		Make the computational supercell instance used to get the oei and tei in real space
		Note: this orbitals are consistent with the self.Ls vector
		'''
		from pyscf.pbc.tools import pbc
		supcell = cell.copy()
		a = cell.lattice_vectors()
		Ts = lib.cartesian_prod((np.arange(nimgs[0])-nimgs[0]//2,
						np.arange(nimgs[1])-nimgs[1]//2,
						np.arange(nimgs[2])-nimgs[2]//2))
		Ls = np.dot(Ts, a)
		symbs = [atom[0] for atom in cell._atom] * len(Ls)
		coords = Ls.reshape(-1,1,3) + cell.atom_coords()
		supcell.atom = list(zip(symbs, coords.reshape(-1,3)))
		supcell.unit = 'B'
		supcell.a = np.einsum('i,ij->ij', nimgs, a)
		supcell.mesh = np.array([nimgs[0]*cell.mesh[0],
								 nimgs[1]*cell.mesh[1],
								 nimgs[2]*cell.mesh[2]])
		supcell.build(False, False, verbose=0)
		supcell.verbose = cell.verbose
		return supcell		

	def make_WFs(self, w90):
		'''
		Compute the Wannier functions at the reference cell in the basis of local Gaussian
		'''
		
		WFs_kpts = []
		for k_id, kpt in enumerate(self.kpts):
			mo_included = w90.mo_coeff_kpts[k_id][:,w90.band_included_list]
			mo_in_window = w90.lwindow[k_id]
			C_opt = mo_included[:,mo_in_window].dot(w90.U_matrix_opt[k_id].T)
			WFs_kpts.append(C_opt.dot(w90.U_matrix[k_id].T))		
			
		WFs_kpts = np.asarray(WFs_kpts, dtype=np.complex128)
		WFs = libdmet.iFTWFs(self.nkpts, self.kpts, self.nLs, self.Ls, WFs_kpts).real
		return WFs_kpts, WFs
		
	def to_Ls_space(self, Mat_kpts):
		'''
		Transform an one-electron matrix M_{pq}(k) to the L-space
		'''
		return libdmet.iFTWFs(self.nkpts, self.kpts, self.nLs, self.Ls, Mat_kpts).real 
		
	def to_Ls_space_sparse(self, Mat_kpt, kpt):
		'''
		Transform an one-electron matrix M_{pq}(k) to the L-space
		'''
		return libdmet.iFTWFs_sparse(self.nkpts, kpt, self.nLs, self.Ls, Mat_kpt).real 		

	def to_localbas(self, Mat_kpts, CO):
		'''
		Transform an one-electron operator M_{pq}(k) to the local basis
		'''
		loc_Mat_kpts = np.asarray([reduce(np.dot, (CO[kpt].T.conj(), Mat_kpts[kpt], CO[kpt])) for kpt in range(self.nkpts)], dtype=np.complex128)
		return loc_Mat_kpts
		
	def get_oei_Ls(self, supcell):
		'''
		Get the Hcore_{pq}^{0L}
		'''
		kin = supcell.intor_symmetric('cint1e_kin_sph')
		nuc = supcell.intor_symmetric('cint1e_nuc_sph')
		return kin + nuc
		
	