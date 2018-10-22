'''
Multipurpose Density Matrix Embedding theory (mp-DMET)
Hung Q. Pham
email: pqh3.14@gmail.com
'''

import os, sys
import numpy as np
from scipy import optimize
from functools import reduce
from pdmet import localbasis, schmidtbasis, helper, qcsolvers
from pdmet.lib.build import libdmet
from pyscf.lib import logger

class pDMET:
	def __init__(self, cell, kmf, w90, kmf_chkfile = None, w90_chkfile = None, schmidt_decomposition_method = 'OED', OEH_type = 'FOCK', SC_CFtype = 'F', solver = 'RHF', umat_kpt = False):
		'''
		Args:
			kmf 							: a rhf wave function from pyscf/pbc
			w90								: a converged wannier90 object
			schmidt_decomposition_method	: OED/overlap
			OEH_type						: One-electron Hamiltonian used in the bath construction, h(k) = OEH(k) + umat(k) 
			SCmethod						: CG/SLSQP/BFGS/L-BFGS-B/LSTSQ self-consistent iteration method, defaut: BFGS
			SC_threshold					: convergence criteria for correlation potential, default: 1e-6
			SC_maxcycle                 	: maximum cycle for self-consistent iteration, default: 50
			SC_CFtype						: FB/diagFB/F/diagF, cost function type, fitting 1RDM of the entire schmidt basis (FB), diagonal FB (diagFB), 
											  fragment (F), or diagonal elements of fragment only (diagF), default: FB
			umat							: correlation potential
			chempot							: global chemical potential
			emb_1RDM						: a list of the 1RDM for each fragment
			emb_orbs						: a list of the fragment and bath orbitals for each fragment			
		Return:
		
		'''		
		
		# General initialized attributes 
		if kmf_chkfile != None:
			self.kmf = helper.load_kmf(kmf, kmf_chkfile)
		else:
			self.kmf = kmf     
            
		if w90_chkfile != None:
			self.w90 = helper.load_w90(w90_chkfile)
		else:
			self.w90 = w90
     
		self.cell = cell
		self.localbasis = localbasis.main(cell, self.kmf, self.w90)
		self.sd_type = schmidt_decomposition_method
		self.OEH_type = OEH_type		
		self.nkpts = self.kmf.kpts.shape[0]
		self.nimporbs = self.localbasis.nActorbs
		self.Norbs = self.localbasis.norbs
		self.Nelecs = self.localbasis.nelec
		self.numPairs = self.localbasis.nActelecs//2		
		
		# Labeling the reference unit cell as the fragment
		# The nkpts is an odd number so the reference unit is assumed to be in the middle of computational supercell
		self.impCluster = np.zeros((self.Norbs))
		self.impCluster[self.nimporbs*(self.nkpts//2):self.nimporbs*(self.nkpts//2 + 1)] = 1
		
		# QC Solver	
		solver_list = ['RHF', 'CASCI', 'CASSCF', 'DMRG-CASCI-C', 'DMRG-CASSCF-C', 'DMRG-CASCI-B', 'DMRG-CASSCF-B', 'CCSD']
		assert solver in solver_list
		self.solver = solver
		self.CAS = None	# (n,m) means n electron in m orbitals
		self.CAS_MO = None

		# Self-consistent parameters		
		self.SC_method = 'BFGS'		# BFGS, L-BFGS-B, CG, Newton-CG, LEASTSQ
		self.umat_kpt = umat_kpt
		self.SC_threshold = 1e-6
		self.SC_maxcycle =	200
		assert (SC_CFtype in ['F','diagF'])
		self.SC_CFtype = SC_CFtype
		self.SC_damping = 0.0

		# Correlation/chemical potential
		self.H1start, self.H1row, self.H1col = self.make_H1()[1:4]	#Use in the calculation of 1RDM derivative
		self.Nterms = self.nimporbs*(self.nimporbs + 1) // 2
		if self.umat_kpt == True: 
			self.kNterms = self.nkpts*self.Nterms
		else:	
			self.kNterms = self.Nterms
			
		self.uvec = np.zeros(self.kNterms, dtype=np.float64)
		self.chempot = 0.0

		# DMET Output
		self.verbose = 1
		self.locOED_Ls = None
		self.baths = None
		self.emb_1RDM = None
		self.emb_orbs = None
		self.energy_per_cell = None	
		self.nelec_per_cell = None
		
		# Others
		self.doSCF = False		# Whether simply diagonalize the H = H0 + H1 to get the new 1-RDM or rerun a SCF with this Hamiltonian
		np.set_printoptions(precision=6)
		
	def kernel(self, chempot = 0.0):
		'''
		This is the main kernel for DMET calculation.
		It is solving the embedding problem, then returning the total number of electrons per unit cell 
		and updating the schmidt orbitals and 1RDM.
		Args:
			chempot					: global chemical potential to adjust the number of electrons in the unit cell
		Return:
			nelecs		 			: the total number of electrons
		Update the class attributes:
			energy					: the energy for the unit cell  
			nelec					: the number of electrons for the unit cell	
			emb_1RDM				: the 1RDM for the unit cell				
		'''			
		
		numImpOrbs = self.nimporbs		
		numBathOrbs = self.nimporbs	
		numBathOrbs, FBEorbs, envOrbs_or_core_eigenvals = self.baths
		Norb_in_imp  = numImpOrbs + numBathOrbs
		assert(Norb_in_imp <= self.Norbs)
		if self.sd_type == 'OED' :
			core_cutoff = 0.001
			for cnt in range(len(envOrbs_or_core_eigenvals)):
				if (envOrbs_or_core_eigenvals[cnt] < core_cutoff):
					envOrbs_or_core_eigenvals[cnt] = 0.0
				elif (envOrbs_or_core_eigenvals[cnt] > 2.0 - core_cutoff):
					envOrbs_or_core_eigenvals[cnt] = 2.0
				else:
					print ("Bad DMET bath orbital selection: trying to put a bath orbital with occupation", envOrbs_or_core_eigenvals[cnt], "into the environment :-(.")
					assert(0 == 1)	
			Nelec_in_imp = int(round(self.Nelecs - np.sum(envOrbs_or_core_eigenvals)))
			Nelec_in_environment = int(np.sum(np.abs(envOrbs_or_core_eigenvals)))				
			core1RDM_local = reduce(np.dot, (FBEorbs, np.diag(envOrbs_or_core_eigenvals), FBEorbs.T))
		elif self.sd_type == 'overlap':
			Nelec_in_imp = int(2*numImpOrbs)
			Nelec_in_environment = int(self.Nelecs - Nelec_in_imp)
			core1RDM_local = 2*np.dot(FBEorbs[:,Norb_in_imp:], FBEorbs[:,Norb_in_imp:].T)				
			
		#Transform the 1e/2e integrals and the JK core constribution to schmidt basis
		dmetOEI  = self.localbasis.dmet_oei(FBEorbs, Norb_in_imp)
		dmetTEI  = self.localbasis.dmet_tei(FBEorbs, Norb_in_imp)			
		dmetCoreJK = self.localbasis.dmet_corejk(FBEorbs, Norb_in_imp, core1RDM_local)

		#Solving the embedding problem with high level wfs
		print("    The embedding problem of [%2d eletrons in (%2d fragment + %2d bath )] is solved by %s" % (Nelec_in_imp, numImpOrbs, numBathOrbs, self.solver))						
		DMguess = reduce(np.dot,(FBEorbs[:,:Norb_in_imp].T, self.locOED_Ls[1], FBEorbs[:,:Norb_in_imp]))	
		
		qcsolver = qcsolvers.QCsolvers(dmetOEI, dmetTEI, dmetCoreJK, DMguess, Norb_in_imp, Nelec_in_imp, numImpOrbs, chempot, periodic = True)
		if self.solver == 'RHF':
			Imp_corrEnergy, E_emb, RDM1 = qcsolver.RHF()
		elif self.solver == 'UHF':
			pass
		elif self.solver == 'CASCI':
			Imp_corrEnergy, E_emb, RDM1 = qcsolver.CAS(self.CAS, self.CAS_MO, Orbital_optimization = False)
		elif self.solver == 'CASSCF':
			Imp_corrEnergy, E_emb, RDM1 = qcsolver.CAS(self.CAS, self.CAS_MO, Orbital_optimization = False)		
		elif self.solver == 'DMRG-CASCI-C':
			Imp_corrEnergy, E_emb, RDM1 = qcsolver.CAS(self.CAS, self.CAS_MO, Orbital_optimization = False, solver = 'CheMPS2')
		elif self.solver == 'DMRG-CASSCF-C':
			Imp_corrEnergy, E_emb, RDM1 = qcsolver.CAS(self.CAS, self.CAS_MO, Orbital_optimization = True, solver = 'CheMPS2')			
		elif self.solver == 'DMRG-CASCI-B':
			Imp_corrEnergy, E_emb, RDM1 = qcsolver.CAS(self.CAS, self.CAS_MO, Orbital_optimization = False, solver = 'Block')
		elif self.solver == 'DMRG-CASSCF-B':
			Imp_corrEnergy, E_emb, RDM1 = qcsolver.CAS(self.CAS, self.CAS_MO, Orbital_optimization = True, solver = 'Block')						
		elif self.solver == 'CCSD':
			pass			
			
		dmetCore1RDM = reduce(np.dot,(FBEorbs[:,:Norb_in_imp].T, core1RDM_local, FBEorbs[:,:Norb_in_imp]))
		self.emb_1RDM = RDM1
		self.emb_orbs = FBEorbs[:,:Norb_in_imp]
		self.nelec_per_cell = np.trace(RDM1[:numImpOrbs,:numImpOrbs])
		self.energy_per_cell = self.localbasis.e_tot + Imp_corrEnergy

		return self.nelec_per_cell, self.energy_per_cell, dmetCore1RDM

	def one_shot(self):
		'''
		Do one-shot DMET, only the chemical potential is optimized
		'''
			
		print("-- ONE-SHOT pDMET CALCULATION : START --")		
		# Optimize the chemical potential
		self.locOED_Ls = self.localbasis.construct_locOED_Ls(self.uvec2umat(self.uvec), self.OEH_type, self.doSCF)		# get both MO coefficients and 1-RDM in the local basis
		schmidt = schmidtbasis.RHF_decomposition(self.cell, self.impCluster, self.nimporbs, self.locOED_Ls)
		schmidt.method = self.sd_type		
		self.baths = schmidt.baths()       
		self.chempot = optimize.newton(self.nelecs_costfunction, self.chempot)
		
		print(" Number of electrons per unit cell: ", self.nelec_per_cell)	
		print(" Total energy per unit cell       : ", self.energy_per_cell)			
		print("-- ONE-SHOT pDMET CALCULATION : END --")
		print()			
		
	def self_consistent(self):
		'''
		Do self-consistent pDMET
		'''	
		print("---------------------------------------------")
		print("- SELF-CONSISTENT pDMET CALCULATION : START -")
		
		u_diff = 1.0
		umat = np.zeros((self.nimporbs, self.nimporbs))
		Ecell = self.localbasis.e_tot
		for cycle in range(self.SC_maxcycle):
			
			print("--- pDMET cycle : ", cycle + 1, " ---")
			umat_old = umat
			Ecell_old = Ecell
			# Do one-shot with each uvec
			self.one_shot()
			print (" Chemical potential = ", self.chempot)

			# Optimize uvec
			if self.SC_method == 'BFGS':
				result = optimize.minimize(self.costfunction, self.uvec, method='BFGS', jac = self.costfunction_gradient)
			elif self.SC_method == 'L-BFGS-B':
				result = optimize.minimize(self.costfunction, self.uvec, method='L-BFGS-B', jac = self.costfunction_gradient)
			elif self.SC_method == 'CG':
				result = optimize.minimize(self.costfunction, self.uvec, method='CG', jac = self.costfunction_gradient)				
			elif self.SC_method == 'Newton-CG':
				result = optimize.minimize(self.costfunction, self.uvec, method='Newton-CG', jac = self.costfunction_gradient)				
			elif self.SC_method == 'LEASTSQ':
				result = optimize.leastsq(self.costfunction, self.uvec, factor=0.1)
			else:
				print(self.SC_method, " is not supported")
			print 
			if result.success == False: 
				print("------------------------------------------------", )				
				print(" WARNING: Correlation potential is not converged", )	
				print("------------------------------------------------", )				
			self.uvec = result.x
			Ecell = self.energy_per_cell
			umat = self.uvec2umat(self.uvec)
			umat = umat - np.eye(umat.shape[0])*np.average(np.diag(umat)) #TODO: check whether it can be removed
			u_diff = np.linalg.norm(umat_old - umat)
			Ecell_diff = abs(Ecell_old - Ecell)
			umat = self.SC_damping*umat_old + (1.0 - self.SC_damping)*umat #Can be eliminated
			print(" 2-norm of difference old and new u-mat : ", u_diff)
			print(" Energy difference      		: ", Ecell_diff)
			if self.umat_kpt == True:
				uvec = self.uvec.reshape(self.nkpts,-1)
				print("Correlation potential vector: ")
				for k,kpt in enumerate(self.cell.get_scaled_kpts(self.kmf.kpts)):
					logger.debug(self.kmf, '  %2d (%6.3f %6.3f %6.3f)   %s', k, kpt[0], kpt[1], kpt[2], uvec[k])				
			else:
				print("Correlation potential vector: ", self.uvec)
			if (u_diff <= self.SC_threshold) and (Ecell_diff <= self.SC_threshold): break
			
		print("--- SELF-CONSISTENT pDMET CALCULATION : END ---")
		print("-----------------------------------------------")		
		
	def nelecs_costfunction(self, chempot):
		'''
		The different in the correct number of electrons (provided) and the calculated one 
		'''
		
		Nelec_dmet = self.kernel(chempot)[0]
		Nelec_target = self.Nelecs // self.nkpts			
		print ("   Chemical potential , number of electrons per unit cell = " , chempot, "," , Nelec_dmet ,"")

		return abs(Nelec_dmet - Nelec_target)	

	def costfunction(self, uvec):
		'''
		Cost function: \mathbf{CF}(u) = \mathbf{\Sigma}_{rs} (D^{mf}_{rs}(u) - D^{corr}_{rs})^2
		where D^{mf} and D^{corr} are the mean-field and correlated 1-RDM, respectively.
		and D^{mf} = \mathbf{FT}(D^{mf}(k))
		'''
		rdm_diff = self.rdm_diff(uvec)
		return np.power(rdm_diff, 2).sum()
		
	def costfunction_gradient(self, uvec):
		'''
		Analytical derivative of the cost function,
		deriv(CF(u)) = Sum^x [Sum_{rs} (2 * rdm_diff^x_{rs}(u) * deriv(rdm_diff^x_{rs}(u))]
		ref: J. Chem. Theory Comput. 2016, 12, 2706−2719
		'''
		
		rdm_diff = self.rdm_diff(uvec)
		rdm_diff_gradient = self.rdm_diff_gradient(uvec)
		CF_gradient = np.zeros(self.kNterms)
		for u in range(self.kNterms):
			CF_gradient[u] = np.sum(2 * rdm_diff * rdm_diff_gradient[u])
		
		return CF_gradient
		
		
	def rdm_diff(self, uvec):
		'''
		Calculating the different between mf-1RDM (transformed in schmidt basis) and correlated-1RDM for the unit cell
		Args:
			uvec			: the correlation potential vector
		Return:
			error			: an array of errors for the unit cell.
		'''
		
		locOED = self.localbasis.construct_locOED_Ls(self.uvec2umat(uvec), self.OEH_type, self.doSCF)[1]
		mf_1RDM = reduce(np.dot, (self.emb_orbs[:,:self.nimporbs].T, locOED, self.emb_orbs[:,:self.nimporbs]))
		corr_1RDM = self.emb_1RDM[:self.nimporbs,:self.nimporbs]		
		error = mf_1RDM - corr_1RDM
		if self.SC_CFtype == 'diagF': error = np.diag(error)	
			
		return error

	def rdm_diff_gradient(self, uvec):
		'''
		Compute the rdm_diff gradient
		Args:
			uvec			: the correlation potential vector
		Return:
			the_gradient	: a list with the size of the number of u values in uvec
							  Each element of this list is an array of derivative corresponding to each rs.
							 
		'''
		
		the_RDM_deriv_kpts = self.construct_1RDM_response_kpts(uvec)

		the_gradient = []	
		if self.umat_kpt == True:
			for kpt in range(self.nkpts):
				for u in range(self.Nterms):
					RDM_deriv_Ls = self.localbasis.to_Ls_space_sparse(the_RDM_deriv_kpts[kpt,u,:,:], self.kmf.kpts[kpt])	# Transform RDM_deriv from k-space to L-space
					error_deriv_in_schmidt_basis = reduce(np.dot, (self.emb_orbs[:,:self.nimporbs].T, RDM_deriv_Ls, self.emb_orbs[:,:self.nimporbs]))	
					if self.SC_CFtype == 'diagF': error_deriv_in_schmidt_basis = np.diag(error_deriv_in_schmidt_basis)
					the_gradient.append(error_deriv_in_schmidt_basis)					
		else:
			for u in range(self.Nterms):
				RDM_deriv_Ls = self.localbasis.to_Ls_space(the_RDM_deriv_kpts[:,u,:,:])	# Transform RDM_deriv from k-space to L-space
				error_deriv_in_schmidt_basis = reduce(np.dot, (self.emb_orbs[:,:self.nimporbs].T, RDM_deriv_Ls, self.emb_orbs[:,:self.nimporbs]))	
				if self.SC_CFtype == 'diagF': error_deriv_in_schmidt_basis = np.diag(error_deriv_in_schmidt_basis)
				the_gradient.append(error_deriv_in_schmidt_basis)
		
		return the_gradient

######################################## USEFUL FUNCTION for pDMET class ######################################## 

	def uvec2umat(self, uvec):
		'''
		Convert uvec to the umat which is will be added up to the local one-electron Hamiltonian at each k-point
		'''	

		mask = np.zeros([self.nimporbs, self.nimporbs], dtype=bool)		
		mask[np.triu_indices(self.nimporbs)] = True
		if self.umat_kpt == True:
			the_umat = []
			uvec = uvec.reshape(self.nkpts, -1)
			for kpt in range(self.nkpts):
				umat = np.zeros([self.nimporbs, self.nimporbs], dtype=np.float64)
				umat[mask] = uvec[kpt]
				umat = umat.T
				umat[mask] = uvec[kpt]
				the_umat.append(umat)
		else:
			the_umat = np.zeros([self.nimporbs, self.nimporbs], dtype=np.float64)
			the_umat[mask] = uvec
			the_umat = the_umat.T
			the_umat[mask] = uvec
			
		return np.asarray(the_umat)				
		
	def make_H1(self):
		'''
		The H1 is the correlation potential operator, used to calculate gradient of 1-RDM at each k-point
		Return:
			H1start: 
			H1row: 
			H1col: 
		'''
		
		theH1 = []

		if self.SC_CFtype == 'diagF' or self.SC_CFtype == 'diagFB': 		#Only fitting the diagonal elements of umat
			for row in range(self.nimporbs):
				H1 = np.zeros([self.nimporbs, self.nimporbs])
				H1[row, row] = 1
				theH1.append(H1)
		else:		
			for row in range(self.nimporbs):									#Fitting the whole umat
				for col in range(row, self.nimporbs):
					H1 = np.zeros([self.nimporbs, self.nimporbs])
					H1[row, col] = 1
					H1[col, row] = 1								
					theH1.append(H1)	
	
		# Convert the sparse H1 to one dimension H1start, H1row, H1col arrays used in libdmet.rhf_response()
		
		H1start = []
		H1row   = []
		H1col   = []
		H1start.append(0)
		totalsize = 0
		for count in range(len(theH1)):
			rowco, colco = np.where(theH1[count] == 1)
			totalsize += len(rowco)
			H1start.append(totalsize)
			for count2 in range(len(rowco)):
				H1row.append(rowco[count2])
				H1col.append(colco[count2])
		H1start = np.array(H1start)
		H1row   = np.array(H1row)
		H1col   = np.array(H1col)	
		return theH1, H1start, H1row, H1col
		
	def construct_1RDM_response_kpts(self, uvec):
		'''
		Calculate the derivative of 1RDM
		'''
		
		if self.umat_kpt == True:
			rdm_deriv_kpts = []
			loc_ActFOCK_kpts = self.localbasis.loc_ActFOCK_kpts + self.uvec2umat(uvec)
			for kpt in range(self.nkpts):
				rdm_deriv = libdmet.rhf_response_c(self.nimporbs, self.Nterms, self.numPairs, self.H1start, self.H1row, self.H1col, loc_ActFOCK_kpts[kpt])
				rdm_deriv_kpts.append(rdm_deriv)			
		else:
			rdm_deriv_kpts = []
			loc_ActFOCK_kpts = self.localbasis.loc_ActFOCK_kpts + self.uvec2umat(uvec)
			for kpt in range(self.nkpts):
				rdm_deriv = libdmet.rhf_response_c(self.nimporbs, self.Nterms, self.numPairs, self.H1start, self.H1row, self.H1col, loc_ActFOCK_kpts[kpt])
				rdm_deriv_kpts.append(rdm_deriv)
			
		return np.asarray(rdm_deriv_kpts, dtype=np.complex128)		

######################################## POST pDMET ANALYSIS ######################################## 
		
	def get_bands(self, kpts_band, cell=None, dm_kpts=None, kpts=None):
		'''Get energy bands at the given (arbitrary) 'band' k-points.

		Returns:
			mo_energy : (nmo,) ndarray or a list of (nmo,) ndarray
			Bands energies E_n(k)
			mo_coeff : (nao, nmo) ndarray or a list of (nao,nmo) ndarray
			Band orbitals psi_n(k)
		'''
		if cell is None: cell = self.cell
		if kpts is None: kpts = self.kmf.kpts
		frozenDMao_kpts = self.localbasis.frozenDMao_kpts
		activeDMloc_kpts = self.localbasis.construct_locOED_kpts(self.uvec2umat(self.uvec), self.OEH_type, self.doSCF)[1]
		CO = self.localbasis.CO
		activeDMao_kpts = np.einsum('kua,kab,bvk ->kuv', CO, activeDMloc_kpts, CO.T.conj(), optimize=True)
		#Total RDM1:
		dm_kpts = frozenDMao_kpts + activeDMao_kpts
		
		kpts_band = np.asarray(kpts_band)
		single_kpt_band = (kpts_band.ndim == 1)
		kpts_band = kpts_band.reshape(-1,3)
		fock = self.kmf.get_hcore(cell, kpts_band)
		fock = fock + self.kmf.get_veff(cell, dm_kpts, kpts, kpts_band)
		s1e = self.kmf.get_ovlp(cell, kpts_band)
		mo_energy, mo_coeff = self.kmf.eig(fock, s1e)
		if single_kpt_band:
			mo_energy = mo_energy[0]
			mo_coeff = mo_coeff[0]
		
		return mo_energy, mo_coeff
		