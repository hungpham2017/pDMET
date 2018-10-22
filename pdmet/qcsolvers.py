'''
Multipurpose Density Matrix Embedding theory (mp-DMET)
Hung Q. Pham
email: pqh3.14@gmail.com
'''

import numpy as np
import sys, os, ctypes
from functools import reduce
import PyCheMPS2
import pyscf
from pyscf import gto, scf, mcscf, dmrgscf, ao2mo

class QCsolvers:
	def __init__(self, OEI, TEI, JK, DMguess, Norb, Nel, Nimp, chempot = 0.0, periodic = False):
		self.OEI = OEI
		self.TEI = TEI
		self.FOCK = OEI + JK
		self.DMguess = DMguess
		self.Norb = Norb
		self.Nel = Nel
		self.Nimp = Nimp
		self.chempot = chempot
		self.periodic = periodic
		
	def RHF(self):
		'''
		Restricted Hartree-Fock
		'''		
		
		Nimp = self.Nimp
		FOCK = self.FOCK.copy()
		
		if (self.chempot != 0.0):
			for orb in range(Nimp):
				FOCK[orb, orb] -= self.chempot	
		
		mol = gto.Mole()
		mol.build(verbose = 0)
		mol.atom.append(('C', (0, 0, 0)))
		mol.nelectron = self.Nel
		mol.incore_anyway = True
		mol.verbose = 0
		mf = scf.RHF(mol)
		mf.get_hcore = lambda *args: FOCK
		mf.get_ovlp = lambda *args: np.eye(self.Norb)
		mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
		mf.scf(self.DMguess)
		
		DMloc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
		if ( mf.converged == False ):
			mf.newton().kernel(dm0=DMloc)
			DMloc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
		
		ERHF = mf.e_tot
		RDM1 = mf.make_rdm1()
		JK   = mf.get_veff(None, dm=RDM1)
	 
		# for pDMET: get the correlation energy instead of the total energy: E_corr = E_high - E_RHF 
		if self.periodic == True:	 
			ImpurityEnergy = 0.0
		else:
			# To calculate the impurity energy, rescale the JK matrix with a factor 0.5 to avoid double counting: 0.5 * ( OEI + FOCK ) = OEI + 0.5 * JK
			ImpurityEnergy = 0.5*np.einsum('ij,ij->', RDM1[:Nimp,:], self.OEI[:Nimp,:] + FOCK[:Nimp,:]) \
							+ 0.5*np.einsum('ij,ij->', RDM1[:Nimp,:], JK[:Nimp,:])

		return (ImpurityEnergy, ERHF, RDM1)
		
	def UHF(self):
		'''
		Unrestricted Hartree-Fock
		'''		
		pass		

	def CCSD(self):
		'''
		Couple-cluster Singly-Doubly 
		'''		
		pass			
		
	def DMRG(self):
		'''
		Density Matrix Renormalization Group using CheMPS2 library 
		NOTE: this is still under testing.			
		'''		
		Norb = self.Norb
		Nimp = self.Nimp
		FOCK = self.FOCK.copy()	
		
		CheMPS2print = False		
		Initializer = PyCheMPS2.PyInitialize()
		Initializer.Init()
		Group = 0
		orbirreps = np.zeros([Norb], dtype=ctypes.c_int)
		HamCheMPS2 = PyCheMPS2.PyHamiltonian(Norb, Group, orbirreps)
		
		#Feed the 1e and 2e integral (T and V)
		for cnt1 in range(Norb):
				for cnt2 in range(Norb):
					HamCheMPS2.setTmat(cnt1, cnt2, FOCK[cnt1, cnt2])
					for cnt3 in range(Norb):
						for cnt4 in range(Norb):
								HamCheMPS2.setVmat(cnt1, cnt2, cnt3, cnt4, self.TEI[cnt1, cnt3, cnt2, cnt4]) #From chemist to physics notation		

		if (self.chempot != 0.0):
			for orb in range(Nimp):
				HamCheMPS2.setTmat(orb, orb, FOCK[orb, orb] - self.chempot)	

		if CheMPS2print == False:
			sys.stdout.flush()
			old_stdout = sys.stdout.fileno()
			new_stdout = os.dup(old_stdout)
			devnull = os.open('/dev/null', os.O_WRONLY)
			os.dup2(devnull, old_stdout)
			os.close(devnull)
			
		assert( self.Nel % 2 == 0 )
		TwoS  = 0
		Irrep = 0
		Prob  = PyCheMPS2.PyProblem( HamCheMPS2, TwoS, self.Nel, Irrep )

		OptScheme = PyCheMPS2.PyConvergenceScheme(4) # 3 instructions
		#OptScheme.setInstruction(instruction, D, Econst, maxSweeps, noisePrefactor)
		OptScheme.setInstruction(0,  200, 1e-8,  5, 0.03)		
		OptScheme.setInstruction(1,  500, 1e-8,  5, 0.03)
		OptScheme.setInstruction(2, 1000, 1e-8,  5, 0.03)
		OptScheme.setInstruction(3, 1000, 1e-8,  100, 0.00) # Last instruction a few iterations without noise

		theDMRG = PyCheMPS2.PyDMRG( Prob, OptScheme )
		EDMRG = theDMRG.Solve()
		theDMRG.calc2DMandCorrelations()
		RDM2 = np.zeros( [Norb, Norb, Norb, Norb], dtype=ctypes.c_double )
		for orb1 in range(Norb):
			for orb2 in range(Norb):
				for orb3 in range(Norb):
					for orb4 in range(Norb):
						RDM2[ orb1, orb3, orb2, orb4 ] = theDMRG.get2DMA( orb1, orb2, orb3, orb4 ) #From physics to chemistry notation

		# theDMRG.deleteStoredMPS()
		theDMRG.deleteStoredOperators()
		del(theDMRG)
		del(OptScheme)
		del(Prob)
		del(HamCheMPS2)
		del(Initializer)	

		if CheMPS2print == False:		
			sys.stdout.flush()
			os.dup2(new_stdout, old_stdout)
			os.close(new_stdout)
			
		RDM1 = np.einsum('ijkk->ij', RDM2)/(self.Nel - 1)
		
		ImpurityEnergy = 0.5 * np.einsum('ij,ij->', RDM1[:Nimp,:], FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.5 * np.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:])		

		return (ImpurityEnergy, EDMRG, RDM1)	

	def FCI(self):
		'''
		Full Configuration Interaction (FCI) using CheMPS2 library
		NOTE: this is still under testing.		
		'''		
		Norb = self.Norb
		Nimp = self.Nimp
		FOCK = self.FOCK.copy()	
		
		CheMPS2print = False
		Initializer = PyCheMPS2.PyInitialize()
		Initializer.Init()
		Group = 0
		orbirreps = np.zeros([Norb], dtype=ctypes.c_int)
		HamCheMPS2 = PyCheMPS2.PyHamiltonian(Norb, Group, orbirreps)
		
		#Feed the 1e and 2e integral (T and V)
		for cnt1 in range(Norb):
				for cnt2 in range(Norb):
					HamCheMPS2.setTmat(cnt1, cnt2, FOCK[cnt1, cnt2])
					for cnt3 in range(Norb):
						for cnt4 in range(Norb):
								HamCheMPS2.setVmat(cnt1, cnt2, cnt3, cnt4, self.TEI[cnt1, cnt3, cnt2, cnt4]) #From chemist to physics notation		

		if (self.chempot != 0.0):
			for orb in range(Nimp):
				HamCheMPS2.setTmat(orb, orb, FOCK[orb, orb] - self.chempot)	

		if CheMPS2print == False:
			sys.stdout.flush()
			old_stdout = sys.stdout.fileno()
			new_stdout = os.dup(old_stdout)
			devnull = os.open('/dev/null', os.O_WRONLY)
			os.dup2(devnull, old_stdout)
			os.close(devnull)
		
		assert( self.Nel % 2 == 0 )
		Nel_up       = self.Nel / 2
		Nel_down     = self.Nel / 2
		Irrep= 0
		maxMemWorkMB = 1000.0
		FCIverbose   = 2
		theFCI = PyCheMPS2.PyFCI(HamCheMPS2, Nel_up, Nel_down, Irrep, maxMemWorkMB, FCIverbose)
		GSvector = np.zeros([theFCI.getVecLength() ], dtype=ctypes.c_double)
		theFCI.FillRandom(theFCI.getVecLength() , GSvector) # Random numbers in [-1,1]
		GSvector[ theFCI.LowestEnergyDeterminant() ] = 12.345 # Large component for quantum chemistry
		EFCI = theFCI.GSDavidson( GSvector )
		#SpinSquared = theFCI.CalcSpinSquared( GSvector )
		RDM2 = np.zeros( [ Norb**4 ], dtype=ctypes.c_double )
		theFCI.Fill2RDM( GSvector, RDM2 )
		RDM2 = RDM2.reshape( [Norb, Norb, Norb, Norb], order='F' )
		RDM2 = np.swapaxes( RDM2, 1, 2 ) #From physics to chemistry notation
		del theFCI
		del HamCheMPS2

		if CheMPS2print == False:		
			sys.stdout.flush()
			os.dup2(new_stdout, old_stdout)
			os.close(new_stdout)
		
		RDM1 = np.einsum('ijkk->ij', RDM2)/(self.Nel - 1)
		ImpurityEnergy = 0.50  * np.einsum('ij,ij->',     RDM1[:Nimp,:],     FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * np.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * np.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * np.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * np.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])			

		return (ImpurityEnergy, EFCI, RDM1)
		
	def CAS(self, CAS, CAS_MO, Orbital_optimization = False, solver = 'FCI'):
		'''
		CASSCF with FCI or DMRG solver from BLOCK or CheMPS2
		'''		
		Norb = self.Norb
		Nimp = self.Nimp
		FOCK = self.FOCK.copy()
		
		if (self.chempot != 0.0):
			for orb in range(Nimp):
				FOCK[orb, orb] -= self.chempot	
				
		mol = gto.Mole()
		mol.build(verbose = 0)
		mol.atom.append(('C', (0, 0, 0)))
		mol.nelectron = self.Nel
		mol.incore_anyway = True
		mf = scf.RHF( mol )
		mf.get_hcore = lambda *args: FOCK
		mf.get_ovlp = lambda *args: np.eye(Norb)
		mf._eri = ao2mo.restore(8, self.TEI, Norb)
		mf.scf(self.DMguess)
		DMloc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
		
		if ( mf.converged == False ):
			mf.newton().kernel(dm0=DMloc)
			DMloc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
			
		if CAS == None:
			CAS_nelec = self.Nel
			CAS_norb = Norb
			CAS = 'full'
		else:
			CAS_nelec = CAS[0]
			CAS_norb = CAS[1]
		print("     Active space: ", CAS)
		
		# Replace FCI solver by DMRG solver in CheMPS2 or BLOCK
		if Orbital_optimization == True: 
			mc = mcscf.CASSCF(mf, CAS_norb, CAS_nelec)	
		else:
			mc = mcscf.CASCI(mf, CAS_norb, CAS_nelec)	
			
		if solver == 'CheMPS2':
			mc.fcisolver = dmrgscf.CheMPS2(mol)
		elif solver == 'Block':
			mc.fcisolver = dmrgscf.DMRGCI(mol)		
		
		if CAS_MO is not None: 
			print("     Active space MOs: ", CAS_MO)
			mo = mc.sort_mo(CAS_MO)
			ECAS = mc.kernel(mo)[0]
		else:
			ECAS = mc.kernel()[0]
		
		###### Get RDM1 + RDM2 #####
		CAS_norb = mc.ncas
		core_norb = mc.ncore
		CAS_nelec = mc.nelecas	
		core_MO = mc.mo_coeff[:,:core_norb]
		CAS_MO = mc.mo_coeff[:,core_norb:core_norb+CAS_norb]

	
		casdm1 = mc.fcisolver.make_rdm12(mc.ci, CAS_norb, CAS_nelec)[0] #in CAS space
		# Transform the casdm1 (in CAS space) to casdm1ortho (orthonormal space).     
		casdm1ortho = np.einsum('ap,pq->aq', CAS_MO, casdm1)
		casdm1ortho = np.einsum('bq,aq->ab', CAS_MO, casdm1ortho)
		coredm1 = np.dot(core_MO, core_MO.T) * 2 #in localized space
		RDM1 = coredm1 + casdm1ortho	

		casdm2 = mc.fcisolver.make_rdm12(mc.ci, CAS_norb, CAS_nelec)[1] #in CAS space
		# Transform the casdm2 (in CAS space) to casdm2ortho (orthonormal space). 
		casdm2ortho = np.einsum('ap,pqrs->aqrs', CAS_MO, casdm2)
		casdm2ortho = np.einsum('bq,aqrs->abrs', CAS_MO, casdm2ortho)
		casdm2ortho = np.einsum('cr,abrs->abcs', CAS_MO, casdm2ortho)
		casdm2ortho = np.einsum('ds,abcs->abcd', CAS_MO, casdm2ortho)	
	
		coredm2 = np.zeros([Norb, Norb, Norb, Norb]) #in AO
		coredm2 += np.einsum('pq,rs-> pqrs',coredm1,coredm1)
		coredm2 -= 0.5*np.einsum('ps,rq-> pqrs',coredm1,coredm1)
	
		effdm2 = np.zeros([Norb, Norb, Norb, Norb]) #in AO
		effdm2 += 2*np.einsum('pq,rs-> pqrs',casdm1ortho,coredm1)
		effdm2 -= np.einsum('ps,rq-> pqrs',casdm1ortho,coredm1)				
					
		RDM2 = coredm2 + casdm2ortho + effdm2

		ImpurityEnergy = 0.50  * np.einsum('ij,ij->',     RDM1[:Nimp,:],     FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * np.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * np.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * np.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * np.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])	
					   					   
		# for pDMET: get the correlation energy instead of the total energy: E_corr = E_high - E_RHF 
		if self.periodic == True:
			RDM1_rhf = mf.make_rdm1()
			JK   = mf.get_veff(None, dm=RDM1_rhf)
			ImpurityEnergy_from_HF = 0.5*np.einsum('ij,ij->', RDM1_rhf[:Nimp,:], self.OEI[:Nimp,:] + FOCK[:Nimp,:]) \
									+ 0.5*np.einsum('ij,ij->', RDM1_rhf[:Nimp,:], JK[:Nimp,:])								
			ImpurityEnergy += -ImpurityEnergy_from_HF		

		return (ImpurityEnergy, ECAS, RDM1)			