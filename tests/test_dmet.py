'''
Testing the implementation of the DMET class.
'''

import pyscf
from pyscf import gto, scf, ao2mo
import numpy as np
import pytest
from mdmet import orthobasis, schmidtbasis, qcsolvers, dmet

def test_makemole1():
	bondlength = 1.0
	nat = 10
	mol = gto.Mole()
	mol.atom = []
	r = 0.5 * bondlength / np.sin(np.pi/nat)
	for i in range(nat):
		theta = i * (2*np.pi/nat)
		mol.atom.append(('H', (r*np.cos(theta), r*np.sin(theta), 0)))

	mol.basis = 'sto-3g'
	mol.build(verbose=0)

	mf = scf.RHF(mol)
	mf.scf()
	atoms_per_imp = 2 # Impurity size = 1 atom
	Norbs = mol.nao_nr()
	assert ( nat % atoms_per_imp == 0 )
	orbs_per_imp = int(Norbs * atoms_per_imp // nat)

	impClusters = []
	for cluster in range(nat // atoms_per_imp):
		impurities = np.zeros([Norbs], dtype=int)
		for orb in range( orbs_per_imp ):
			impurities[orbs_per_imp*cluster + orb] = 1
		impClusters.append(impurities)

	return mol, mf, impClusters 

def test_makemole2():
	bondlength = 1.0
	nat = 10
	mol = gto.Mole()
	mol.atom = []
	r = 0.5 * bondlength / np.sin(np.pi/nat)
	for i in range(nat):
		theta = i * (2*np.pi/nat)
		if i%3 == 0: 
			element = 'He'
		else:
			element = 'Be'
		mol.atom.append((element, (r*np.cos(theta), r*np.sin(theta), 0)))

	mol.basis = 'sto-3g'
	mol.build(verbose=0)

	mf = scf.RHF(mol)
	mf.scf()
	atoms_per_imp = 2 # Impurity size = 1 atom
	Norbs = mol.nao_nr()
	assert ( nat % atoms_per_imp == 0 )

	#Parition: (He-Be)-(BeHe)-(BeBe)-(HeBe)-(BeHe) = 6-6-10-6-6 orb
	impClusters = []
	for cluster in range(nat // atoms_per_imp):
		if cluster == 2:
			impurities = np.zeros([Norbs], dtype=int)
			impurities[12:22] = 1
		elif cluster == 3:
			impurities = np.zeros([Norbs], dtype=int)
			impurities[22:28] = 1	
		elif cluster == 4:
			impurities = np.zeros([Norbs], dtype=int)
			impurities[28:34] = 1			
		else:
			impurities = np.zeros([Norbs], dtype=int)
			impurities[(6*cluster):(6*(cluster+1))] = 1		
		impClusters.append(impurities)
	assert (sum(impClusters).sum() == Norbs)
	return mol, mf, impClusters 
	
def test_kernel():
	mol, mf, impClusters  = test_makemole2()
	symmetry = None
	runDMET = dmet.DMET(mf, impClusters, symmetry, orthogonalize_method = 'overlap', schmidt_decomposition_method = 'OED', OEH_type = 'FOCK', SC_CFtype = 'FB', solver = 'RHF')
	Nelecs = runDMET.kernel()
	Etotal = runDMET.fragment_energies.sum()

	assert np.isclose(Nelecs, mol.nelectron)
	assert np.isclose(Etotal, mf.energy_elec()[0])

	
def test_one_shot_DMET():
	mol, mf, impClusters  = test_makemole2()	
	symmetry = [0, 1, 2, 3, 4]
	runDMET = dmet.DMET(mf, impClusters, symmetry, orthogonalize_method = 'overlap', schmidt_decomposition_method = 'OED', OEH_type = 'FOCK', SC_CFtype = 'FB', solver = 'RHF')
	runDMET.embedding_symmetry = [0, 1, 2, 3, 4]
	E_total = runDMET.one_shot()
	Nelecs = runDMET.fragment_nelecs.sum()
	Etotal = runDMET.fragment_energies.sum()

	assert np.isclose(Nelecs, mol.nelectron)
	assert np.isclose(Etotal, mf.energy_elec()[0])

	
def test_single_embedding():
	mol, mf, impClusters  = test_makemole2()
	impClusters = [impClusters[0]]
	symmetry = None
	runDMET = dmet.DMET(mf, impClusters, symmetry, orthogonalize_method = 'overlap', schmidt_decomposition_method = 'overlap', OEH_type = 'FOCK', SC_CFtype = 'FB', solver = 'RHF')
	runDMET.single_embedding = True
	runDMET.one_shot()
	E_total = runDMET.Energy_total
	Nelecs = runDMET.fragment_nelecs.sum()	

	assert np.isclose(Nelecs, mol.nelectron)	
	assert np.isclose(E_total, mf.e_tot)
	
def test_rdm_diff():
	mol, mf, impClusters  = test_makemole2()
	symmetry = [0, 1, 2, 1, 0]
	runDMET = dmet.DMET(mf, impClusters, symmetry, orthogonalize_method = 'overlap', schmidt_decomposition_method = 'OED', OEH_type = 'FOCK', SC_CFtype = 'FB', solver = 'RHF')
	runDMET.one_shot()
	uvec_size = runDMET.uvec.size
	umat1 = np.zeros(uvec_size)	
	umat2 = np.random.rand(uvec_size)	
	CF1 = runDMET.costfunction(umat1)
	CF2 = runDMET.costfunction(umat2) 
	assert (CF1 < 1e-8)
	assert (CF2 > 1e-5)