'''
Multipurpose Density Matrix Embedding theory (mp-DMET)
Hung Q. Pham
email: pqh3.14@gmail.com
'''

import numpy as np
import scipy as scipy
from functools import reduce
from mpdmet.lib.build import libdmet

class RHF_decomposition:
	def __init__(self, cell, impOrbs, numBathOrbs, locOED_Ls, method = 'OED'):	
		self.cell = cell
		self.impOrbs = impOrbs
		self.method = method
		self.numBathOrbs = numBathOrbs		
		self.locMO_Ls, self.locOED_Ls = locOED_Ls
		
	def baths(self):
		'''
		This function is used to call the Schmidt basis using either an overlap matrix or 1-rdm (or OED)
		'''                  
		if self.method == 'OED':  
			return self.UsingOED(self.numBathOrbs, threshold = 1e-13)
		elif self.method == 'overlap':
			return self.UsingOverlap(self.numBathOrbs, threshold = 1e-7)
			
	def UsingOverlap(self, numBathOrbs, threshold = 1e-7):
		'''
		Construct the RHF bath using a projector
		ref: PHYSICAL REVIEW B 89, 035140 (2014)
		'''

		# Build the projectors for fragment and bath
		nao = self.cell.nao_nr()
		P_F = np.zeros((nao,nao))
		P_F[self.impOrbs == 1,self.impOrbs == 1] = 1
		P_B = np.identity(nao)- P_F		
		
		
		# Build the overlap matrix between hole states and fragment orbs
		nelec_pairs = self.cell.nelectron // 2 	
		Occ = self.orthoMO_Ls[:,:nelec_pairs]
		M = reduce(np.dot,(Occ.T, P_F,Occ))
		d, V = np.linalg.eigh(M) 				# 0 <= d <= 1
		idx = (-d).argsort() 					#d close to 1 come first
		d, V = d[idx], V[:, idx] 
		tokeep = np.sum(d > threshold)
		if tokeep < numBathOrbs:
			print ("BATH CONSTRUCTION: using only ", tokeep, " orbitals which are within ", threshold, " of 0 or 1")
		numBathOrbs = min(tokeep, numBathOrbs) #TODO: throw away some bath orbitals scheme for this construction????
		
		Forbs = np.einsum('pi,up->ui',V[:,:tokeep], np.dot(P_F, Occ))/np.sqrt(d[:numBathOrbs], optimize=True)
		Borbs = np.einsum('pi,up->ui',V[:,:tokeep], np.dot(P_B, Occ))/np.sqrt(1 - d[:numBathOrbs], optimize=True)
		
		pureEnorbs = np.einsum('pi,up->ui', V[:,tokeep:], Occ, optimize=True)
		entorbs = np.einsum('pi,up->ui', V[:,:tokeep], Occ, optimize=True)		
		FBEorbs = np.hstack((Forbs, Borbs, pureEnorbs))
			
		return (numBathOrbs, FBEorbs, entorbs)
			
	def UsingOED( self, numBathOrbs, threshold=1e-13 ):
		'''
		Construct the RHF bath using one-electron density matrix (OED)
		This function is a modified version of qcdmethelper/constructbath funtion 
		in the QC-DMET <Copyright (C) 2015 Sebastian Wouters>
		ref: 
			J. Chem. Theory Comput. 2016, 12, 2706−2719
		'''	
		
		OneDM = self.locOED_Ls	
		impurityOrbs = np.asarray(self.impOrbs)
		embeddingOrbs = 1 - impurityOrbs
		if (embeddingOrbs.shape[0] > 1):
			embeddingOrbs = embeddingOrbs.T		
		embeddingOrbs = np.matrix(embeddingOrbs) 				#Converse embeddingOrbs to a matrix (1, x)
		isEmbedding = np.dot(embeddingOrbs.T , embeddingOrbs) == 1
		numEmbedOrbs = np.sum(embeddingOrbs, dtype = np.int64)
		embedding1RDM = np.reshape(OneDM[isEmbedding], (numEmbedOrbs, numEmbedOrbs))		
		numImpOrbs   = np.sum(impurityOrbs, dtype = np.int64)
		numTotalOrbs = len(impurityOrbs)
		eigenvals, eigenvecs = np.linalg.eigh(embedding1RDM, UPLO='U')  	# 0 <= eigenvals <= 2		
		idx = np.maximum(-eigenvals, eigenvals - 2.0).argsort() # Occupation numbers closest to 1 come first
		
		#TODO: whether the truncation should be used
		tokeep = np.sum(-np.maximum(-eigenvals, eigenvals - 2.0)[idx] > threshold)
		if tokeep < numBathOrbs:
			print ("DMET::constructbath : Throwing out", numBathOrbs - tokeep, "orbitals which are within", threshold, "of 0 or 2.")
		numBathOrbs = min(np.sum(tokeep), numBathOrbs)
		
		eigenvals = eigenvals[idx]
		eigenvecs = eigenvecs[:,idx]
		pureEnvals = eigenvals[numBathOrbs:]
		pureEnorbs = eigenvecs[:,numBathOrbs:]
		idx = (-pureEnvals).argsort()
		eigenvecs[:,numBathOrbs:] = pureEnorbs[:,idx]
		pureEnvals = pureEnvals[idx]
		coreOccupations = np.hstack((np.zeros([numImpOrbs + numBathOrbs]), pureEnvals)) #Use to calculate the 1e rdm of core orbitals

		# Reconstruct the fragment orbitals so that the density matrix has a trivial form:
		embeddingOrbs_frag = np.matrix(impurityOrbs) 				#Converse embeddingOrbs to a matrix (1, x)
		isEmbedding_frag = np.dot(embeddingOrbs_frag.T , embeddingOrbs_frag) == 1
		numEmbedOrbs_frag = np.sum(embeddingOrbs_frag, dtype = np.int32)
		embedding1RDM_frag = np.reshape(OneDM[isEmbedding_frag], (numEmbedOrbs_frag, numEmbedOrbs_frag))
		eigenvals_frag, eigenvecs_frag = np.linalg.eigh(embedding1RDM_frag)  	# 0 <= eigenvals <= 2

		#Debug: rotate the fragment orbitals among themselves
		if False: eigenvecs_frag = np.eye(eigenvecs_frag.shape[0],eigenvecs_frag.shape[0])
		
		#Fragment orbitals: stack columns with zeros in the end
		#Embedding orbitals: stack columns with zeros in the beginning	
		eigenvecs_frag = np.hstack((eigenvecs_frag, np.zeros((numImpOrbs, numEmbedOrbs)))) 
		eigenvecs = np.hstack((np.zeros((numEmbedOrbs, numImpOrbs)), eigenvecs))
		row = 0
		for ao in range(0, numTotalOrbs):
			if impurityOrbs[ao]:
				eigenvecs = np.insert(eigenvecs, ao, eigenvecs_frag[row], axis=0)
				row += 1
    
        # Orthonormality is guaranteed due to (1) stacking with zeros and (2) orthonormality eigenvecs for symmetric matrix
		assert(np.linalg.norm(np.dot(eigenvecs.T, eigenvecs) - np.identity(numTotalOrbs)) < 1e-12 )
	
		return (numBathOrbs, eigenvecs, coreOccupations)

		
