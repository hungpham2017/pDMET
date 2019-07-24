/*
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
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <fstream>
#include <iostream>
#include <lawrap/blas.h>
#include <lawrap/lapack.h>
#include <omp.h>


namespace py = pybind11; 
namespace consts
{
	const double Pi = 3.141592653589793;
	const std::complex<double> Onecomp(0.0, 1.0);
	const std::complex<double> Zerocomp(0.0, 0.0);	
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Analytical 1RDM derivative for Hamiltonian: H = H0 + H1, ref: J. Chem. Theory Comput. 2016, 12, 2706−2719
////////////////////////////////////////////////////////////////////////////////////////////////////////////

py::array_t<double> rhf_response(const int Norb, const int Nterms, const int numPairs, py::array_t<int> inH1start, 
			py::array_t<int> inH1row, py::array_t<int> inH1col, py::array_t<double> inH0)
{
	py::buffer_info H1start_info = inH1start.request();
	py::buffer_info H1row_info = inH1row.request();
	py::buffer_info H1col_info = inH1col.request();
	py::buffer_info H0_info = inH0.request();	
	
	if (H0_info.shape[0] != Norb or H0_info.shape[1] != Norb)
		throw std::runtime_error("H0 size does not match with the number of basis functions");

	const int * H1start_data = static_cast<int*>(H1start_info.ptr);
	const int * H1row_data = static_cast<int*>(H1row_info.ptr);
	const int * H1col_data = static_cast<int*>(H1col_info.ptr);	
	const double * H0_data = static_cast<double*>(H0_info.ptr);

	std::vector<int> H1start(H1start_data, H1start_data + H1start_info.size);
	std::vector<int> H1row(H1row_data, H1row_data + H1row_info.size);
	std::vector<int> H1col(H1col_data, H1col_data + H1col_info.size);
	std::vector<double> H0(H0_data, H0_data + H0_info.size);	
	std::vector<double> rdm_deriv(Nterms*Norb*Norb, 0);

    const int size = Norb * Norb;
    const int nVir = Norb - numPairs;	
	
	std::vector<double> eigvecs(size);
	std::vector<double> eigvals(Norb);		
	
    //eigvecs, eigvals: eigenvectors and eigenvalues of H0
	int inc = 1;
	LAWrap::copy(size, H0.data(), inc, eigvecs.data(), inc); 
	LAWrap::heev('V', 'U',  Norb, eigvecs.data(), Norb, eigvals.data());
	
    //Calculating 1RDM: H0 = 2 * OCC * OCC.T
	LAWrap::gemm('N', 'T', Norb, Norb, numPairs, 2.0, eigvecs.data(), Norb, eigvecs.data(), Norb, 0.0, H0.data(), Norb);

	//Get the unocc orbitals, for the occ orbitals we can just use eigvals with the appropriate indices
	std::vector<double>::const_iterator first = eigvecs.begin() + Norb*numPairs;
	std::vector<double>::const_iterator last = eigvecs.begin() + size;
	std::vector<double> virt(first, last);

    // temp[ vir + nVir * occ ] = - 1 / ( eps_vir - eps_occ )
	std::vector<double> temp(nVir*numPairs);	
	std::vector<double> work1(size);	
	std::vector<double> work2(Norb*numPairs);
	
    for ( int deriv = 0; deriv < Nterms; deriv++ ){
        // work1 = - VIRT.T * H1 * OCC / ( eps_vir - eps_occ ), work1 here is Z1 in the equation (44), JCTC 2016, 12, 2706
        for ( int orb_vir = 0; orb_vir < nVir; orb_vir++ ){
            for ( int orb_occ = 0; orb_occ < numPairs; orb_occ++ ){
                double value = 0.0;
                for ( int elem = H1start[deriv]; elem < H1start[deriv + 1]; elem++ ){
                    value += virt[Norb*orb_vir + H1row[elem]] * eigvecs[Norb*orb_occ + H1col[elem]];
                }
                work1[nVir*orb_occ + orb_vir] = - 1.0 / (eigvals[numPairs + orb_vir] - eigvals[orb_occ]) * value; //value * temp[orb_vir + nVir*orb_occ];
            }
        }
		
	
        // work1 = 2 * VIRT * work1 * OCC.T
                char notr = 'N';
                double alpha = 2.0;
                double beta = 0.0;
		// work2 = 2 * VIRT * work1		
		LAWrap::gemm('N', 'N', Norb, numPairs, nVir, 2.0, virt.data(), Norb, work1.data(), nVir, 0.0, work2.data(), Norb);		
		
		// work1 = work2 * OCC.T, work1 here is Cvir*Z1*Cocc.T in the equation (45), JCTC 2016, 12, 2706		
		LAWrap::gemm('N', 'T', Norb, Norb, numPairs, 1.0, work2.data(), Norb, eigvecs.data(), Norb, 0.0, work1.data(), Norb);		

        // rdm_deriv[ row + Norb * ( col + Norb * deriv ) ] = work1 + work1.T
        for ( int row = 0; row < Norb; row++ ){
            for ( int col = 0; col < Norb; col++ ){
                rdm_deriv[size*deriv + Norb*row + col] = work1[row + Norb*col] + work1[col + Norb*row];
				}
        }
    }	

	size_t pyNterms = Nterms;
	size_t pyNorb = Norb;
	size_t pyNorb2 = size;
	py::buffer_info rdm_deriv_buf =
		{
			rdm_deriv.data(),
			sizeof(double),
			py::format_descriptor<double>::format(),
			3,
			{pyNterms, pyNorb, pyNorb},
			{pyNorb2 * sizeof(double), pyNorb * sizeof(double), sizeof(double)}
		};
		
	return py::array_t<double>(rdm_deriv_buf);

}	

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Analytical 1RDM derivative for Hamiltonian: H = H0 + H1, ref: J. Chem. Theory Comput. 2016, 12, 2706−2719
//For complex Hamiltonian
////////////////////////////////////////////////////////////////////////////////////////////////////////////

 py::array_t<std::complex<double>> rhf_response_c(const int Norb, const int Nterms, const int numPairs, py::array_t<int> inH1start, 
			py::array_t<int> inH1row, py::array_t<int> inH1col, py::array_t<std::complex<double>> inH0)
{
	py::buffer_info H1start_info = inH1start.request();
	py::buffer_info H1row_info = inH1row.request();
	py::buffer_info H1col_info = inH1col.request();
	py::buffer_info H0_info = inH0.request();	
	
	if (H0_info.shape[0] != Norb or H0_info.shape[1] != Norb)
		throw std::runtime_error("H0 size does not match with the number of basis functions");

	const int * H1start_data = static_cast<int*>(H1start_info.ptr);
	const int * H1row_data = static_cast<int*>(H1row_info.ptr);
	const int * H1col_data = static_cast<int*>(H1col_info.ptr);	
	const std::complex<double> * H0_data = static_cast<std::complex<double>*>(H0_info.ptr);

	std::vector<int> H1start(H1start_data, H1start_data + H1start_info.size);
	std::vector<int> H1row(H1row_data, H1row_data + H1row_info.size);
	std::vector<int> H1col(H1col_data, H1col_data + H1col_info.size);
	std::vector<std::complex<double>> H0(H0_data, H0_data + H0_info.size);	
	std::vector<std::complex<double>> rdm_deriv(Nterms*Norb*Norb, 0);

    const int size = Norb * Norb;
    const int nVir = Norb - numPairs;	
	
	std::vector<std::complex<double>> eigvecs(size);
	std::vector<double> eigvals(Norb);		
	
    //eigvecs, eigvals: eigenvectors and eigenvalues of H0
	int inc = 1;
	LAWrap::copy(size, H0.data(), inc, eigvecs.data(), inc); 
	LAWrap::heev('V', 'U',  Norb, eigvecs.data(), Norb, eigvals.data());
	
    //Calculating 1RDM: H0 = 2 * OCC * OCC.T
	LAWrap::gemm('N', 'C', Norb, Norb, numPairs, 2.0, eigvecs.data(), Norb, eigvecs.data(), Norb, 0.0, H0.data(), Norb);

	//Get the unocc orbitals, for the occ orbitals we can just use eigvals with the appropriate indices
	std::vector<std::complex<double>>::const_iterator first = eigvecs.begin() + Norb*numPairs;
	std::vector<std::complex<double>>::const_iterator last = eigvecs.begin() + size;
	std::vector<std::complex<double>> virt(first, last);

#pragma omp parallel default(none) shared(H1start,H1row,H1col,virt,eigvals,eigvecs,rdm_deriv)	
{        
	std::vector<std::complex<double>> work1(size);	
	std::vector<std::complex<double>> work2(Norb*numPairs);
    
#pragma omp for schedule(dynamic)	
    for ( int deriv = 0; deriv < Nterms; deriv++ ){
        // work1 = - VIRT.T * H1 * OCC / ( eps_vir - eps_occ ), work1 here is Z1 in the equation (44), JCTC 2016, 12, 2706
        for ( int orb_vir = 0; orb_vir < nVir; orb_vir++ ){
            for ( int orb_occ = 0; orb_occ < numPairs; orb_occ++ ){
                std::complex<double> value = 0.0;
                for ( int elem = H1start[deriv]; elem < H1start[deriv + 1]; elem++ ){
                    value += std::conj(virt[Norb*orb_vir + H1row[elem]]) * eigvecs[Norb*orb_occ + H1col[elem]];
                }
                work1[nVir*orb_occ + orb_vir] = - 1.0 / (eigvals[numPairs + orb_vir] - eigvals[orb_occ]) * value;
            }
        }
		
		// work2 = 2 * VIRT * work1		
		LAWrap::gemm('N', 'N', Norb, numPairs, nVir, 2.0, virt.data(), Norb, work1.data(), nVir, 0.0, work2.data(), Norb);		
		
		// work1 = work2 * OCC.T, work1 here is Cvir*Z1*Cocc.T in the equation (45), JCTC 2016, 12, 2706		
		LAWrap::gemm('N', 'C', Norb, Norb, numPairs, 1.0, work2.data(), Norb, eigvecs.data(), Norb, 0.0, work1.data(), Norb);		

        // rdm_deriv[ row + Norb * ( col + Norb * deriv ) ] = work1 + work1.T
        for ( int row = 0; row < Norb; row++ ){
            for ( int col = 0; col < Norb; col++ ){
                rdm_deriv[size*deriv + Norb*row + col] = std::conj(work1[row + Norb*col]) + work1[col + Norb*row];
				}
        }
    }	
}
	size_t pyNterms = Nterms;
	size_t pyNorb = Norb;
	size_t pyNorb2 = size;
	py::buffer_info rdm_deriv_buf =
		{
			rdm_deriv.data(),
			sizeof(std::complex<double>),
			py::format_descriptor<std::complex<double>>::format(),
			3,
			{pyNterms, pyNorb, pyNorb},
			{pyNorb2 * sizeof(std::complex<double>), pyNorb * sizeof(std::complex<double>), sizeof(std::complex<double>)}
		};
		
	return py::array_t<std::complex<double>>(rdm_deriv_buf);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Forward and inverse Fourier Transform (FFT) for one and two electron integrals
////////////////////////////////////////////////////////////////////////////////////////////////////////////

py::array_t<std::complex<double>>  FT1e(int num_kpts, py::array_t<double> kpts, int num_Ls, py::array_t<double> Ls, 
										py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> ML)
{
	//inverse FT for one-electron matrix: from L-space to k-space
	// Notation:
    // i,j,k,l: k-point           ;  p,q,r,s: matrix indexes   
    // a,b,c,d: lattice indexes   ;  u,v,w,x: lattice indexes of the output
	
	py::buffer_info kpts_info = kpts.request();
	const double * kpts_data = static_cast<double*>(kpts_info.ptr);
	py::buffer_info Ls_info = Ls.request();
	const double * Ls_data = static_cast<double*>(Ls_info.ptr);
	py::buffer_info ML_info = ML.request();
	const std::complex<double> * ML_data = static_cast<std::complex<double>*>(ML_info.ptr);
	int MLrow = ML_info.shape[0];	  
	if (MLrow != ML_info.shape[1])
		throw std::runtime_error("M must be a square matrix");
	
	int nao = (int)MLrow / num_Ls;
	int nao2 = nao * nao;    
	int size = num_Ls*nao;    
	std::vector<std::complex<double>> output(num_kpts*nao2,0);

#pragma omp parallel default(none) shared(num_Ls,num_kpts,nao,nao2,size,kpts_data,Ls_data,ML_data,output)	
{        
	double weight = 1.0/num_Ls;
//printf("Debug: hello from HP %d of %d HP\n", omp_get_thread_num() + 1, omp_get_num_threads());
#pragma omp for schedule(static) collapse(5)
	for (int k = 0; k < num_kpts; k++){	        
        for (int a = 0; a < num_Ls; a++){
        for (int b = 0; b < num_Ls; b++){		
            for (int p = 0; p < nao; p++){			
            for (int q = 0; q < nao; q++){
						std::vector<double> kpts(kpts_data + k*3, kpts_data + k*3 + 3);
						std::vector<double> Ls1(Ls_data + a*3, Ls_data + a*3 + 3);
						std::vector<double> Ls2(Ls_data + b*3, Ls_data + b*3 + 3);						
						LAWrap::axpy(3, -1.0, Ls1.data(), 1, Ls2.data(), 1);
						double dot = LAWrap::dotc(3, kpts.data(), 1, Ls2.data(), 1);
						int u = b*nao + p;
						int v = a*nao + q;						
						output[k*nao2 + p*nao + q] +=  weight * std::exp(consts::Onecomp*dot) * ML_data[u*size + v];
            }
            }				
        }
		}
	}
}

	size_t nkpts = num_kpts;
	size_t pnao = nao;
	py::buffer_info output_buf =
		{
			output.data(),
			sizeof(std::complex<double>),
			py::format_descriptor<std::complex<double>>::format(),
			3,
			{nkpts,pnao, pnao},
			{pnao*pnao*sizeof(std::complex<double>), pnao*sizeof(std::complex<double>), sizeof(std::complex<double>)}
		};
		
	return py::array_t<std::complex<double>> (output_buf);
	
}


///--------------------------------------------------------------------------------------------------------------------
py::array_t<std::complex<double>>  iFT1e(int num_kpts, py::array_t<double> kpts, int num_Ls, py::array_t<double> Ls, 
										py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> Mk)
{
	//inverse FT for one-electron matrix: from k-space to L-space
	// Notation:
    // i,j,k,l: k-point           ;  p,q,r,s: matrix indexes   
    // a,b,c,d: lattice indexes   ;  u,v,w,x: lattice indexes of the output
	
	py::buffer_info kpts_info = kpts.request();
	const double * kpts_data = static_cast<double*>(kpts_info.ptr);
	py::buffer_info Ls_info = Ls.request();
	const double * Ls_data = static_cast<double*>(Ls_info.ptr);
	py::buffer_info Mk_info = Mk.request();
	const std::complex<double> * Mk_data = static_cast<std::complex<double>*>(Mk_info.ptr);
	int row = Mk_info.shape[1];	
	int col = Mk_info.shape[2];	    
	int rowcol = row*col;	    
	if (Mk_info.shape[0] != num_kpts)
		throw std::runtime_error("Mk.shape[0] is not equal to the num_Ls");
	
	int osize_row = num_Ls*row;
	int osize_col = num_Ls*col;    
	std::vector<std::complex<double>> output(osize_row*osize_col,0);

#pragma omp parallel default(none) shared(num_Ls,num_kpts,row,col,rowcol,osize_row,osize_col,kpts_data,Ls_data,Mk_data,output)	
{        
	double weight = 1.0/num_kpts;
//printf("Debug: hello from HP %d of %d HP\n", omp_get_thread_num() + 1, omp_get_num_threads());
#pragma omp for schedule(static) collapse(5)
	for (int a = 0; a < num_Ls; a++){
    for (int b = 0; b < num_Ls; b++){		
        for (int k = 0; k < num_kpts; k++){	
            for (int p = 0; p < row; p++){			
            for (int q = 0; q < col; q++){
						std::vector<double> kpts(kpts_data + k*3, kpts_data + k*3 + 3);
						std::vector<double> Ls1(Ls_data + a*3, Ls_data + a*3 + 3);
						std::vector<double> Ls2(Ls_data + b*3, Ls_data + b*3 + 3);						
						LAWrap::axpy(3, -1.0, Ls1.data(), 1, Ls2.data(), 1);
						double dot = LAWrap::dotc(3, kpts.data(), 1, Ls2.data(), 1);
						int u = a*row + p;
						int v = b*col + q;						
						output[u*osize_col + v] += weight * std::exp(consts::Onecomp*dot) * Mk_data[k*rowcol + p*col + q];
            }
            }				
        }
    }
	}
}

	size_t prow = osize_row;
	size_t pcol = osize_col;    
	py::buffer_info output_buf =
		{
			output.data(),
			sizeof(std::complex<double>),
			py::format_descriptor<std::complex<double>>::format(),
			2,
			{prow, pcol},
			{pcol*sizeof(std::complex<double>), sizeof(std::complex<double>)}
		};
		
	return py::array_t<std::complex<double>> (output_buf);
	
}


///--------------------------------------------------------------------------------------------------------------------
py::array_t<std::complex<double>>  iFFT1e(py::array_t<int> tmap,
                                        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> phase,
										py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> Mk)
{
	//inverse FT for one-electron matrix: from k-space to L-space
	// Notation:
    // i,j,k,l: k-point           ;  p,q,r,s: matrix indexes   
    // a,b,c,d: lattice indexes   ;  u,v,w,x: lattice indexes of the output

	py::buffer_info tmap_info = tmap.request();
	const int * tmap_data = static_cast<int*>(tmap_info.ptr);    
	py::buffer_info phase_info = phase.request();
	const std::complex<double> * phase_data = static_cast<std::complex<double>*>(phase_info.ptr);	
	py::buffer_info Mk_info = Mk.request();
	const std::complex<double> * Mk_data = static_cast<std::complex<double>*>(Mk_info.ptr);
	int nkpts = Mk_info.shape[0];
	int nLs = nkpts;    
	int row = Mk_info.shape[1];	
	int col = Mk_info.shape[2];	    
	int rowcol = row*col;	    
	
	int size_row = nLs*row;
	int size_col = nLs*col;    
	std::vector<std::complex<double>> output(size_row*size_col,0);
    int a0 = nLs/2;
    
// For L0
#pragma omp parallel default(none) shared(a0,nLs,nkpts,row,col,rowcol,size_row,size_col,phase_data,Mk_data,output)	
{        
	double weight = 1.0/nkpts;
#pragma omp for schedule(static) collapse(3)
    for (int b = 0; b < a0+1; b++){		
        for (int p = 0; p < row; p++){                
        for (int q = 0; q < col; q++){
                int u1 = a0*row + p;
                int v2 = (nLs-1-b)*col + p;	
                int v1 = b*col + q;	
                int u2 = a0*row + q;                  
                std::complex<double> temp = (0,0);
                for (int k = 0; k < nkpts; k++){	
                    temp += weight * phase_data[b*nkpts + k] * Mk_data[k*rowcol + p*col + q];
                }
                output[u1*size_col + v1] = temp;
                output[u2*size_col + v2] = std::conj(temp);
        }				
        }
    }
}

// For other Ls
#pragma omp parallel default(none) shared(a0,nLs,nkpts,row,col,rowcol,size_row,size_col,tmap_data,output)		
{        
#pragma omp for schedule(static) collapse(4)
	for (int a = 0; a < a0; a++){
    for (int b = 0; b < nLs; b++){
        for (int p = 0; p < row; p++){	
        for (int q = 0; q < col; q++){
            int a_ = nLs-1-a;
            int b1 = tmap_data[a*nLs+b];
            int b2 = tmap_data[a_*nLs+b];
            int u = a0*row + p;            
            int u1 = a*row + p;
            int u2 = a_*row + p;
            int v = b*col + q;	
            int v1 = b1*col + q;	
            int v2 = b2*col + q;	                    
            output[u1*size_col + v] = output[u*size_col + v1];
            output[u2*size_col + v] = output[u*size_col + v2];  
        }
        }				
    }
	}
}

	size_t drow = size_row;
	size_t dcol = size_col;    
	py::buffer_info output_buf =
		{
			output.data(),
			sizeof(std::complex<double>),
			py::format_descriptor<std::complex<double>>::format(),
			2,
			{drow, dcol},
			{dcol*sizeof(std::complex<double>), sizeof(std::complex<double>)}
		};
		
	return py::array_t<std::complex<double>> (output_buf);
	
}


///--------------------------------------------------------------------------------------------------------------------
py::array_t<std::complex<double>>  iFT1e_sparse(int num_kpts, py::array_t<double> kpt, int num_Ls, py::array_t<double> Ls, py::array_t<std::complex<double>> Mk)
{
	//inverse FT for one-electron matrix: from k-space to L-space
	
	py::buffer_info kpt_info = kpt.request();
	const double * kpt_data = static_cast<double*>(kpt_info.ptr);
	py::buffer_info Ls_info = Ls.request();
	const double * Ls_data = static_cast<double*>(Ls_info.ptr);
	py::buffer_info Mk_info = Mk.request();
	const std::complex<double> * Mk_data = static_cast<std::complex<double>*>(Mk_info.ptr);
    
	int size1 = Mk_info.shape[0];
	int size2 = Mk_info.shape[1];	
	int sizerow = num_Ls*size1;
	int sizecol = num_Ls*size2;
	std::vector<std::complex<double>> output(sizerow*sizecol,0);
	
#pragma omp parallel default(none) shared(num_Ls,num_kpts,size1,size2,sizecol,kpt_data,Ls_data,Mk_data,output)	
{ 
	double weight = 1.0/num_kpts;

#pragma omp for schedule(static) collapse(4)	
	for (int a = 0; a < num_Ls; a++){
		for (int b = 0; b < num_Ls; b++){		
			for (int p = 0; p < size1; p++){			
				for (int q = 0; q < size2; q++){
					std::vector<double> Ls1(Ls_data + a*3, Ls_data + a*3 + 3);
					std::vector<double> Ls2(Ls_data + b*3, Ls_data + b*3 + 3);						
					LAWrap::axpy(3, -1.0, Ls1.data(), 1, Ls2.data(), 1);
					double dot = LAWrap::dotc(3, kpt_data, 1, Ls2.data(), 1);
					int m = b*size1 + p;
					int n = a*size2 + q;						
					output[m*sizecol + n] += weight * std::exp(-consts::Onecomp*dot) * Mk_data[p*size2 + q];
				}
			}				
		}
	}
}
	size_t psizerow = sizerow;
	size_t psizecol = sizecol;
	py::buffer_info output_buf =
		{
			output.data(),
			sizeof(std::complex<double>),
			py::format_descriptor<std::complex<double>>::format(),
			2,
			{psizerow, psizecol},
			{psizecol*sizeof(std::complex<double>), sizeof(std::complex<double>)}
		};
		
	return py::array_t<std::complex<double>> (output_buf);
	
}

///--------------------------------------------------------------------------------------------------------------------
py::array_t<std::complex<double>>  iFT2e(int num_kpts, py::array_t<double> kpts, int num_Ls, py::array_t<double> Ls, 
										py::array_t<int, py::array::c_style | py::array::forcecast> kconserv,
										py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> Mijk)
{
	//inverse FT for two-electron matrix: from k-space to L-space
	// Notation:
    // i,j,k,l: k-point           ;  p,q,r,s: matrix indexes   
    // a,b,c,d: lattice indexes   ;  u,v,w,x: lattice indexes of the output
    
	py::buffer_info kpts_info = kpts.request();
	const double * kpts_data = static_cast<double*>(kpts_info.ptr);
	py::buffer_info Ls_info = Ls.request();
	const double * Ls_data = static_cast<double*>(Ls_info.ptr);
	py::buffer_info kconserv_info = kconserv.request();
	const int * kconserv_data = static_cast<int*>(kconserv_info.ptr);    
	py::buffer_info Mijk_info = Mijk.request();
	const std::complex<double> * Mijk_data = static_cast<std::complex<double>*>(Mijk_info.ptr);

	int nao = Mijk_info.shape[3];    
	int outsize = num_Ls*nao;
	std::vector<std::complex<double>> output(outsize*outsize*outsize*outsize,0);

    
#pragma omp parallel default(none) shared(num_Ls,num_kpts,nao,outsize,kpts_data,Ls_data,kconserv_data,Mijk_data,output)
{ 
	int nao2 = nao*nao;
	int nao3 = nao2*nao; 
	int nao4 = nao3*nao;     
	int nk_nao4 = num_kpts*nao4; 
	int nk2_nao4 = num_kpts*num_kpts*nao4; 
    int outsize2 = outsize*outsize;
    int outsize3 = outsize*outsize*outsize;      
    int num_kpts2 = num_kpts*num_kpts;
	double weight = 1.0/(num_kpts*num_kpts*num_kpts);

#pragma omp for schedule(static) collapse(11)	    
	for (int a = 0; a < num_Ls; a++){
    for (int b = 0; b < num_Ls; b++){	
    for (int c = 0; c < num_Ls; c++){
    for (int d = 0; d < num_Ls; d++){
        for (int i = 0; i < num_kpts; i++){	
        for (int j = 0; j < num_kpts; j++){	
        for (int k = 0; k < num_kpts; k++){	                          
            for (int p = 0; p < nao; p++){			
            for (int q = 0; q < nao; q++){
            for (int r = 0; r < nao; r++){			
            for (int s = 0; s < nao; s++){
                                    
                int l = kconserv_data[i*num_kpts2 + j*num_kpts + k];
                std::vector<double> ki(kpts_data + i*3, kpts_data + i*3 + 3);
                std::vector<double> kj(kpts_data + j*3, kpts_data + j*3 + 3);
                std::vector<double> kk(kpts_data + k*3, kpts_data + k*3 + 3);
                std::vector<double> kl(kpts_data + l*3, kpts_data + l*3 + 3);
                std::vector<double> Ls1(Ls_data + a*3, Ls_data + a*3 + 3);
                std::vector<double> Ls2(Ls_data + b*3, Ls_data + b*3 + 3);	
                std::vector<double> Ls3(Ls_data + c*3, Ls_data + c*3 + 3);
                std::vector<double> Ls4(Ls_data + d*3, Ls_data + d*3 + 3);	                                
                std::complex<double> expo4 = std::exp(-consts::Onecomp*LAWrap::dotc(3, ki.data(), 1, Ls1.data(), 1));
                std::complex<double> expo3 = std::exp(consts::Onecomp*LAWrap::dotc(3, kj.data(), 1, Ls2.data(), 1));
                std::complex<double> expo2 = std::exp(-consts::Onecomp*LAWrap::dotc(3, kk.data(), 1, Ls3.data(), 1));
                std::complex<double> expo1 = std::exp(consts::Onecomp*LAWrap::dotc(3, kl.data(), 1, Ls4.data(), 1));
                int u = a*nao + p;
                int v = b*nao + q;
                int w = c*nao + r;
                int x = d*nao + s;                             
                output[u*outsize3 + v*outsize2 + w*outsize + x] += weight * expo1 * expo2 * expo3 * expo4
                * Mijk_data[i*nk2_nao4 + j*nk_nao4 + k*nao4 + p*nao3 + q*nao2 + r*nao + s];
            }
            }
            }
            }
                                
        }
        }	
        }
    }            
    }
    }
	}
}

	size_t psize = outsize;
	py::buffer_info output_buf =
		{
			output.data(),
			sizeof(std::complex<double>),
			py::format_descriptor<std::complex<double>>::format(),
			4,
			{psize, psize, psize, psize},
			{psize*psize*psize*sizeof(std::complex<double>), psize*psize*sizeof(std::complex<double>), psize*sizeof(std::complex<double>), sizeof(std::complex<double>)}
		};
		
	return py::array_t<std::complex<double>> (output_buf);
	
}


///--------------------------------------------------------------------------------------------------------------------
py::array_t<std::complex<double>>  iFFT2e(py::array_t<int> tmap,
                                        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> phase,
										py::array_t<int, py::array::c_style | py::array::forcecast> kconserv,
										py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> Mijk)
{
	//inverse FT for two-electron matrix: from k-space to L-space
	// Notation:
    // i,j,k,l: k-point           ;  p,q,r,s: matrix indexes   
    // a,b,c,d: lattice indexes   ;  u,v,w,x: lattice indexes of the output
    
	py::buffer_info tmap_info = tmap.request();
	const int * tmap_data = static_cast<int*>(tmap_info.ptr);    
	py::buffer_info phase_info = phase.request();
	const std::complex<double> * phase_data = static_cast<std::complex<double>*>(phase_info.ptr);
	py::buffer_info kconserv_info = kconserv.request();
	const int * kconserv_data = static_cast<int*>(kconserv_info.ptr);    
	py::buffer_info Mijk_info = Mijk.request();
	const std::complex<double> * Mijk_data = static_cast<std::complex<double>*>(Mijk_info.ptr);

	int nkpts = Mijk_info.shape[0];
	int nLs = nkpts;        
	int nao = Mijk_info.shape[3];    
	int outsize = nLs*nao;
	std::vector<std::complex<double>> output(outsize*outsize*outsize*outsize,0);
    int a0 = nLs/2;
    
// For L0s    
#pragma omp parallel default(none) shared(a0,nLs,nkpts,nao,outsize,phase_data,kconserv_data,Mijk_data,output)
{ 
	int nao2 = nao*nao;
	int nao3 = nao2*nao; 
	int nao4 = nao3*nao;     
	int nk_nao4 = nkpts*nao4; 
	int nk2_nao4 = nkpts*nkpts*nao4; 
    int outsize2 = outsize*outsize;
    int outsize3 = outsize*outsize*outsize;      
    int nkpts2 = nkpts*nkpts;
	double weight = 1.0/(nkpts*nkpts*nkpts);

#pragma omp for schedule(static) collapse(7)	    
    for (int b = 0; b < nLs; b++){	
    for (int c = 0; c < nLs; c++){
    for (int d = 0; d < nLs; d++){    
        for (int p = 0; p < nao; p++){			
        for (int q = 0; q < nao; q++){
        for (int r = 0; r < nao; r++){			
        for (int s = 0; s < nao; s++){
            std::complex<double> temp = (0,0);
            for (int i = 0; i < nkpts; i++){	
            for (int j = 0; j < nkpts; j++){	
            for (int k = 0; k < nkpts; k++){	
                int l = kconserv_data[i*nkpts2 + j*nkpts + k];                               
                temp += weight * phase_data[b*nkpts + j] * std::conj(phase_data[c*nkpts + k]) * phase_data[d*nkpts + l]
                        * Mijk_data[i*nk2_nao4 + j*nk_nao4 + k*nao4 + p*nao3 + q*nao2 + r*nao + s];
           }
           }
           }
            int u = a0*nao + p;
            int v = b*nao + q;
            int w = c*nao + r;
            int x = d*nao + s;  
            output[u*outsize3 + v*outsize2 + w*outsize + x] = temp;
        }        
        }
        }	
        }
    }            
    }
    }
}

// For other Ls    
#pragma omp parallel default(none) shared(a0,nLs,nkpts,nao,outsize,tmap_data,output)
{ 
    int outsize2 = outsize*outsize;
    int outsize3 = outsize*outsize*outsize;      

#pragma omp for schedule(static) collapse(8)	 
    for (int a = 0; a < nLs; a++){   
    for (int b = 0; b < nLs; b++){	
    for (int c = 0; c < nLs; c++){
    for (int d = 0; d < nLs; d++){    
        for (int p = 0; p < nao; p++){			
        for (int q = 0; q < nao; q++){
        for (int r = 0; r < nao; r++){			
        for (int s = 0; s < nao; s++){
            int u = a*nao + p;
            int v = b*nao + q;
            int w = c*nao + r;
            int x = d*nao + s; 
            int b0 = tmap_data[a*nLs+b];     
            int c0 = tmap_data[a*nLs+c];     
            int d0 = tmap_data[a*nLs+d];
            int u0 = a0*nao + p;
            int v0 = b0*nao + q;
            int w0 = c0*nao + r;
            int x0 = d0*nao + s; 
            output[u*outsize3 + v*outsize2 + w*outsize + x] = output[u0*outsize3 + v0*outsize2 + w0*outsize + x0];
        }        
        }
        }	
        }
    }            
    }
    }
    }
}

	size_t psize = outsize;
	py::buffer_info output_buf =
		{
			output.data(),
			sizeof(std::complex<double>),
			py::format_descriptor<std::complex<double>>::format(),
			4,
			{psize, psize, psize, psize},
			{psize*psize*psize*sizeof(std::complex<double>), psize*psize*sizeof(std::complex<double>), psize*sizeof(std::complex<double>), sizeof(std::complex<double>)}
		};
		
	return py::array_t<std::complex<double>> (output_buf);
	
}


///--------------------------------------------------------------------------------------------------------------------
py::array_t<double>  get_RDM_global(py::array_t<int> tmap,
                                   int nLs,
								   py::array_t<double, py::array::c_style | py::array::forcecast> RDM0)
{
	//inverse FT for one-electron matrix: from k-space to L-space
	// Notation:
    // i,j,k,l: k-point           ;  p,q,r,s: matrix indexes   
    // a,b,c,d: lattice indexes   ;  u,v,w,x: lattice indexes of the output

	py::buffer_info tmap_info = tmap.request();
	const int * tmap_data = static_cast<int*>(tmap_info.ptr);    
	py::buffer_info RDM0_info = RDM0.request();
	const double * RDM0_data = static_cast<double*>(RDM0_info.ptr);
	int nao = RDM0_info.shape[0];	    
	int nao2 = nao*nao;	    
	int size = nLs*nao;   
	std::vector<double> output(size*size,0);
    int a0 = nLs/2;
    
// For L0s
#pragma omp parallel default(none) shared(a0,nao,size,RDM0_data,output)		
{        
#pragma omp for schedule(static) collapse(1)
    for (int p = 0; p < nao; p++){	
        int shift = a0*nao;
        for (int q = 0; q < size; q++){                    
            output[(p+shift)*size + q] = RDM0_data[p*size + q];
        }
    }				
}
    
#pragma omp parallel default(none) shared(a0,nLs,nao,nao2,size,tmap_data,RDM0_data,output)		
{        
#pragma omp for schedule(static) collapse(4)
	for (int a = 0; a < a0; a++){
    for (int b = 0; b < nLs; b++){
        for (int p = 0; p < nao; p++){	
        for (int q = 0; q < nao; q++){
            int a_ = nLs-1-a;
            int b1 = tmap_data[a*nLs+b];
            int b2 = tmap_data[a_*nLs+b];          
            int u1 = a*nao + p;
            int u2 = a_*nao + p;
            int v = b*nao + q;	
            int v1 = b1*nao + q;	
            int v2 = b2*nao + q;	                    
            output[u1*size + v] = RDM0_data[p*size + v1];
            output[u2*size + v] = RDM0_data[p*size + v2];  
        }
        }				
    }
	}
}

	size_t dsize = size;   
	py::buffer_info output_buf =
		{
			output.data(),
			sizeof(double),
			py::format_descriptor<double>::format(),
			2,
			{dsize, dsize},
			{dsize*sizeof(double), sizeof(double)}
		};
		
	return py::array_t<double> (output_buf);
	
}

PYBIND11_MODULE(libdmet,m)
{
	m.doc() = "DMET library"; // optional
	m.def("rhf_response", &rhf_response);
	m.def("rhf_response_c", &rhf_response_c);
	m.def("FT1e", &FT1e);	
	m.def("iFT1e", &iFT1e);	
    m.def("iFFT1e", &iFFT1e);	
	m.def("iFT1e_sparse", &iFT1e_sparse);
	m.def("iFT2e", &iFT2e);	
	m.def("iFFT2e", &iFFT2e);		
	m.def("get_RDM_global", &get_RDM_global);
}
