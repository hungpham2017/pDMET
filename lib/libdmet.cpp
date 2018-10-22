/*
Useful library for DMET
Author: Hung Q. Pham
email: phamx494@umn.edu
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <fstream>
#include <iostream>
#include <lawrap/blas.h>
#include <lawrap/lapack.h>


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
//For complex 1-RDM
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
	std::vector<std::complex<double>> work1(size);	
	std::vector<std::complex<double>> work2(Norb*numPairs);
	
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

py::array_t<std::complex<double>>  iFTWFs(int num_kpts, py::array_t<double> kpts, int num_Ls, py::array_t<double> Ls, 
										py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> input)
{
	//inverse FT for one-electron operator: from k-space to L-space
	
	py::buffer_info kpts_info = kpts.request();
	const double * kpts_data = static_cast<double*>(kpts_info.ptr);
	py::buffer_info Ls_info = Ls.request();
	const double * Ls_data = static_cast<double*>(Ls_info.ptr);
	py::buffer_info input_info = input.request();
	const std::complex<double> * input_data = static_cast<std::complex<double>*>(input_info.ptr);
	int size1 = input_info.shape[1];
	int size2 = input_info.shape[2];	
	if (input_info.shape[0] != num_kpts)
		throw std::runtime_error("input.shape[0] is not equal to the num_Ls");
	
	int sizerow = num_Ls*size1;
	int sizecol = num_Ls*size2;
	double normalized = 1.0/num_kpts;	
	std::vector<std::complex<double>> output(sizerow*sizecol,0);
	
	for (int i = 0; i < num_Ls; i++){
		for (int j = 0; j < num_Ls; j++){		
			for (int k = 0; k < num_kpts; k++){	
				for (int row = 0; row < size1; row++){			
					for (int col = 0; col < size2; col++){
						std::vector<double> kpts(kpts_data + k*3, kpts_data + k*3 + 3);
						std::vector<double> Ls1(Ls_data + i*3, Ls_data + i*3 + 3);
						std::vector<double> Ls2(Ls_data + j*3, Ls_data + j*3 + 3);						
						LAWrap::axpy(3, -1.0, Ls1.data(), 1, Ls2.data(), 1);
						double dot = LAWrap::dotc(3, kpts.data(), 1, Ls2.data(), 1);
						int indexrow = j*size1 + row;
						int indexcol = i*size2 + col;						
						output[indexrow*sizecol + indexcol] += normalized * std::exp(-consts::Onecomp*dot) * input_data[k*size1*size2 + row*size2 + col];
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

py::array_t<std::complex<double>>  iFTWFs_sparse(int num_kpts, py::array_t<double> kpt, int num_Ls, py::array_t<double> Ls, py::array_t<std::complex<double>> input)
{
	//inverse FT for one-electron operator: from k-space to L-space
	
	py::buffer_info kpt_info = kpt.request();
	const double * kpt_data = static_cast<double*>(kpt_info.ptr);
	py::buffer_info Ls_info = Ls.request();
	const double * Ls_data = static_cast<double*>(Ls_info.ptr);
	py::buffer_info input_info = input.request();
	const std::complex<double> * input_data = static_cast<std::complex<double>*>(input_info.ptr);
	int size1 = input_info.shape[0];
	int size2 = input_info.shape[1];	
	int sizerow = num_Ls*size1;
	int sizecol = num_Ls*size2;
	double normalized = 1.0/num_kpts;	
	std::vector<std::complex<double>> output(sizerow*sizecol,0);
	
	
	for (int i = 0; i < num_Ls; i++){
		for (int j = 0; j < num_Ls; j++){		
			for (int row = 0; row < size1; row++){			
				for (int col = 0; col < size2; col++){
					std::vector<double> Ls1(Ls_data + i*3, Ls_data + i*3 + 3);
					std::vector<double> Ls2(Ls_data + j*3, Ls_data + j*3 + 3);						
					LAWrap::axpy(3, -1.0, Ls1.data(), 1, Ls2.data(), 1);
					double dot = LAWrap::dotc(3, kpt_data, 1, Ls2.data(), 1);
					int indexrow = j*size1 + row;
					int indexcol = i*size2 + col;						
					output[indexrow*sizecol + indexcol] += normalized * std::exp(-consts::Onecomp*dot) * input_data[row*size2 + col];
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

PYBIND11_PLUGIN(libdmet)
{
	py::module m("libdmet", "DMET library");
	m.def("rhf_response", &rhf_response);
	m.def("rhf_response_c", &rhf_response_c);
	m.def("iFTWFs", &iFTWFs);	
	m.def("iFTWFs_sparse", &iFTWFs_sparse);	
	return m.ptr();
	
}
