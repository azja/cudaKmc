#include "transparams.h"
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include "../../macros/errors.h"
#include "../../headers/kernels.cuh"
#include "../../headers/cpukernels.h"
#include "../../headers/utils.h"
#include "../../headers/siminput.cuh"
#include "../../headers/isinge.cuh"
#include "../../headers/algorithms.cuh"
#include "../../headers/simulation.cuh"
#include "../builder/buildlattice.h"

namespace tests {
namespace transition {

float h_energies[4] = { -0.12f, -0.125f, -0.125f, -0.07f };
float h_saddle[2] = { 0.0f, 0.0f };
int CUBE_SIZE = 16;
const int STEPS = 1000;

struct kernel1 {

	int max;

	kernel1(int m) :
			max(m) {
	}
	;

	__host__ __device__ int operator()(int4& x) {
		return (x.w >= 0 && x.w < max) ? 0 : -1;
	}
};

void test(int sample_size) {

	CUBE_SIZE = sample_size;

	int z = 8;
	int z_t = z;
	int n_v = CUBE_SIZE * CUBE_SIZE * CUBE_SIZE * 2;
	int atoms_n = 2;
	int N = CUBE_SIZE * CUBE_SIZE * CUBE_SIZE * 2;
	int3 dims = { CUBE_SIZE, CUBE_SIZE, CUBE_SIZE };
	float radius = 2.5f;
	int BLOCK_SIZE = 512;

	float4* h_sites = tests::lattice::cubicB2(CUBE_SIZE);
	float4* d_sites;
	CHECK_ERROR(cudaMalloc((void**)&d_sites,sizeof(float4) * N));
	CHECK_ERROR(
			cudaMemcpy(d_sites,h_sites, sizeof(float4) * N,cudaMemcpyHostToDevice));

	int4* d_neigbours;
	CHECK_ERROR(cudaMalloc((void**)&d_neigbours,sizeof(int4) * N * z));
	utils::CudaTimer timer;
	timer.start();
	findNeigboursXyz<<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE,
			BLOCK_SIZE * sizeof(float4)>>>(d_sites, d_neigbours,
			tests::lattice::b2cell1, tests::lattice::b2cell2,
			tests::lattice::b2cell3, dims, radius, z, 0, N);
	cudaDeviceSynchronize();
	timer.stop();

	std::cout << "Neigbours list generated in:" << timer << std::endl;
	timer.reset();

	//Check if neigbours are properly generated
	thrust::device_ptr<int4> yu = thrust::device_pointer_cast(d_neigbours);
	assert(
			thrust::transform_reduce(yu, yu + N *z,kernel1(N * z),0,thrust::plus<int>()) == 0);

	float* d_transitions;
	CHECK_ERROR(cudaMalloc((void**)&d_transitions,sizeof(float) * N * z));

	typedef SimulationDeviceInput<float, kmcenergy::ising::PairEnergy_I,
			algorithms::ConstantSaddleEnergy, algorithms::ExchangeEnergy> isingDeviceInput;

	isingDeviceInput input;

	float* d_energies;
	CHECK_ERROR(
			cudaMalloc((void**)&d_energies,sizeof(float) * atoms_n * atoms_n));
	CHECK_ERROR(
			cudaMemcpy(d_energies,h_energies, sizeof(float) * atoms_n * atoms_n,cudaMemcpyHostToDevice));

	float* d_saddle;
	CHECK_ERROR(cudaMalloc((void**)&d_saddle,sizeof(float) * atoms_n));
	CHECK_ERROR(
			cudaMemcpy(d_saddle,h_saddle, sizeof(float) * atoms_n,cudaMemcpyHostToDevice));

	int* d_vacancies;
	CHECK_ERROR(cudaMalloc((void**)&d_vacancies,sizeof(int) * N));

	input.N = N;
	input.atoms_n = atoms_n;
	input.n_v = n_v;
	input.z = z;
	input.z_t = z_t;


	thrust::device_ptr<int> ptr = thrust::device_pointer_cast(d_vacancies);
	thrust::sequence(ptr, ptr + N);

	input.neigbours = d_neigbours;
	input.sites = d_sites;
	input.vacancies = d_vacancies;
	input.transitions = d_transitions;
	input.jumpingNeigbours = d_neigbours;

	input.exchangeEnergyCalculation = d_energies;
	input.saddleEnergyCalculation = d_saddle;

	thrust::device_ptr<float> ptr1 = thrust::device_pointer_cast(d_transitions);
	float result = 0;

	timer.start();
	for (int i = 0; i < STEPS; ++i) {
		calculateTransitionParameter<isingDeviceInput> <<<
				(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(input);
		cudaDeviceSynchronize();
		//CHECK_ERROR(cudaGetLastError());
		result = thrust::reduce(ptr1, ptr1 + N * z) / (N * z);
	}
	timer.stop();

	//std::cout<<"Avg Et = "<<result<< std::endl;
	std::cout << CUBE_SIZE << " " << timer << " " << result << std::endl;

	free(h_sites);
	cudaFree(d_sites);
	cudaFree(d_transitions);
	cudaFree(d_neigbours);
	cudaFree(d_energies);
	cudaFree(d_saddle);
	cudaFree(d_vacancies);
}

/////////////////////////////////////////////////////////////////////////////////

void testCpu(int sample_size) {

	CUBE_SIZE = sample_size;

	int z = 8;
	int z_t = z;
	int n_v = CUBE_SIZE * CUBE_SIZE * CUBE_SIZE * 2;
	int atoms_n = 2;
	int N = CUBE_SIZE * CUBE_SIZE * CUBE_SIZE * 2;
	int3 dims = { CUBE_SIZE, CUBE_SIZE, CUBE_SIZE };
	float radius = 2.5f;
	int BLOCK_SIZE = 256;

	float4* h_sites = tests::lattice::cubicB2(CUBE_SIZE);
	float4* d_sites;
	CHECK_ERROR(cudaMalloc((void**)&d_sites,sizeof(float4) * N));
	CHECK_ERROR(
			cudaMemcpy(d_sites,h_sites, sizeof(float4) * N,cudaMemcpyHostToDevice));

	int4* d_neigbours;
	CHECK_ERROR(cudaMalloc((void**)&d_neigbours,sizeof(int4) * N * z));

	findNeigboursXyz<<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE,
			BLOCK_SIZE * sizeof(float) * 3>>>(d_sites, d_neigbours,
			tests::lattice::b2cell1, tests::lattice::b2cell2,
			tests::lattice::b2cell3, dims, radius, z, 0, N);
	/*	    float3 base1 = { 2.86f, 0.0f, 0.0f };
	 float3 base2 = { 0.0f, 2.86f, 0.0f };
	 float3 base3 = { 0.0f, 0.0f, 2.86f };
	 catch (...) {

		 }



	 int3 dimensions = { 16, 16, 16 };
	 findNeigboursXyz<<<16, 512, 512 * sizeof(float) *3>>>(d_sites, d_neigbours, base1, base2, base3,
	 dimensions, 2.5, 8, 0, N);*/

	cudaDeviceSynchronize();

	int4* h_neigbours = (int4 *) malloc(sizeof(int4) * N * z);
	CHECK_ERROR(
			cudaMemcpy(h_neigbours,d_neigbours, sizeof(int4) * N * z,cudaMemcpyDeviceToHost));

	float* h_transitions = (float*) malloc(sizeof(float) * N * z);

	typedef SimulationDeviceInput<float, kmcenergy::ising::PairEnergy_I,
			algorithms::ConstantSaddleEnergy, algorithms::ExchangeEnergy> isingDeviceInput;

	isingDeviceInput input;

	int* h_vacancies = (int*) malloc(sizeof(int) * N);

	input.N = N;
	input.atoms_n = atoms_n;
	input.n_v = n_v;
	input.z = z;
	input.z_t = z_t;

	thrust::sequence(h_vacancies, h_vacancies + N);

	input.neigbours = h_neigbours;
	input.sites = h_sites;
	input.vacancies = h_vacancies;
	input.transitions = h_transitions;
	input.jumpingNeigbours = h_neigbours;

	input.exchangeEnergyCalculation = h_energies;
	input.saddleEnergyCalculation = h_saddle;

	float result = 0;

	utils::CudaTimer timer;
	timer.start();
	for (int i = 0; i < STEPS; ++i) {
		calculateTransitionParameterCpu<isingDeviceInput>(input);

		result = thrust::reduce(h_transitions, h_transitions + N * z - 1)
				/ (N * z);
	}
	timer.stop();

	//std::cout<<"Avg Et = "<<result<< std::endl;
	std::cout << CUBE_SIZE << " " << timer << " " << result << std::endl;

	free(h_sites);
	free(h_vacancies);
	free(h_neigbours);
	cudaFree(d_sites);
	cudaFree(d_neigbours);

}

}
}

