#include "energytest.h"
#include "../../headers/isinge.cuh"
#include <iostream>
#include <stdio.h>
#include "../../macros/errors.h"
#include "../../headers/kernels.cuh"
#include "../../headers/cpukernels.h"
#include "../../headers/utils.h"
#include "../../headers/isinge.cuh"
#include "../../headers/algorithms.cuh"
#include "energytest.h"
#include <thrust/reduce.h>
#include <thrust/scan.h>

template <typename T,typename U>
__global__ void calcEnergy(float4* sites, int4 *neigs, int z, T calcer,
		                   float* output, const U* params, int N, int nkind = 2){
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < N) {
		output[id] = (static_cast<int>(calcer(sites[id].w, sites, neigs, z, id, params, nkind)));

	}

}





template <typename T,typename U>
 void calcEnergyCpu(float4* sites, int4 *neigs, int z, T calcer, float* output,
		            const U* params, int N, int nkind = 2){
	for(int id =0; id < N;++id){
	  output[id]=(calcer(sites[id].w,sites,neigs,z,id,params,nkind));
	}

}

template <class T,class R, template  <class,class> class U>
__global__ void calcExchangeEnergy(float4* sites, int4 *neigs, int z, U<T,R> calcer,
		                   float* output, const T* params, int N, int nkind = 2){
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < N) {

		float result;
        result =calcer( id,neigs[id *z].w,
                 sites,
                 neigs,
                 z, params,
                 nkind);

		output[id] = result;
	}

}

template <class T,class R, template  <class,class> class U>
void calcExchangeEnergyCpu(float4* sites, int4 *neigs, int z, U<T,R> calcer,
		                   float* output, const T* params, int N, int nkind = 2){

	for(int id =0; id < N; ++id) {

		float result;
        result =calcer( id,neigs[id *z].w,
                 sites,
                 neigs,
                 z, params,
                 nkind);

		output[id] = result;
	}

}

#define NSITES 8192
#define NEIGS 8
#define DIM 16

static void b2Builder(float4* sites, float3 b1, float3 b2, float3 b3, float4 A,
		float4 B) {

	for (int i = 0; i < DIM; ++i) {
		for (int j = 0; j < DIM; ++j) {
			for (int k = 0; k < DIM; ++k) {

				int id = i * DIM * DIM + j * DIM + k;

				float4 sa = { (b1.x * A.x + b2.x * A.y + b3.x * A.z) + i * b1.x
						+ j * b2.x + k * b3.x, (b2.x * A.y + b2.y * A.y
								+ b2.z * A.y) + i * b1.y + j * b2.y + k * b3.y, (b3.x
										* A.z + b3.y * A.z + b3.z * A.z) + i * b1.z + j * b2.z
										+ k * b3.z };
				sa.w = 0.0f;

				float4 sb = { (b1.x * B.x + b2.x * B.y + b3.x * B.z) + i * b1.x
						+ j * b2.x + k * b3.x, (b2.x * B.y + b2.y * B.y
								+ b2.z * B.y) + i * b1.y + j * b2.y + k * b3.y, (b3.x
										* B.z + b3.y * B.z + b3.z * B.z) + i * b1.z + j * b2.z
										+ k * b3.z };

				sb.w = 1.0f;

				sites[id * 2] = sa;
				sites[id * 2 + 1] = sb;

			}
		}
	}
}



namespace tests {
namespace energy {
namespace ising {



void test() {

	float4* h_sites;
	int4* h_neigs;
	float* h_energies;

	int devCount;
	CHECK_ERROR(cudaGetDeviceCount(&devCount));

	utils::CudaTimer timer;
	utils::CpuTimer timer_cpu;
	utils::CpuTimer timer_cpu1;

	float energies[4];
	energies[0] = -0.12;
	energies[1] = -0.125;
	energies[2] = -0.125;
	energies[3] = -0.07;

	float3 base1 = { 2.86f, 0.0f, 0.0f };
	float3 base2 = { 0.0f, 2.86f, 0.0f };
	float3 base3 = { 0.0f, 0.0f, 2.86f };

	float4 siteA = { 0.0f, 0.0f, 0.0f, 1.0f };
	float4 siteB = { 0.5f, 0.5f, 0.5f, 2.0f };

	int3 dimensions = { DIM, DIM, DIM };

	float radius = 2.5f;

	h_sites = (float4*) malloc(sizeof(float4) * NSITES);
	h_neigs = (int4*) malloc(sizeof(int4) * NSITES * NEIGS);
	h_energies = (float*) malloc(sizeof(float) * NSITES );

	timer_cpu.start();
	b2Builder(h_sites, base1, base2, base3, siteA, siteB);
	timer_cpu.stop();

	std::cout << "Building lattice time:" << timer_cpu.elapsed() << " ms"
			<< std::endl;

	timer_cpu.reset();

	float4* d_sites;
	float* d_energies;
	int4* d_neigs;
	float* d_params;

	if (devCount > 0) {
		CHECK_ERROR(cudaMalloc((void**)&d_sites,sizeof(float4) * NSITES));
		CHECK_ERROR(
				cudaMalloc((void**)&d_neigs,sizeof(float4) * NSITES * NEIGS));

		CHECK_ERROR(
				cudaMemcpy(d_sites,h_sites,sizeof(float4)*NSITES,cudaMemcpyHostToDevice));

		CHECK_ERROR(cudaMalloc((void**)&d_energies,sizeof(float) * NSITES));



		CHECK_ERROR(cudaMalloc((void**)&d_params,sizeof(float) * 4));
		CHECK_ERROR(
						cudaMemcpy(d_params,energies,sizeof(float)*4,cudaMemcpyHostToDevice));

		//Neigbour - shared memory used
				std::cout<<"-------------------------------------------------------------------------"<<std::endl;
				std::cout<<"Looking for neigbours...";

		timer.start();
		findNeigboursXyz<<<16, 512, 512 * sizeof(float) *3>>>(d_sites, d_neigs, base1, base2, base3,
				dimensions, radius, 8, 0, NSITES);
		cudaDeviceSynchronize();
        timer.stop();

        std::cout<<"... found in "<<timer.elapsed()<<"ms (shared)"<<std::endl;

        timer.reset();
        /*
        std::cout<<"And once again - looking for neigbours...";

        timer.start();
        findNeigboursXyz<<<16, 512>>>(d_sites, d_neigs, base1, base2, base3,
        				dimensions, radius, 8, 0, NSITES);
        		cudaDeviceSynchronize();
                timer.stop();

                std::cout<<"... found in "<<timer.elapsed()<<"ms (shared not used)"<<std::endl;

                timer.reset();
        */

		CHECK_ERROR(
								cudaMemcpy(h_neigs,d_neigs,sizeof(float4)*NSITES * NEIGS,cudaMemcpyDeviceToHost));
		//Pair_Energy_I tests
		std::cout<<"-------------------------------------------------------------------------"<<std::endl;
		std::cout<<"Starting PairEnergy_I tests: GPU";
		//GPU TEST

		thrust::device_ptr<float> ptr = thrust::device_pointer_cast ( d_energies);
		float energy;
		timer.start();
        for(int i =0; i < 10000;++i){
		  for(int j=0;j<2;++j){
        	calcEnergy<kmcenergy::ising::PairEnergy_I,float><<<16,512>>>(d_sites,d_neigs,8,
				kmcenergy::ising::PairEnergy_I(),d_energies,d_params,NSITES);
		cudaDeviceSynchronize();}
	    energy = thrust::reduce(ptr,ptr + NSITES)* 0.5;
        }
		timer.stop();

		float elapsedTime = timer.elapsed();


		std::cout<<"Time on GPU:"<<elapsedTime<<"ms"<<" Energy calculated:"<<energy<<std::endl;

		//CPU TEST
		std::cout<<"-------------------------------------------------------------------------"<<std::endl;
		std::cout<<"Starting PairEnergy_I tests: CPU";

				timer_cpu.start();
		        for(int i =0; i < 10000;++i){
		        	for(int j=0;j<2;++j){
		        	calcEnergyCpu<kmcenergy::ising::PairEnergy_I,float>(h_sites,h_neigs,8,
						kmcenergy::ising::PairEnergy_I(),h_energies,energies,NSITES);}

			    energy = thrust::reduce(h_energies,h_energies + NSITES)* 0.5;
		        }
				timer_cpu.stop();

		 elapsedTime = timer_cpu.elapsed();


		 std::cout<<" Time: "<<elapsedTime<<"ms"<<" Energy calculated:"<<energy<<std::endl;
		 std::cout<<"PairEnergy_I completed."<<std::endl;
		 std::cout<<"-------------------------------------------------------------------------"<<std::endl;


		 timer.reset();

		 std::cout<<"Starting ExchangeEnergy test - Ising case - GPU:";

		 timer.start();
		 for(int i = 0 ;i <10000; ++i)
		 calcExchangeEnergy<float,kmcenergy::ising::PairEnergy_I,algorithms::ExchangeEnergy >
		 <<<16,512>>>(d_sites,d_neigs,8,
				 algorithms::ExchangeEnergy<float,kmcenergy::ising::PairEnergy_I>(),d_energies,d_params,NSITES);
		 timer.stop();
		 energy = thrust::reduce(ptr,ptr + NSITES)/NSITES;

		 std::cout<<"...avarage exchange energy E ="<<energy<<" "<<timer<<std::endl;

		 timer_cpu1.reset();
		 std::cout<<"Starting ExchangeEnergy test - Ising case - CPU:";
		 timer_cpu1.start();
		  	for(int i = 0 ;i <10000; ++i)
		 	 calcExchangeEnergyCpu<float,kmcenergy::ising::PairEnergy_I,algorithms::ExchangeEnergy >
		 		 (h_sites,h_neigs,8,
		 		 algorithms::ExchangeEnergy<float,kmcenergy::ising::PairEnergy_I>(),h_energies,energies,NSITES);
		 		  energy= thrust::reduce(h_energies,h_energies + NSITES)/NSITES;
		 timer_cpu1.stop();

		 				 std::cout<<"...avarage exchange energy E ="<<energy<<" "<<timer_cpu1<<std::endl;


	} else
	{
		std::cout << "No device found"<< std::endl;
		std::cout<<"Starting ExchangeEnergy test - Ising case - CPU:";
		timer_cpu1.start();
			findNeigboursXyzCpu(h_sites, h_neigs, base1, base2, base3, dimensions,
					radius, 8, 0, NSITES);
			timer_cpu1.stop();
		for(int i = 0 ;i <1; ++i)
				 calcExchangeEnergyCpu<float,kmcenergy::ising::PairEnergy_I,algorithms::ExchangeEnergy >
				 (h_sites,h_neigs,8,
						 algorithms::ExchangeEnergy<float,kmcenergy::ising::PairEnergy_I>(),h_energies,energies,NSITES);
				 float energyAvg = thrust::reduce(h_energies,h_energies + NSITES)/NSITES;

				 std::cout<<"Avarage exchange energy ="<<energyAvg<<std::endl;

	}




		free(h_sites);
		free(h_neigs);
		free(h_energies);
		if (devCount > 0) {
		CHECK_ERROR(cudaFree(d_sites));
		CHECK_ERROR(cudaFree(d_neigs));
		CHECK_ERROR(cudaFree(d_params));
		}
}
}
}
}

