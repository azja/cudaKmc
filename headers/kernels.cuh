/*
 * kernels.cuh
 *
 *  Created on: 27-03-2013
 *      Author: biborski
 */

#ifndef KERNELS_CUH_
#define KERNELS_CUH_

__global__ void findNeigboursXyz(const float4 * const sites,
		int4 * neigbours, float3 base1, float3 base2, float3 base3,
		int3 dimensions, float radius, int offset, int beginFrom, int size);

/*
 * SimulationInput interface should provide fields as defined in siminput.cuh:
 *
 * uint N       - total number of sites
 * uint n_v     - number of vacancies/active items
 * uint z       - maximal number of items(atoms) considered for interaction
 * uint z_t     - maximal number of items(atoms) considered for performing transition (e.g. jumping atoms)
 * uint atoms_n - number of atoms kind
 * float4* sites [N]
 * float* transitions [n_v * z_T]
 * uint4* jumpingNeigbours[n_v * z_T]
 * uint* vacancies [n_v]
 * uint4* neigbours [n_v * z]
 *
 */

 template<typename SimulationInput>
__global__ void calculateTransitionParameter(SimulationInput input) {

	typename SimulationInput::ExchangeEnergyFunctor exchangeCalcer;
	typename SimulationInput::SaddleEnergyFunctor saddleCalcer;


	int thId = blockIdx.x * blockDim.x + threadIdx.x;
	if (thId < input.n_v) {

		int id1 = input.vacancies[thId];

		for (int i = 0; i < input.z_t; ++i) {

			int id2 = input.jumpingNeigbours[id1 * input.z_t + i].w;

			input.transitions[thId * input.z_t + i] =


				0.5	* exchangeCalcer(id1, id2, input.sites, input.neigbours,
							input.z, input.exchangeEnergyCalculation,
							input.atoms_n) +


					saddleCalcer(id1, id2, input.sites, input.neigbours,
							input.saddleEnergyCalculation);

		}
	}

}




 template<typename SimulationInput>
  void calculateTransitionParameterCpu(SimulationInput input) {

 	typename SimulationInput::ExchangeEnergyFunctor exchangeCalcer;
 	typename SimulationInput::SaddleEnergyFunctor saddleCalcer;

 	int thId;
#pragma omp parallel for private(thId)
 	for(thId = 0; thId < input.n_v; ++thId)
 	 {
 		int id1 = input.vacancies[thId];
        int i;
       #pragma omp parallel for private(i)
 		for (i = 0; i < input.z_t; ++i) {
 			int id2 = input.jumpingNeigbours[thId * input.z_t + i].w;

 			input.transitions[thId * input.z_t + i] =


 				0.5	* exchangeCalcer(id1, id2, input.sites, input.neigbours,
 							input.z, input.exchangeEnergyCalculation,
 							input.atoms_n) +


 					saddleCalcer(id1, id2, input.sites, input.neigbours,
 							input.saddleEnergyCalculation);

 		}
 	}

 }

 template<typename SimulationInput>
 __global__ void calculateTransitionParameterNoId(SimulationInput input,float beta) {

     typename SimulationInput::ExchangeEnergyFunctor exchangeCalcer;
     typename SimulationInput::SaddleEnergyFunctor saddleCalcer;


     int thId = blockIdx.x * blockDim.x + threadIdx.x;
     if (thId < input.n_v) {

         int id1 = input.vacancies[thId];

         for (int i = 0; i < input.z_t; ++i) {

             int id2 = input.jumpingNeigbours[id1 * input.z_t + i].w;
           /*  input.transitions[thId * input.z_t + i] = 1000000000.0f;*/
             input.transitions[thId * input.z_t + i] = 0.0f;
             if(static_cast<int>(input.sites[id1].w) != static_cast<int>(input.sites[id2].w)){
             input.transitions[thId * input.z_t + i] = expf(-beta*(


                 0.5 * exchangeCalcer(id1, id2, input.sites, input.neigbours,
                             input.z, input.exchangeEnergyCalculation,
                             input.atoms_n) +


                     saddleCalcer(id1, id2, input.sites, input.neigbours,
                             input.saddleEnergyCalculation)));
             }

         }
     }

 }

/*
 * set float4 fields
 */


__global__ void setFloat4x(int index,float value,float4* input, int size);

__global__ void setFloat4y(int index,float value,float4* input, int size);

__global__ void setFloat4z(int index,float value,float4* input, int size);
__global__ void setFloat4w(int index,float value,float4* input, int size);

/*
 * set element value in array
 */

template<typename T>
__global__ void set1dArrayElement(int index, T value, T* input, int size) {
	int thId = blockIdx.x * blockDim.x + threadIdx.x;

	if(thId < size && thId == index)
		input[index] = value;

}
/*
 * Exchange values in Float4 fields
 */

__global__ void  exchangeFloat4x(int index1, int index2,float4* input, int size);

__global__ void  exchangeFloat4y(int index1, int index2,float4* input, int size);

__global__ void  exchangeFloat4z(int index1, int index2,float4* input, int size);

__global__ void  exchangeFloat4w(int index1, int index2,float4* input, int size);

#endif /* KERNELS_CUH_ */
