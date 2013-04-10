#include "../headers/kernels.cuh"

__global__ void findNeigboursXyz(const float4 * const sites,
		int4 * neigbours, float3 base1, float3 base2, float3 base3,
		int3 dimensions, float radius, int offset, int beginFrom, int size) {

	int id = blockDim.x * blockIdx.x + threadIdx.x + beginFrom;

	extern __shared__ float X[];

	if (id < size) {

		float x = sites[id].x;
		float y = sites[id].y;
		float z = sites[id].z;

		float lx;
		float ly;
		float lz;

		int nCntr = 0;

		for (int j = 0; j < gridDim.x; ++j) {
			__syncthreads();
			if ((threadIdx.x + blockDim.x * j) < size) {

				X[threadIdx.x] = sites[threadIdx.x + blockDim.x * j].x;
				X[threadIdx.x + blockDim.x] =
						sites[threadIdx.x + blockDim.x * j].y;
				X[threadIdx.x + 2 * blockDim.x] = sites[threadIdx.x
				                                        + blockDim.x * j].z;
				__syncthreads();

				for (int i = 0; i < blockDim.x; ++i) {
					if ((j * blockDim.x + i) != id) {

						float xp = X[i] - x;
						float yp = X[i + blockDim.x] - y;
						float zp = X[i + 2 * blockDim.x] - z;
#pragma unroll
						for (int k = -1; k <= 1; ++k) {
#pragma unroll
							for (int l = -1; l <= 1; ++l) {
#pragma unroll
								for (int m = -1; m <= 1; ++m) {

									lx = k * dimensions.x * base1.x
											+ l * dimensions.y * base2.x
											+ m * dimensions.z * base3.x;
									ly = k * dimensions.x * base1.y
											+ l * dimensions.y * base2.y
											+ m * dimensions.z * base3.y;
									lz = k * dimensions.x * base1.z
											+ l * dimensions.z * base2.z
											+ m * dimensions.z * base3.z;

									float distance = sqrtf(
											(xp + lx) * (xp + lx)
											+ (yp + ly) * (yp + ly)
											+ (zp + lz) * (zp + lz));

									if (distance < radius && nCntr < offset) {

										neigbours[id * offset + nCntr].x = -k;
										neigbours[id * offset + nCntr].y = -l;
										neigbours[id * offset + nCntr].z = -m;
										neigbours[id * offset + nCntr].w = j
												* blockDim.x + i;
										nCntr++;
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

	typename SimulationInput::EnergyExchange exchangeCalcer;
	typename SimulationInput::SaddleEnergyFunctor saddleCalcer;

	int thId = blockIdx.x * blockDim.x + threadIdx.x;
	if (thId < input.N) {
		int id1 = input.vacancies[thId];

		for (int i = 0; i < input.z_t; ++i) {
			int id2 = input.jumpingNeigbours[thId * input.z_t + i].w;

			input.transitions[thId * input.z_t + i] =
			/*0.5 x dE */

				0.5	* exchangeCalcer(id1, id1, input.sites, input.neigbours,
							input.N, input.exchangeEnergyCalculation,
							input.atoms_n)
					+
					/* Es */
					saddleCalcer(id1, id2, input.sites, input.neigbours,
							input.saddleEnergyCalculation);

		}
	}

}
