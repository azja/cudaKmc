#include "../headers/kernels.cuh"

<<<<<<< HEAD



#ifndef DEBUG

 __global__ void findNeigboursXyz(const float4 * const sites,
        int4 * neigbours, float3 base1, float3 base2, float3 base3,
        int3 dimensions, float radius, int offset, int beginFrom, int size) {

    int id = blockDim.x * blockIdx.x + threadIdx.x + beginFrom;
    radius = radius*radius;

    extern __shared__ float4 X[];

    int nCntr = 0;
    int i,tile;


    float x = sites[id].x;
    float y = sites[id].y;
    float z = sites[id].z;

    float lx;
    float ly;
    float lz;

    float xp;
    float yp;
    float zp;

    for( i = 0, tile = 0; i < size; i += blockDim.x,++tile)
    {

        if(tile * blockDim.x + threadIdx.x < size)
        {
            float4 d = sites[tile * blockDim.x + threadIdx.x];
            X[threadIdx.x] = d;
        }
        else
        {
            // nie koniecznie, ale lepiej miec pelna kontrole
            X[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
=======
__global__ void findNeigboursXyz(
         const float4* const sites,
         int4*  const neighbours,
         float3 base1,
         float3 base2,
         float3 base3,
         int3   dimensions,
         float  radius,
         int    offset,
         int    beginFrom,
         int    size)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x + beginFrom;

    // TODO pass square of radius as input parameter
    const float squareRadius = radius * radius;

    // TODO store as float3 structures
    extern __shared__ float4 shared_sites[];

    // load site data assigned for this thread
    float4 currentSite = sites[id];

    int neighbourCounter = 0;

    for (int i = 0, tile = 0; i < size; i += blockDim.x, ++tile)
    {
        if(tile * blockDim.x + threadIdx.x < size)
        {
            float4 site = sites[tile * blockDim.x + threadIdx.x];
            shared_sites[threadIdx.x] = site;
        }
        else
        {
            shared_sites[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
>>>>>>> 77227424e40455e57d5f50baef79fe6646aba680
        }

        __syncthreads();

<<<<<<< HEAD
        for (int ii =0; ii < blockDim.x; ++ii)
        {

            if (tile * blockDim.x + ii != id && tile * blockDim.x + ii < size) //<<---- (ii != id jest OK)
            {
                xp = X[ii].x - x;
                yp = X[ii].y - y;
                zp = X[ii].z - z;
=======
        for (int isite = 0; isite < blockDim.x; ++isite)
        {
            if ((tile * blockDim.x + isite != id) && (tile * blockDim.x + isite < size))
            {
                float4 site = shared_sites[isite];

                float xp = site.x - currentSite.x;
                float yp = site.y - currentSite.y;
                float zp = site.z - currentSite.z;
>>>>>>> 77227424e40455e57d5f50baef79fe6646aba680

                #pragma unroll
                for (int k = -1; k <= 1; ++k)
                {
                    #pragma unroll
                    for (int l = -1; l <= 1; ++l)
                    {
                        #pragma unroll
                        for (int m = -1; m <= 1; ++m)
                        {
<<<<<<< HEAD
                            lx = k * dimensions.x * base1.x
                                    + l * dimensions.y * base2.x
                                    + m * dimensions.z * base3.x;
                            ly = k * dimensions.x * base1.y
                                    + l * dimensions.y * base2.y
                                    + m * dimensions.z * base3.y;
                            lz = k * dimensions.x * base1.z
                                    + l * dimensions.y * base2.z
                                    + m * dimensions.z * base3.z;

                            float distance = (
                                    (xp + lx) * (xp + lx)
                                    + (yp + ly) * (yp + ly)
                                    + (zp + lz) * (zp + lz));

                            if (distance < radius && nCntr < offset)
                            {
                                neigbours[id * offset + nCntr].x = -k;
                                neigbours[id * offset + nCntr].y = -l;
                                neigbours[id * offset + nCntr].z = -m;
                                neigbours[id * offset + nCntr].w =  tile * blockDim.x + ii;
                                nCntr++;
=======
                            float lx = k * dimensions.x * base1.x + l * dimensions.y * base2.x + m * dimensions.z * base3.x;
                            float ly = k * dimensions.x * base1.y + l * dimensions.y * base2.y + m * dimensions.z * base3.y;
                            float lz = k * dimensions.x * base1.z + l * dimensions.y * base2.z + m * dimensions.z * base3.z;

                            float squareDistance = (xp + lx) * (xp + lx) + (yp + ly) * (yp + ly) + (zp + lz) * (zp + lz);

                            // TODO use fabs
                            if ((neighbourCounter < offset) && (squareDistance < squareRadius))
                            {
                                neighbours[id * offset + neighbourCounter] = make_int4(-k, -l, -m, tile * blockDim.x + isite);
                                ++neighbourCounter;
>>>>>>> 77227424e40455e57d5f50baef79fe6646aba680
                            }
                        }
                    }
                }
            }
        }

<<<<<<< HEAD
        __syncthreads(); //<<---- bez tego nie ma kontroli nad shared bufforem X.
    }
}

#endif

 /*
  * set float4 fields
  */


 __global__ void setFloat4x(int index,float value,float4* input, int size) {
  	int thId = blockIdx.x * blockDim.x + threadIdx.x;
  	if(thId < size && index == thId)
  		input[thId].x = value;
  }

 __global__ void setFloat4y(int index,float value,float4* input, int size) {
  	int thId = blockIdx.x * blockDim.x + threadIdx.x;
  	if(thId < size && index == thId)
  		input[thId].y = value;
  }

 __global__ void setFloat4z(int index,float value,float4* input, int size) {
  	int thId = blockIdx.x * blockDim.x + threadIdx.x;
  	if(thId < size && index == thId)
  		input[thId].z = value;
  }
 __global__ void setFloat4w(int index,float value,float4* input, int size) {
 	int thId = blockIdx.x * blockDim.x + threadIdx.x;
 	if(thId < size && index == thId)
 		input[thId].w = value;
 }


 /*
  * Exchange values in Float4 fields
  */

 __global__ void  exchangeFloat4x(int index1, int index2,float4* input, int size) {
 	int thId = blockIdx.x * blockDim.x + threadIdx.x;
 	if(thId < size && thId == index1) {
 		float temp = input[thId].x;
 		input[thId].x = input[index2].x;
 		input[index2].x = temp;
 	}
 }

 __global__ void  exchangeFloat4y(int index1, int index2,float4* input, int size) {
 	int thId = blockIdx.x * blockDim.x + threadIdx.x;
 	if(thId < size && thId == index1) {
 		float temp = input[thId].y;
 		input[thId].y = input[index2].y;
 		input[index2].y = temp;
 	}
 }

 __global__ void  exchangeFloat4z(int index1, int index2,float4* input, int size) {
 	int thId = blockIdx.x * blockDim.x + threadIdx.x;
 	if(thId < size && thId == index1) {
 		float temp = input[thId].z;
 		input[thId].z = input[index2].z;
 		input[index2].z = temp;
 	}
 }

 __global__ void  exchangeFloat4w(int index1, int index2,float4* input, int size) {
 	int thId = blockIdx.x * blockDim.x + threadIdx.x;

 	if(thId < size && thId == index1) {
 		float temp = input[thId].w;
 		input[thId].w = input[index2].w;
 		input[index2].w = temp;
 	}
 }






 /* To jest wersja działająca - nie jest w inej używana pamięć shared*/


#ifdef DEBUG

 __global__ void findNeigboursXyz(const float4 * const sites,
		int4 * neigbours, float3 base1, float3 base2, float3 base3,
		int3 dimensions, float radius, int offset, int beginFrom, int size) {

	int id = blockDim.x * blockIdx.x + threadIdx.x + beginFrom;

	if (id < size) {

		float x = sites[id].x;
		float y = sites[id].y;
		float z = sites[id].z;

		float lx;
		float ly;
		float lz;

		int nCntr = 0;

		for (int i = beginFrom; i < size; ++i) {
			if (i != id) {

				float xp = sites[i].x - x;
				float yp = sites[i].y - y;
				float zp = sites[i].z - z;

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
									+ l * dimensions.y * base2.z
									+ m * dimensions.z * base3.z;

							float distance = sqrt(
									(xp + lx) * (xp + lx)
									+ (yp + ly) * (yp + ly)
									+ (zp + lz) * (zp + lz));

							if (distance < radius && nCntr < offset) {

								neigbours[id * offset + nCntr].x = -k;
								neigbours[id * offset + nCntr].y = -l;
								neigbours[id * offset + nCntr].z = -m;
								neigbours[id * offset + nCntr].w =  i;
								nCntr++;
							}
						}
					}
				}
			}
		}

	}
=======
        __syncthreads();

    }
>>>>>>> 77227424e40455e57d5f50baef79fe6646aba680
}

#endif

<<<<<<< HEAD
=======
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
>>>>>>> 77227424e40455e57d5f50baef79fe6646aba680
