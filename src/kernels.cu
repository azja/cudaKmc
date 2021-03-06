#include "../headers/kernels.cuh"

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
        }

        __syncthreads();

        for (int isite = 0; isite < blockDim.x; ++isite)
        {
            if ((tile * blockDim.x + isite != id) && (tile * blockDim.x + isite < size))
            {
                float4 site = shared_sites[isite];

                float xp = site.x - currentSite.x;
                float yp = site.y - currentSite.y;
                float zp = site.z - currentSite.z;

                #pragma unroll
                for (int k = -1; k <= 1; ++k)
                {
                    #pragma unroll
                    for (int l = -1; l <= 1; ++l)
                    {
                        #pragma unroll
                        for (int m = -1; m <= 1; ++m)
                        {
                            float lx = k * dimensions.x * base1.x + l * dimensions.y * base2.x + m * dimensions.z * base3.x;
                            float ly = k * dimensions.x * base1.y + l * dimensions.y * base2.y + m * dimensions.z * base3.y;
                            float lz = k * dimensions.x * base1.z + l * dimensions.y * base2.z + m * dimensions.z * base3.z;

                            float squareDistance = (xp + lx) * (xp + lx) + (yp + ly) * (yp + ly) + (zp + lz) * (zp + lz);

                            // TODO use fabs
                            if ((neighbourCounter < offset) && (squareDistance < squareRadius))
                            {
                                neighbours[id * offset + neighbourCounter] = make_int4(-k, -l, -m, tile * blockDim.x + isite);
                                ++neighbourCounter;
                            }
                        }
                    }
                }
            }
        }

        __syncthreads();
    }
}


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

 __global__ void findNeigboursXyzGlobal(const float4 * const sites,
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
}

#endif

