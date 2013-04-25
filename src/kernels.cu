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
