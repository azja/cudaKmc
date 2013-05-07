/*
 * algorithms.cuh
 *
 *  Created on: 05-04-2013
 *      Author: biborski
 */

#ifndef ALGORITHMS_CUH_
#define ALGORITHMS_CUH_

#include "isinge.cuh"
/*
 * EnergyKernel  template parameter is class/structure  including two functors:
 *
 * __host__ __device__  float operator()(int atom,
 *	     	                               const float4* const sites,
 *  		                               const int4 * const neigbours,
 *		                                   int size,
 *			                               int id,
 *		                                   const T* parameters,
 *		                                   int atomsN);
 * U<T,R>
 *
 *
 * and another version:
 * __host__ __device__  float operator()(int atom,
 *			                               const float4* const sites,
 *		                                   const int4 * const neigbours,
 *		                                   int size,
 *			                               int id,
 *  		                               const T* parameters,
 *	    	                               int atomsN,int4 exchange)
 */

namespace algorithms {

template<typename EnergyParameters, typename EnergyKernel>
struct ExchangeEnergy {

    inline __host__ __device__ float operator()(int id1, int id2,
            const float4* const sites, const int4* const neigbours, int size,
            const EnergyParameters* parameters, int atomsN) {

        EnergyKernel kernel;

        int exAtom1 = static_cast<int>(sites[id1].w);
        int exAtom2 = static_cast<int>(sites[id2].w);

        float E0 = 0.0f;

        int base1 = id1 * size;
        int base2 = id2 * size;

        int n1;
        int n2;

        int k1;
        int k2;

        for (int i = 0; i < size; ++i) {
            k1 = neigbours[base1 + i].w;
            k2 = neigbours[base2 + i].w;
            n1 = static_cast<int>(sites[k1].w);
            n2 = static_cast<int>(sites[k2].w);
            if (neigbours[base1 + i].w != id2)
                E0 += (kernel(n1, sites, neigbours, size, k1, parameters,
                        atomsN));
            if (neigbours[base2 + i].w != id1)
                E0 += (kernel(n2, sites, neigbours, size, k2, parameters,
                        atomsN));
        }

        int4 exchanger;
        exchanger.w = id1;
        exchanger.x = id2;
        exchanger.y = exAtom2;
        exchanger.z = exAtom1;

        float E1 = 0.0f;

        for (int i = 0; i < size; ++i) {
            k1 = neigbours[base1 + i].w;
            k2 = neigbours[base2 + i].w;
            n1 = static_cast<int>(sites[k1].w);
            n2 = static_cast<int>(sites[k2].w);
            if (neigbours[base1 + i].w != id2)
                E1 += kernel(n1, sites, neigbours, size, k1, parameters, atomsN,
                        exchanger);
            if (neigbours[base2 + i].w != id1)
                E1 += kernel(n2, sites, neigbours, size, k2, parameters, atomsN,
                        exchanger);
        }

        return E1 - E0;

    }
};

template<>
struct ExchangeEnergy<float, kmcenergy::ising::PairEnergy_I> {

    __host__ __device__ float operator()(int id1, int id2,
            const float4* const sites, const int4* const neigbours, int size,
            const float* parameters, int atomsN) {

        kmcenergy::ising::PairEnergy_I kernel;

        int exAtom1 = static_cast<int>(sites[id1].w);
        int exAtom2 = static_cast<int>(sites[id2].w);

        float E0 = 0.0f;

        E0 +=
                (kernel(exAtom1, sites, neigbours, size, id1, parameters,
                        atomsN));
        E0 +=
                (kernel(exAtom2, sites, neigbours, size, id2, parameters,
                        atomsN));

        int4 exchanger;
        exchanger.w = id1;
        exchanger.x = id2;
        exchanger.y = exAtom2;
        exchanger.z = exAtom1;

        float E1 = 0.0f;

        E1 += kernel(exAtom2, sites, neigbours, size, id1, parameters, atomsN,
                exchanger);

        E1 += kernel(exAtom1, sites, neigbours, size, id2, parameters, atomsN,
                exchanger);

        return E1 - E0;

    }

};

struct ConstantSaddleEnergy {

    /*
     * Provides saddle point - barrier value according to the type of jumping atom
     * Some of funtor paramaters are reduntant - they are kept to maintain consistency
     * of interface.
     */
    __host__ __device__ float operator()(int id1, int id2,
            const float4* const sites, const int4 * const neigbours,
            const float* params) {

        return params[static_cast<int>(sites[id1].w)];
    }
};

}
#endif /* ALGORITHMS_CUH_ */
