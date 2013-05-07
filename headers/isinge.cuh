/*
 * isinge.cuh
 *
 *  Created on: 01-04-2013
 *      Author: biborski
 */

#ifndef ISINGE_CUH_
#define ISINGE_CUH_

#define N_ATOMS_ 2



namespace kmcenergy {
namespace ising {



struct PairEnergy_I {
    /* This function provides energy of interacting atom according to:
     * atom     - central interacting atom
     * sites    - table of sites
     * size     - number of neigbours, this variable has stride characterstics
     * id       - thread id
     * energies - table of pair-wise interactions of size of atomN x atomN
     * atoms    - number of atomic kinds
     */

    __host__ __device__ float operator()(int atom, const float4* const sites,
            const int4 * const neigbours, int size, int id,
            const float* energies, int atomsN) {

        float energy = 0.0f;
        int  offset = id * size;
        int siteIndex = 0;
        for (int i = 0; i < size; ++i) {
            siteIndex = neigbours[offset + i].w;
            energy +=  energies[atom*atomsN
                                + static_cast<int>(sites[siteIndex].w)];
        }
        return energy;
    }

    /* This function provides energy of interacting atom according to:
     * atom     - central interacting atom
     * sites    - table of sites
     * size     - number of neigbours, this variable has stride characterstics
     * id       - thread id
     * energies - table of pair-wise interactions of size of atomN x atomN
     * atoms    - number of atomic kinds
     * exchange - float4 structure for energy calculation when atomic species are interchanged:
     *             exchange.w 1st atom id in sites
     *             exchange.x 2nd atom id in sites
     *             exchange.y 1st atom kind substitution
     *             exchange.z 2nd atom kind substitution
     *
     */

    __host__ __device__ float operator()(int atom, const float4* const sites,
            const int4 * const neigbours, int size, int id,
            const float* energies, int atomsN, int4 exchange) {

        float energy = 0.0f;

        for (int i = 0; i < size; ++i) {
            int siteIndex =  neigbours[id * size + i].w;


            if (siteIndex == exchange.w) {
                energy +=  energies[atom * atomsN + exchange.y];
                continue;
            }
            if (siteIndex == exchange.x) {
                energy += energies[atom * atomsN + exchange.z];
                continue;
            }
            energy += energies[atom * atomsN +
                               + static_cast<int>(sites[siteIndex].w)];
        }
        return energy;
    }

};

struct PairEnergy_II {

    __device__ float operator()(float4* sites, int4 *neigbours, int size,
            int id);

};

struct PairEnergy_III {

    __device__ float operator()(float4* sites, int4 *neigbours, int size,
            int id);

};

}
}
#endif /* ISINGE_CUH_ */
