/*
 * siminput.cuh
 *
 *  Created on: 10-04-2013
 *      Author: biborski
 */

#ifndef SIMINPUT_CUH_
#define SIMINPUT_CUH_

#include "../macros/errors.h"
/*
 * R - should be the type of parameters for simulation
 * S - struct/class providing energy calculation functor
 * T - struct/class providing saddle energy calculation functor
 * U - struct/class providing exchange energy calculation functor
 */

template <class R,class S, class T, template  <class,class> class U>
struct SimulationDeviceInput {

    typedef R        SimulationParameters;
    typedef S        EnergyFunctor;
    typedef T        SaddleEnergyFunctor;
    typedef U<R,S>   ExchangeEnergyFunctor;

    /*
     * N   - total number of sites
     * n_v - number of vacancies/active items
     * z   - maximal number of items(atoms) considered for interaction
     * z_t - maximal number of items(atoms) considered for performing transition (e.g. jumping atoms)
     * atoms_n - number of atomic specimen kinds
     */

    int N;
    int n_v;
    int z;
    int z_t;
    int atoms_n;

    /*
     * sites [N]
     * transitions [n_v * z_T]
     * jumpingNeigbours[n_v * z_T]
     * vacancies [n_v]
     * neigbours [n_v * z]
     */


    float4* sites;
    float*  transitions;
    int4*   jumpingNeigbours;
    int*    vacancies;
    int4*   neigbours;

    float temperature;

    SimulationParameters* exchangeEnergyCalculation;
    int numberOfItemsExchangeEnergy;
    SimulationParameters* saddleEnergyCalculation;
    int numberOfItemsSaddleEnergy;

    SimulationDeviceInput():numberOfItemsExchangeEnergy(1),numberOfItemsSaddleEnergy(1){}



    __host__ void ToDevice(SimulationDeviceInput& input) const {

        CHECK_ERROR(cudaMalloc((void**)&input.sites,sizeof(float4) * N));
        CHECK_ERROR(cudaMemcpy(input.sites,sites,sizeof(float4) *N,cudaMemcpyHostToDevice));

        CHECK_ERROR(cudaMalloc((void**)&input.transitions,sizeof(float) * n_v * z_t));
        CHECK_ERROR(cudaMemcpy(input.transitions,transitions,sizeof(float) * n_v * z_t,cudaMemcpyHostToDevice));

        CHECK_ERROR(cudaMalloc((void**)&input.jumpingNeigbours,sizeof(int4) * n_v * z_t));
        CHECK_ERROR(cudaMemcpy(input.jumpingNeigbours,jumpingNeigbours,sizeof(int4) * n_v * z_t,cudaMemcpyHostToDevice));

        CHECK_ERROR(cudaMalloc((void**)&input.vacancies,sizeof(int) * n_v ));
        CHECK_ERROR(cudaMemcpy(input.vacancies,vacancies,sizeof(int) * n_v ,cudaMemcpyHostToDevice));

        CHECK_ERROR(cudaMalloc((void**)&input.neigbours,sizeof(int4) * n_v * z ));
        CHECK_ERROR(cudaMemcpy(input.neigbours,neigbours,sizeof(int4) * n_v * z ,cudaMemcpyHostToDevice));

        CHECK_ERROR(cudaMalloc((void**)&exchangeEnergyCalculation,sizeof(SimulationParameters) * numberOfItemsExchangeEnergy));
        CHECK_ERROR(cudaMalloc((void**)&saddleEnergyCalculation,sizeof(SimulationParameters) * numberOfItemsSaddleEnergy));

        input.N = N;
        input.temperature = temperature;
        input.atoms_n = atoms_n;
        input.n_v = n_v;
        input.z_t = z_t;
        input.z = z;
        input.numberOfItemsExchangeEnergy = numberOfItemsExchangeEnergy;
        input.numberOfItemsSaddleEnergy = numberOfItemsSaddleEnergy;

    }

    __host__ void ToHost(SimulationDeviceInput& input) const {

        input.sites = (float4*)malloc(sizeof(float4) * N);
        CHECK_ERROR(cudaMemcpy(input.sites,sites,sizeof(float4) *N,cudaMemcpyDeviceToHost));

        input.transitions=(float*)malloc(sizeof(float) * n_v * z_t);
        CHECK_ERROR(cudaMemcpy(input.transitions,transitions,sizeof(float) * n_v * z_t,cudaMemcpyDeviceToHost));

        input.jumpingNeigbours=(int4*)malloc(sizeof(int4) * N * z_t);
        CHECK_ERROR(cudaMemcpy(input.jumpingNeigbours,jumpingNeigbours,sizeof(int4) * N * z_t,cudaMemcpyDeviceToHost));

        input.vacancies =(int*)malloc(sizeof(int) * n_v );
        CHECK_ERROR(cudaMemcpy(input.vacancies,vacancies,sizeof(int) * n_v ,cudaMemcpyDeviceToHost));

        input.neigbours = (int4*)malloc(sizeof(int4) * N * z );
        CHECK_ERROR(cudaMemcpy(input.neigbours,neigbours,sizeof(int4) * N * z ,cudaMemcpyDeviceToHost));



        input.N = N;
        input.temperature = temperature;
        input.atoms_n = atoms_n;
        input.n_v = n_v;
        input.z_t = z_t;
        input.z = z;
        input.numberOfItemsExchangeEnergy = numberOfItemsExchangeEnergy;
        input.numberOfItemsSaddleEnergy = numberOfItemsSaddleEnergy;

    }

    __host__ void FreeToDevice( ) {
        cudaFree(sites);
        cudaFree(transitions);
        cudaFree(jumpingNeigbours);
        cudaFree(vacancies);
        cudaFree(neigbours);
        cudaFree(exchangeEnergyCalculation);
        cudaFree(saddleEnergyCalculation);
    }

    __host__ void FreeToHost( ) {
        free(sites);
        free(transitions);
        free(jumpingNeigbours);
        free(vacancies);
        free(neigbours);
        free(exchangeEnergyCalculation);
        free(saddleEnergyCalculation);
    }


};




#endif /* SIMINPUT_CUH_ */
