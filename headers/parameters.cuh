/*
 * siminput.cuh
 *
 * Created on: 10-04-2013
 * Author: biborski
 */

#ifndef SIMINPUT_CUH_
#define SIMINPUT_CUH_
/*
 * R - should be the type of parameters for simulation
 * S - struct/class providing energy calculation functor
 * T - struct/class providing saddle energy calculation functor
 * U - struct/class providing exchange energy calculation functor
 */
template <class R,class S, class T, template <class,class> class U>
struct SimulationDeviceInput {

    typedef R SimulationParameters;;
    typedef S EnergyFunctor;
    typedef T SaddleEnergyFunctor;
    typedef U<R,S> ExchangeEnergyFunctor;

    /*
     * N - total number of sites
     * n_v - number of vacancies/active items
     * z - maximal number of items(atoms) considered for interaction
     * z_t - maximal number of items(atoms) considered for performing transition (e.g. jumping atoms)
     * atoms_n - number of atomic speciment kinds
     */

    uint N;
    uint n_v;
    uint z;
    uint z_t;
    uint atoms_n;

    /*
     * sites [N]
     * transitions [n_v * z_T]
     * jumpingNeigbours[n_v * z_T]
     * vacancies [n_v]
     * neigbours [n_v * z]
     */


    float4* sites;
    float* transitions;
    int4* jumpingNeigbours;
    uint* vacancies;
    int4* neigbours;

    SimulationParameters* exchangeEnergyCalculation;
    SimulationParameters* saddleEnergyCalculation;

};




#endif /* SIMINPUT_CUH_ */
