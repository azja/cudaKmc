/*
 * algorithms.cuh
 *
 *  Created on: 05-04-2013
 *      Author: biborski
 */

#ifndef ALGORITHMS_CUH_
#define ALGORITHMS_CUH_

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

template <typename EnergyParameters, typename EnergyKernel>
struct ExchangeEnergy{

	__host__ __device__ float operator()(int id1, int id2,
			                             const float4* const sites,
			                             const int4* const neigbours,
			                             int size,  const EnergyParameters* parameters,
			                             int atomsN) {

	  EnergyKernel kernel;

	  int exAtom1 = static_cast<int>(sites[id1].w);
	  int exAtom2 = static_cast<int>(sites[id2].w);

	  float E0 ;

      int base1 = id1 * size;
      int base2 = id2 * size;

      int n1;
      int n2;

      for(int i = 0; i < size; ++i){
        n1 =  static_cast<int>(sites[neigbours[base1 +i].w].w);
        n2 =  static_cast<int>(sites[neigbours[base2 +i].w].w);
          if(neigbours[base1 +i].w != id2)
    	E0 += kernel (n1, sites, neigbours, size,neigbours[base1 +i].w, parameters, atomsN);
          if(neigbours[base2 +i].w != id1)
        E0 += kernel (n2, sites, neigbours, size, neigbours[base2 +i].w, parameters, atomsN);}


      int4 exchanger;
      exchanger.w = id1;
      exchanger.x = id2;
      exchanger.y = exAtom2;
      exchanger.z = exAtom1;


	  float E1;

      for(int i = 0; i < size; ++i) {
    	  n1 =  static_cast<int>(sites[neigbours[base1 +i].w].w);
    	  n2 =  static_cast<int>(sites[neigbours[base2 +i].w].w);
    	  if(neigbours[base1 +i].w != id2)
    	    E1 += kernel (n1, sites, neigbours, size,neigbours[base1 +i].w, parameters, atomsN, exchanger);
    	  if(neigbours[base2 +i].w != id1)
    	    E1 += kernel (n2, sites, neigbours, size, neigbours[base2 +i].w, parameters, atomsN,exchanger);
      }

	  return E1-E0;

	}
};

}


#endif /* ALGORITHMS_CUH_ */
