#include "../headers/utils.h"
#include <iostream>
namespace utils{


int MaxNumberOfThreads( cudakmc::verbosity v, int devNumber){

	cudaDeviceProp properties;
	CHECK_ERROR(cudaGetDeviceProperties(&properties,devNumber));
	int nOfDev = properties.maxThreadsPerMultiProcessor * properties.multiProcessorCount;

	if(v == cudakmc::verbose)
		std::cout<<"Maximum no. threads allowed:"<<nOfDev<<" for "<<properties.name<<std::endl;

	return nOfDev;
	}

}
