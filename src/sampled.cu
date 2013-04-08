/*
 * Implementation of SampleDevice class
 */

#include "../headers/sampled.h"
#include "../headers/utils.h"
#include "../macros/errors.h"
#include <thrust/count.h>
#include <thrust/device_ptr.h>

samples::SampleDevice::~SampleDevice() {
	CHECK_ERROR(cudaFree(_sites));
	CHECK_ERROR(cudaFree(_transitions));
	CHECK_ERROR(cudaFree(_jumpingNeigbours));
	CHECK_ERROR(cudaFree(_vacancies));
	CHECK_ERROR(cudaFree(_neigbours));
}

uint samples::SampleDevice::getNumberOf(definitions::Atom atom) const {

	thrust::device_ptr<float4> d = thrust::device_pointer_cast(_sites);
	return thrust::count_if(d, d + _N, utils::IsAtomPredicate(atom));

}

void samples::SampleDevice::calculateNeigbours() {

}

void samples::SampleDevice::allocate(uint N, uint n_v, uint z, uint z_t) {

	CHECK_ERROR(cudaMalloc((void**)&_sites, sizeof(float4) * N));
	CHECK_ERROR(cudaMalloc((void**)&_transitions, sizeof(float) * n_v * z_t));
	CHECK_ERROR(cudaMalloc((void**)&_jumpingNeigbours, sizeof(float) * n_v * z_t));
	CHECK_ERROR(cudaMalloc((void**)&_vacancies, sizeof(int) * n_v));
	CHECK_ERROR(cudaMalloc((void**)&_neigbours, sizeof(int4) * N * z ));

}
