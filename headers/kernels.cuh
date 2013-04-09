/*
 * kernels.cuh
 *
 *  Created on: 27-03-2013
 *      Author: biborski
 */

#ifndef KERNELS_CUH_
#define KERNELS_CUH_

extern "C" __global__ void findNeigboursXyz(const float4 * const sites,
		int4 * neigbours, float3 base1, float3 base2, float3 base3,
		int3 dimensions, float radius, int offset, int beginFrom, int size);

extern "C" __global__ void findNeigboursXyzShared(const float4 * const sites,
		int4 * neigbours, float3 base1, float3 base2, float3 base3,
		int3 dimensions, float radius, int offset, int beginFrom, int size);

#endif /* KERNELS_CUH_ */
