/*
 * cpukernels.h
 *
 *  Created on: 29-03-2013
 *      Author: biborski
 */

#ifndef CPUKERNELS_H_
#define CPUKERNELS_H_

extern "C" void findNeigboursXyzCpu(const float4 * const sites,
        int4 * neigbours, float3 base1, float3 base2, float3 base3,
        int3 dimensions, float radius, int offset, int begin, int end);

extern "C"  void LatticeBuilder(float4* sites, float3 b1, float3 b2,
        float3 b3,float4 * atoms, int numOfAtoms, int3 dim);

#endif /* CPUKERNELS_H_ */
