/*
 * cpukernels.cpp
 *
 *  Created on: 29-03-2013
 *      Author: biborski
 */
#include "../headers/cpukernels.h"
#include <iostream>

extern "C" void findNeigboursXyzCpu(const float4 * const sites,
		int4 * neigbours, float3 base1, float3 base2, float3 base3,
		int3 dimensions, float radius, int offset, int begin, int end) {

	for (int id = 0; id < end; ++id) {

		float x = sites[id].x;
		float y = sites[id].y;
		float z = sites[id].z;

		float lx;
		float ly;
		float lz;

		int nCntr = 0;

		for (int i = begin; i < end; ++i) {
			if (i != id) {

				float xp = sites[i].x - x;
				float yp = sites[i].y - y;
				float zp = sites[i].z - z;

#pragma unroll
				for (int k = -1; k <= 1; ++k) {
#pragma unroll
					for (int l = -1; l <= 1; ++l) {
#pragma unroll
						for (int m = -1; m <= 1; ++m) {

							lx = k * dimensions.x * base1.x
									+ l * dimensions.y * base2.x
									+ m * dimensions.z * base3.x;
							ly = k * dimensions.x * base1.y
									+ l * dimensions.y * base2.y
									+ m * dimensions.z * base3.y;
							lz = k * dimensions.x * base1.z
									+ l * dimensions.z * base2.z
									+ m * dimensions.z * base3.z;

							float distance = sqrtf(
									(xp + lx) * (xp + lx)
									+ (yp + ly) * (yp + ly)
									+ (zp + lz) * (zp + lz));

							if (distance < radius && nCntr < offset) {
								neigbours[id * offset + nCntr].x = -k;
								neigbours[id * offset + nCntr].y = -l;
								neigbours[id * offset + nCntr].z = -m;
								neigbours[id * offset + nCntr].w = i;
								nCntr++;

							}
						}
					}
				}
			}
		}

	}
}

