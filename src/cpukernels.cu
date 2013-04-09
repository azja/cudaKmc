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

/*
 * sites[ dim.x * dim.y * dim.z * numOfAtoms]
 * atoms[numOfAtoms]
 */
extern "C"  void LatticeBuilder(float4* sites, float3 b1, float3 b2, float3 b3,float4 * atoms, int numOfAtoms, int3 dim) {

	for (int i = 0; i < dim.x; ++i) {
		for (int j = 0; j < dim.y; ++j) {
			for (int k = 0; k < dim.z; ++k) {

				int id = i * dim.x * dim.y + j * dim.y + k;

				for(int l =0;l<numOfAtoms;l++){

					float4 A;
					A.x = atoms[l].x;
					A.y = atoms[l].y;
					A.z = atoms[l].z;
					A.w = atoms[l].w;

					float4 siteToPut = { (b1.x * A.x + b2.x * A.y + b3.x * A.z) + i * b1.x
											+ j * b2.x + k * b3.x, (b2.x * A.y + b2.y * A.y
													+ b2.z * A.y) + i * b1.y + j * b2.y + k * b3.y, (b3.x
															* A.z + b3.y * A.z + b3.z * A.z) + i * b1.z + j * b2.z
															+ k * b3.z, A.w };
					sites[id *numOfAtoms +l] = siteToPut;
				}


			}
		}
	}
}

