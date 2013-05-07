/*
 * buildlattice.c
 *
 *  Created on: 08-04-2013
 *      Author:Andrzej  Biborski
 */

#include "../../headers/cpukernels.h"
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include "buildlattice.h"

namespace tests {
namespace lattice {

float4* cubicL10(int size) {

	float3 cell1 = { 5.0f, 0.0f, 0.0f };
	float3 cell2 = { 0.0f, 5.0f, 0.0f };
	float3 cell3 = { 0.0f, 0.0f, 5.0f };

	float4 atom1 = { 0.0f, 0.0f, 0.0f, 0.0f };
	float4 atom2 = { 0.5f, 0.5f, 0.0f, 0.0f };

	float4 atom3 = { 0.5f, 0.0f, 0.5f, 1.0f };
	float4 atom4 = { 0.0f, 0.5f, 0.5f, 1.0f };

	float4 atoms[4];

	atoms[0] = atom1;
	atoms[1] = atom2;
	atoms[2] = atom3;
	atoms[3] = atom4;

	int3 dim = { size, size, size };
	float4* sites = (float4*) malloc(size * size * size * sizeof(float4) * 4);
	LatticeBuilder(sites, cell1, cell2, cell3, atoms, 4, dim);

	return sites;
}

float4* cubicB2(int size) {


	float4 atom1 = { 0.0f, 0.0f, 0.0f, 0.0f };
	float4 atom2 = { 0.5f, 0.5f, 0.5f, 1.0f };

	float4 atoms[2];

	atoms[0] = atom1;
	atoms[1] = atom2;

	int3 dim = { size, size, size };
	float4* sites = (float4*) malloc(size * size * size * sizeof(float4) * 2);
	LatticeBuilder(sites, b2cell1, b2cell2, b2cell3, atoms, 2, dim);

	return sites;
}

float4* hexagonal(int size) {

	float3 cell1 = { -sqrt(3.0f), 1.0f, 0.0f };
	float3 cell2 = { sqrt(3.0), 1.0f, 0.0f };
	float3 cell3 = { 0.0f, 0.0f, 3.0f };

	float4 atom1 = { 0.0f, 0.0f, 0.0f, 0.0f };
	float4 atom2 = { 2.0f / 3.0f, 1.0f / 3.0f, 0.5f, 1.0f };

	float4 atoms[2];

	atoms[0] = atom1;
	atoms[1] = atom2;

	int3 dim = { size, size, size };
	float4* sites = (float4*) malloc(size * size * size * sizeof(float4) * 2);
	LatticeBuilder(sites, cell1, cell2, cell3, atoms, 2, dim);

	return sites;
}

#define _LINE "------------------------------------------------------------------------------"

static void writeToFileXyz(FILE* f, float4* table, int3 size, int n_base) {

	fprintf(f, "%d\n \n", size.x * size.y * size.z * n_base);

	for (int i = 0; i < size.x * size.y * size.z * n_base; ++i) {
		int atom = static_cast<int>(table[i].w);
		if (atom == 0)
			fprintf(f, "%s %lf %lf %lf\n", "Ni", table[i].x, table[i].y,
					table[i].z);
		if (atom == 1)
			fprintf(f, "%s %lf %lf %lf\n", "Al", table[i].x, table[i].y,
					table[i].z);
	}

}

static const int3 SIZE = { 64, 64, 16 };

void test() {
	std::cout << _LINE << std::endl;
	std::cout << "Lattices generation test - only CPU supported" << std::endl;
	std::cout << _LINE << std::endl;

	FILE* file = fopen("b2.xyz", "w");
	writeToFileXyz(file, cubicB2(SIZE.x), SIZE, 2);

	if (fclose(file)) {
		std::cout << "Error closing b2.xyz file...exiting" << std::endl;
		exit(0);
	}

	file = fopen("l10.xyz", "w");
	writeToFileXyz(file, cubicL10(SIZE.x), SIZE, 4);

	if (fclose(file)) {
		std::cout << "Error closing l10.xyz file...exiting" << std::endl;
		exit(0);
	}

	file = fopen("hexagonal.xyz", "w");
	writeToFileXyz(file, hexagonal(SIZE.x), SIZE, 2);

	if (fclose(file)) {
		std::cout << "Error closing hexagonal.xyz file...exiting" << std::endl;
		exit(0);
	}

	std::cout << "Generated: l10.xyz, b2.xyz, hexagonal.xyz" << std::endl;
	std::cout << _LINE << std::endl;

}

}
}
