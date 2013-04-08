#include <iostream>
#include <stdio.h>
#include "../../macros/errors.h"
#include "../../headers/kernels.cuh"
#include "../../headers/cpukernels.h"
#include "../../headers/utils.h"

#define NSITES 8192
#define NEIGS 8
#define DIM 16

static void b2Builder(float4* sites, float3 b1, float3 b2, float3 b3, float4 A,
		float4 B) {

	for (int i = 0; i < DIM; ++i) {
		for (int j = 0; j < DIM; ++j) {
			for (int k = 0; k < DIM; ++k) {

				int id = i * DIM * DIM + j * DIM + k;

				float4 sa = { (b1.x * A.x + b2.x * A.y + b3.x * A.z) + i * b1.x
						+ j * b2.x + k * b3.x, (b2.x * A.y + b2.y * A.y
								+ b2.z * A.y) + i * b1.y + j * b2.y + k * b3.y, (b3.x
										* A.z + b3.y * A.z + b3.z * A.z) + i * b1.z + j * b2.z
										+ k * b3.z };
				sa.w = 1.0f;

				float4 sb = { (b1.x * B.x + b2.x * B.y + b3.x * B.z) + i * b1.x
						+ j * b2.x + k * b3.x, (b2.x * B.y + b2.y * B.y
								+ b2.z * B.y) + i * b1.y + j * b2.y + k * b3.y, (b3.x
										* B.z + b3.y * B.z + b3.z * B.z) + i * b1.z + j * b2.z
										+ k * b3.z };

				sb.w = 2.0f;

				sites[id * 2] = sa;
				sites[id * 2 + 1] = sb;

			}
		}
	}
}

void show(float4* sites) {

	std::cout << NSITES << std::endl;

	for (int i = 0; i < NSITES; ++i) {
		int atom = (int) sites[i].w;

		if (atom == 1)
			std::cout << "Ni " << " " << sites[i].x << " " << sites[i].y << " "
			<< sites[i].z << std::endl;
		if (atom == 2)
			std::cout << "Al " << " " << sites[i].x << " " << sites[i].y << " "
			<< sites[i].z << std::endl;
	}

}

void showSite(float4& site) {

	if (static_cast<int>(site.w) == 1)
		std::cout << "Ni" << " " << site.x << " " << site.y << " " << site.z
		<< " " << std::endl;
	else
		std::cout << "Al" << " " << site.x << " " << site.y << " " << site.z
		<< " " << std::endl;

}

void showNeigs(float4* sites, int4* neigs, int z, int n) {

	for (int i = 0; i < z; ++i) {
		showSite(sites[neigs[n * z + i].w]);
	}

}

namespace tests {
namespace neigbours {

void test() {

	float4* h_sites;
	int4* h_neigs;

	int devCount;
	CHECK_ERROR(cudaGetDeviceCount(&devCount));

	utils::CudaTimer timer;
	utils::CpuTimer timer_cpu;
	utils::CpuTimer timer_cpu1;

	float cudaTime;

	float3 base1 = { 2.86f, 0.0f, 0.0f };
	float3 base2 = { 0.0f, 2.86f, 0.0f };
	float3 base3 = { 0.0f, 0.0f, 2.86f };

	float4 siteA = { 0.0f, 0.0f, 0.0f, 1.0f };
	float4 siteB = { 0.5f, 0.5f, 0.5f, 2.0f };

	int3 dimensions = { DIM, DIM, DIM };

	float radius = 2.5f;

	h_sites = (float4*) malloc(sizeof(float4) * NSITES);
	h_neigs = (int4*) malloc(sizeof(int4) * NSITES * NEIGS);

	timer_cpu.start();
	b2Builder(h_sites, base1, base2, base3, siteA, siteB);
	timer_cpu.stop();

	std::cout << "Building lattice time:" << timer_cpu.elapsed() << " ms"
			<< std::endl;

	timer_cpu.reset();
	//show(h_sites);

	float4* d_sites;
	int4* d_neigs;

	if (devCount > 0) {
		CHECK_ERROR(cudaMalloc((void**)&d_sites,sizeof(float4) * NSITES));
		CHECK_ERROR(
				cudaMalloc((void**)&d_neigs,sizeof(float4) * NSITES * NEIGS));

		CHECK_ERROR(
				cudaMemcpy(d_sites,h_sites,sizeof(float4)*NSITES,cudaMemcpyHostToDevice));

		timer.start();
		findNeigboursXyz<<<16, 512>>>(d_sites, d_neigs, base1, base2, base3,
				dimensions, radius, 8, 0, NSITES);
		cudaDeviceSynchronize();
		timer.stop();

		cudaTime = timer.elapsed();
		timer.reset();
	} else
		std::cout << "No device found: only host reference code will be tested"
		<< std::endl;

	timer_cpu1.start();
	findNeigboursXyzCpu(h_sites, h_neigs, base1, base2, base3, dimensions,
			radius, 8, 0, NSITES);
	timer_cpu1.stop();

	float cpuTime = timer_cpu1.elapsed();

	if (devCount) {
		CHECK_ERROR(
				cudaMemcpy(h_neigs,d_neigs,sizeof(int4)*NSITES*NEIGS,cudaMemcpyDeviceToHost));
		//showNeigs(h_sites,h_neigs,8,0);
		free(h_sites);
		free(h_neigs);

		std::cout << "Cuda time:" << cudaTime << " ms " << std::endl
				<< "Cpu time:" << cpuTime << " ms" << std::endl;

		CHECK_ERROR(cudaFree(d_sites));
		CHECK_ERROR(cudaFree(d_neigs));
	} else
		std::cout << "Cpu time:" << cpuTime << " ms" << std::endl;

}

}
}
