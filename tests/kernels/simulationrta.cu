#include "simulationrta.h"

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include "../../macros/errors.h"
#include "../../headers/kernels.cuh"
#include "../../headers/cpukernels.h"
#include "../../headers/utils.h"
#include "../../headers/siminput.cuh"
#include "../../headers/isinge.cuh"
#include "../../headers/algorithms.cuh"
#include "../../headers/simulation.cuh"
#include "../../headers/schedule.cuh"
#include "../../headers/writer.cuh"
#include "../builder/buildlattice.h"

namespace tests {
namespace simulations {

float stdrand() {

    return static_cast<float>(rand()) / RAND_MAX;
}

float h_energies[9] = { -0.12f, -0.125f, 0.04f, -0.125f, -0.05f, -0.04f, 0.04f, -0.04f,
        0.0f };
float h_saddle[3] = { 0.5f, 1.0f, 0.0f };



void testRta(int sample_size) {

    int CUBE_SIZE = sample_size;

    int z = 8;
    int z_t = z;
    int n_v = 0.037 * CUBE_SIZE * CUBE_SIZE * CUBE_SIZE * 2;
    int atoms_n = 3;
    int N = CUBE_SIZE * CUBE_SIZE * CUBE_SIZE * 2;
    int3 dims = { CUBE_SIZE, CUBE_SIZE, CUBE_SIZE };
    float radius = 2.5f;
    int BLOCK_SIZE = 256;
    definitions::Atom simulationAtoms[atoms_n];// ={definitions::Ni,definitions::Al,definitions::vacancy};

    simulationAtoms[0] = definitions::vacancy;
    simulationAtoms[1] = definitions::Ni;
    simulationAtoms[2] = definitions::Al;
    float4* h_sites = tests::lattice::cubicB2(CUBE_SIZE);

    /*
    for (int i = 0; i < n_v; ++i) {
        h_sites[i].w = 2;
    }
    */

    int *h_vacancies = (int*)malloc(n_v * sizeof(int));
    int cntr = 0;
    while(cntr < n_v) {
        int ind = (float(rand())/RAND_MAX) * (N -1);

        if(static_cast<int>(h_sites[ind].w)!= 2) {
            h_sites[ind].w = 2.0f;
            h_vacancies[cntr] = ind;
            cntr++;
        }
    }
    float4* d_sites;
    CHECK_ERROR(cudaMalloc((void**)&d_sites,sizeof(float4) * N));
    CHECK_ERROR(
            cudaMemcpy(d_sites,h_sites, sizeof(float4) * N,cudaMemcpyHostToDevice));

    int4* d_neigbours;
    CHECK_ERROR(cudaMalloc((void**)&d_neigbours,sizeof(int4) * N * z));
    utils::CudaTimer timer;
    timer.start();
    findNeigboursXyz<<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE,
            BLOCK_SIZE * sizeof(float4)>>>(d_sites, d_neigbours,
                    tests::lattice::b2cell1, tests::lattice::b2cell2,
                    tests::lattice::b2cell3, dims, radius, z, 0, N);
    cudaDeviceSynchronize();
    timer.stop();

    std::cout << "Neigbours list generated in:" << timer << std::endl;
    timer.reset();

    float* d_transitions;
    CHECK_ERROR(cudaMalloc((void**)&d_transitions,sizeof(float) * n_v * z_t));

    typedef SimulationDeviceInput<float, kmcenergy::ising::PairEnergy_I,
            algorithms::ConstantSaddleEnergy, algorithms::ExchangeEnergy> isingDeviceInput;

    isingDeviceInput input;

    float* d_energies;
    CHECK_ERROR(
            cudaMalloc((void**)&d_energies,sizeof(float) * atoms_n * atoms_n));
    CHECK_ERROR(
            cudaMemcpy(d_energies,h_energies, sizeof(float) * atoms_n * atoms_n,cudaMemcpyHostToDevice));

    float* d_saddle;
    CHECK_ERROR(cudaMalloc((void**)&d_saddle,sizeof(float) * atoms_n));
    CHECK_ERROR(
            cudaMemcpy(d_saddle,h_saddle, sizeof(float) * atoms_n,cudaMemcpyHostToDevice));

    int* d_vacancies;
    CHECK_ERROR(cudaMalloc((void**)&d_vacancies,sizeof(int) * n_v));
    CHECK_ERROR(cudaMemcpy(d_vacancies,h_vacancies,sizeof(int)*n_v,cudaMemcpyHostToDevice));

    input.N = N;
    input.atoms_n = atoms_n;
    input.n_v = n_v;
    input.z = z;
    input.z_t = z_t;

    input.neigbours = d_neigbours;
    input.sites = d_sites;
    input.vacancies = d_vacancies;
    input.transitions = d_transitions;
    input.jumpingNeigbours = d_neigbours;
    input.exchangeEnergyCalculation = d_energies;
    input.saddleEnergyCalculation = d_saddle;

    isingDeviceInput h_input;

    free(h_sites);
    h_sites = tests::lattice::cubicB2(CUBE_SIZE); //Regeneration for statistics
    h_input.sites = h_sites;

    schedule::Schedule schedule(10000, 10000);
     utils::AtomMapper mapper(&simulationAtoms[0],atoms_n);
    utils::TestWriterCopyFromDeviceXyz<isingDeviceInput> writer(input,h_input,
            "test_simulation",mapper);
    simulation::RtaVacancyBarrierSimulationDevice<isingDeviceInput> rtaSimulation(
            input, schedule, writer, stdrand, 16);
    rtaSimulation.run();

    free(h_sites);

}

}
}
