/*
 /*
 * simulation.cuh
 *
 * Created on: 27-04-2013
 * Author: biborski
 */

#ifndef SIMULATION_CUH_
#define SIMULATION_CUH_
#include "siminput.cuh"
#include "kernels.cuh"
#include "schedule.cuh"
#include "utils.h"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <thrust/transform_scan.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

namespace simulation {

struct Simulation {
    virtual void run() = 0;
    virtual ~Simulation() {
    }

};

/*
 * Residence Time Algorithm (RTA) for Kinetic Monte Carlo
 */

template<typename SimulationInput, int BlockSize = 256>
struct RtaVacancyBarrierSimulationDevice: public Simulation {

public:
    RtaVacancyBarrierSimulationDevice(SimulationInput input,
            schedule::Schedule& schedule,
            utils::Writer<SimulationInput> &writer, float (*rand)(),
            float beta = 0) :
            d_input(input), _schedule(schedule), _writer(writer), _rand(rand), _beta(
                    beta) {

    }

    virtual void run() {

        d_input.ToHost(h_input);

        float time = 0.0f;

        float reducedTransitions = 0.0f;
        int toSiteIndex;
        int fromSiteIndex;

        int indexMax = d_input.z_t * d_input.n_v;

        for (int i = 0; i < _schedule.getNSubSteps(); ++i) {

            _writer.Prepare();

            if (!_writer.StartWriteThread(time)) {
                std::cout << "Error in writing thread creation - exiting"
                        << std::endl;
                exit(0);
            }
#ifdef DEBUG
            utils::CudaTimer timer;
            timer.start();
#endif
            for (int j = 0; j < _schedule.getNStep(); ++j) {

                calculateTransitionParameterNoId<<<
                ((d_input.n_v) - 1) / BlockSize + 1, BlockSize>>>(
                        d_input,_beta);
                cudaDeviceSynchronize();
                CHECK_ERROR(cudaGetLastError());
                thrust::device_ptr<float> ptr1 = thrust::device_pointer_cast(
                        d_input.transitions);

                /*
                 * Reduce transitions and update time
                 */

                thrust::device_ptr<float> d_transitions =
                thrust::device_pointer_cast(d_input.transitions);

                thrust::inclusive_scan(d_transitions,d_transitions + d_input.n_v * d_input.z_t,
                        d_transitions);

                CHECK_ERROR(
                        cudaMemcpy(&reducedTransitions,d_input.transitions + (d_input.n_v *d_input.z_t -1),sizeof(float),cudaMemcpyDeviceToHost));

                time += -log(_rand()) / reducedTransitions;

                /*
                 * Find wi < R <= w_i+1 item
                 */

                float tot = reducedTransitions * _rand();

                int index = thrust::transform_reduce(d_transitions,
                        d_transitions + d_input.n_v * d_input.z_t,utils::IfLessReturnIntUnary<float>(tot),0,
                        thrust::plus<int>());

                if(index == d_input.z_t * d_input.n_v)
                    std::cout<<"----------------index = "<<index<<"-----------------------"<<std::endl;
                /*
                 * Exchange atoms
                 */

                index = index < indexMax ? index : indexMax - 1;

                int localizeFactor = (index ) / (h_input.z_t);
                toSiteIndex = h_input.vacancies[localizeFactor];
                int offset = toSiteIndex * d_input.z_t;

                fromSiteIndex = h_input.jumpingNeigbours[offset
                + index % h_input.z_t].w;

                float4 sourceSite;
                CHECK_ERROR(cudaMemcpy(&sourceSite,d_input.sites + fromSiteIndex,sizeof(float4),cudaMemcpyDeviceToHost));
                if(static_cast<int>(sourceSite.w) != 2) {

                    exchangeFloat4w<<<((d_input.N) - 1) / BlockSize + 1, BlockSize>>>(
                            fromSiteIndex, toSiteIndex, d_input.sites, d_input.N);
                    cudaDeviceSynchronize();

                    set1dArrayElement<<<((d_input.n_v) - 1) / BlockSize + 1,
                    BlockSize>>>((index) / (h_input.z_t), fromSiteIndex,
                            d_input.vacancies, d_input.n_v);

                    cudaDeviceSynchronize();

                    h_input.vacancies[(index) / (h_input.z_t)] = fromSiteIndex;
                }

            }
            _writer.WaitForWriteThreadToExit();
#ifdef DEBUG
            timer.stop();
            std::cout<<"Inner loop "<<_schedule.getNStep()<<" iterations: "<<timer<<std::endl;
            timer.reset();
#endif
        }
    }
    virtual ~RtaVacancyBarrierSimulationDevice() {
        d_input.FreeToDevice();
    }

protected:

    SimulationInput h_input;
    SimulationInput d_input;
    schedule::Schedule& _schedule;
    utils::Writer<SimulationInput>& _writer;
    float (*_rand)();
    float _beta;

};

}

#endif /* SIMULATION_CUH_ */
