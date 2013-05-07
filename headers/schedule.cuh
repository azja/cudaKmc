/*
 * schedule.cuh
 *
 *  Created on: 02-05-2013
 *      Author: biborski
 */

#ifndef SCHEDULE_CUH_
#define SCHEDULE_CUH_
#include <pthread.h>
#include <fstream>
#include <sstream>
#include "writer.cuh"

namespace schedule {

struct Schedule {
public:
    Schedule(int step, int substeps):_step(step),_substeps(substeps) {

    }

    int  getNStep() const {
        return _step;
    }

    int getNSubSteps() const {
        return _substeps;
    }

    /*
     * Sites are on device!!!
     */


    //virtual void actions(int step, float time = 0.0f) = 0;


    virtual ~Schedule() {}
protected:
    int _step;
    int _substeps;

};
/*
template <typename SimulationInput>
struct ScheduleWriteXyzFake:public Schedule<SimulationInput> {
public:
    ScheduleWriteXyzFake(int step,int substeps, const char* filePrefix, const Writer& wirter) : Schedule<SimulationInput>(step,substeps,writer),
        _filePrefix(filePrefix) {
        _counter = 0;

    }

    virtual void actions(int step, float time,const  SimulationInput& sample,const SimulationInput& comparer) {
        writeToFile(sample);
        ++_counter;
    }

    virtual ~ScheduleWriteXyzFake() {}

private:
    const char* _filePrefix;
    size_t _counter;

    virtual void writeToFile(const SimulationInput& sample) const {

        std::ofstream output;
        std::stringstream ss;
        ss<<_counter;
        std::string postfix = ss.str();
        std::string prefix = std::string(_filePrefix);
        std::string fileName = prefix + postfix + ".xyz";
        output.open (fileName.c_str(), std::ofstream::out);
        output<<sample.N<<std::endl<<std::endl;

        float4* h_sites = (float4*)malloc(sizeof(float4)*sample.N);


        for(int i = 0; i < sample.N ; ++i) {
            float4 site = h_sites[i];
            if(site.w == 0)
                output<<"Na "<<site.x<<" "<<site.y<<" "<<site.z<<" "<<std::endl;
            if(site.w == 1)
                output<<"Al "<<site.x<<" "<<site.y<<" "<<site.z<<" "<<std::endl;
            if(site.w == 2)
                output<<"H "<<site.x<<" "<<site.y<<" "<<site.z<<" "<<std::endl;

        }

        output.close();
        free(h_sites);

    }

};
 */
/*
 * End of Schedule
 */
//////////////////////////////////////////////////////////////

}

#endif /* SCHEDULE_CUH_ */
