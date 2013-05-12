/*
 * writer.cuh
 *
 *  Created on: 04-05-2013
 *      Author: biborski
 */
#ifndef WRITER_CUH_
#define WRITER_CUH_

#include <pthread.h>
#include <fstream>
#include <string>
#include <sstream>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include "utils.h"
#include "anames.h"

namespace utils{

template<typename Input,typename Measure = float>
struct Writer {
public:


    virtual ~Writer(){};

    /*
     * Measure m is time or MC step
     */
    bool StartWriteThread(const Measure& m)
    {
        _measure = m;
        return (pthread_create(&_thread, NULL, InternalThreadEntryFunc, this) == 0);
    }

    void WaitForWriteThreadToExit()
    {
        (void) pthread_join(_thread, NULL);
    }

    virtual void Prepare() = 0;


protected:
    Measure _measure;
    const utils::AtomMapper& _mapper;
    Writer(const utils::AtomMapper& mapper):_mapper(mapper){}
private:

    pthread_t _thread;

    virtual void PerformWriteAction() = 0;

    static void* InternalThreadEntryFunc(void* arg){
        static_cast<Writer*>(arg)->PerformWriteAction();
        return NULL;
        }
    };



    template <class In>
    struct TestWriterCopyFromDeviceXyz:public Writer<In>{

        TestWriterCopyFromDeviceXyz(const In& data,const In& compare,const char* filePrefix,const utils::AtomMapper& mapper):
                 Writer<In>(mapper),
                _filePrefix(filePrefix),
                _sample(data),
                _compare(compare),
                _counter(0) {

        }

        virtual void Prepare() {
            h_sites = (float4*)malloc(sizeof(float4)*_sample.N);
            CHECK_ERROR(cudaMemcpy(h_sites,_sample.sites, _sample.N * sizeof(float4),cudaMemcpyDeviceToHost));
        }

        virtual ~TestWriterCopyFromDeviceXyz() {};

    private:
        const char* _filePrefix;
        int _counter;
        float4* h_sites;
        const In _sample;
        const In _compare;





        virtual void PerformWriteAction() {
            std::ofstream output;         //_schedule.actions(i,time,d_input,h_input);
            std::stringstream ss;
            ss<<_counter;
            std::string postfix = ss.str();
            std::string prefix = std::string(_filePrefix);
            std::string fileName = prefix + postfix + ".xyz";
          /*
            output.open (fileName.c_str(), std::ofstream::out);
            output<<_sample.N<<std::endl<<std::endl;


            for(int i = 0; i < _sample.N ; ++i) {
                float4 site = h_sites[i];
                if(site.w == 0)
                    output<<"Na "<<site.x<<" "<<site.y<<" "<<site.z<<" "<<std::endl;
                if(site.w == 1)
                    output<<"Al "<<site.x<<" "<<site.y<<" "<<site.z<<" "<<std::endl;
                if(site.w == 2)
                    output<<"H "<<site.x<<" "<<site.y<<" "<<site.z<<" "<<std::endl;

            }
            ++_counter;
            output.close();
*/
            std::ofstream order_output;
            fileName = prefix + "STAT" + ".dat";
            order_output.open (fileName.c_str(), std::ofstream::app);

            int N_A_A = thrust::inner_product(h_sites,h_sites + _sample.N,_compare.sites,0,thrust::plus<int>(),ReturnOneIfTheSameAtom(static_cast<definitions::Atom>(0)));
            int N_B_B = thrust::inner_product(h_sites,h_sites + _sample.N,_compare.sites,0,thrust::plus<int>(),ReturnOneIfTheSameAtom(static_cast<definitions::Atom>(1)));


            order_output<<Writer<In>::_measure<<" "<<N_A_A<<" "<<N_B_B<<std::endl;
            order_output.close();
            free(h_sites);


        }
    };

}

#endif /* WRITER_CUH_ */
