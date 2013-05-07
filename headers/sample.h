/*
 * sample.h
 *
 *  Created on: 22-03-2013
 *      Author: Andrzej Biborski
 */
#include "anames.h"
#include "nbuilder.h"
#ifndef SAMPLE_H_
#define SAMPLE_H_

namespace samples {

class Sample {
private:
    virtual void calculateNeigbours() = 0;
    virtual void allocate(uint N, uint n_v, uint z, uint z_t) = 0;
    nbuilder::NeigbourFinder* _nFinder;

protected:
    uint _N;
    uint _n_v;
    uint _z;
    uint _z_t;

    float4* _sites; //sites[N]
    float* _transitions; //transitions [n_v * z_T]
    int4* _jumpingNeigbours; //jumpingNeigbours[n_v * z_T]
    uint* _vacancies; //vacancies[n_v]
    int4* _neigbours; //neigbours [n_v *z]

    Sample(uint N, nbuilder::NeigbourFinder* nFinder);

public:
    virtual uint getNumberOf(definitions::Atom atom) const = 0;

    void Initialize() {

        _n_v = getNumberOf(definitions::vacancy);
        allocate(_N, _n_v, _z, _z_t);
        calculateNeigbours();
    }

    virtual ~Sample() {
    }

};

}
#endif /* SAMPLE_H_ */
