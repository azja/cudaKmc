/*
 * sampled.h
 *
 *  Created on: 23-03-2013
 *      Author: Andrzej Biborski
 */

#ifndef SAMPLED_H_
#define SAMPLED_H_
#include "thrust/functional.h"
#include "sample.h"
#include "nbuilder.h"

namespace samples {

class SampleDevice: public Sample {

	virtual void calculateNeigbours();
	virtual void allocate(uint N, uint n_v, uint z, uint z_t);

public:

	SampleDevice(uint N, float4 * const sites, nbuilder::NeigbourFinder* finder) :
		Sample(N, finder) {
	}

	SampleDevice(uint N, nbuilder::NeigbourFinder* finder) :
		Sample(N, finder) {
	}

	virtual uint getNumberOf(definitions::Atom i) const;

	~SampleDevice();
};

}/* samples*/

#endif /* SAMPLED_H_ */
