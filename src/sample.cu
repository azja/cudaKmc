#include "../headers/sample.h"

/*
 * Implementation of virtual - template method Sample class
 */

namespace samples {

Sample::Sample(uint N,nbuilder::NeigbourFinder*  nfinder) :
				_N(N),_nFinder(nfinder) {

}

}
