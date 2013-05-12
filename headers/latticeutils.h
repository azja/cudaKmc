/*
 * latticeutils.h
 *
 *  Created on: 26-04-2013
 *      Author: Andrzej Biborski
 *  Only CPU
 */
#ifndef LATTICEUTILS_H_
#define LATTICEUTILS_H_

#include <stdlib.h>
#include "anames.h"

namespace utils {
namespace lattice {

typedef float (*floatFunc)();

/*
 * firstKind  - kind of atom to be replaced
 * secondKind - kind of atom which replace atom of "firstKind" kind
 * n          - size of sites[]
 * k          - number of atoms to be replaced: k <= n
 * sites[n]   - sites array
 * output[k]  - stores indices of replaced atoms
 * generator  - random number generator (0,1], if not given, standard
 */


void replaceAtoms(definitions::Atom firstKind,definitions::Atom secondKind,
        int n,int k, float4* const sites,int * const output, floatFunc generator = 0);
}
}
#endif /* LATTICEUTILS_H_ */
