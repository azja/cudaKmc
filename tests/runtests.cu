/*
 * runtests.cu
 *
 *  Created on: 29-03-2013
 *      Author: A. Biborski
 */

#include "../tests/kernels/findntest.h"
#include "../tests/kernels/transparams.h"
#include "../tests/energy/energytest.h"
#include "../tests/builder/buildlattice.h"
#include "../tests/helpers/shared_tests.cuh"
#include "../tests/kernels/simulationrta.h"
int main(int argc, char* argv[]) {

int sample_size = atoi(argv[1]);
//	tests::neigbours::test();
//	tests::energy::ising::test();
//	tests::lattice::test();
//	tests::transition::test(sample_size);
//	tests::helpers::test(sample_size);
	tests::simulations::testRta(sample_size);
return 0;
}

