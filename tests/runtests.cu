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
#include <iostream>

const char* neigbours = "-neigbours";
const char* energy_ising = "-energy_ising";
const char* lattice_generator = "-lattice_generator";
const char* transitions = "-transitions";
const char* simulations_rta_vacancy = "-sim_rta_vacancy";

int main(int argc, char* argv[]) {


char* selection = "not given";
if(argc > 2)
selection = argv[1];

int sample_size = atoi(argv[2]);
if(!sample_size)
    sample_size = 16;

if (!strcmp(selection, neigbours) )
  tests::neigbours::test();
else
if (!strcmp(selection, energy_ising) )
  tests::energy::ising::test();
else
if (!strcmp(selection, lattice_generator) )
  tests::lattice::test();
else
if (!strcmp(selection, transitions) )
  tests::transition::test(sample_size);
else
if (!strcmp(selection, simulations_rta_vacancy) )
  tests::simulations::testRta(sample_size);
else {
  std::cout<<"Options are:"<<std::endl<<neigbours<<std::endl;
  std::cout<<energy_ising<<std::endl;
  std::cout<<neigbours<<std::endl;
  std::cout<<lattice_generator<<std::endl;
  std::cout<<transitions<<" [integer::problem_size (=16 default)]  "<<std::endl;
  std::cout<<simulations_rta_vacancy<<"[integer::problem_size (=16 default)] "<<std::endl;
 }

return 0;
}

