/*
 * buildlattice.h
 *
 *  Created on: 09-04-2013
 *      Author: biborski
 */

#ifndef BUILDLATTICE_H_
#define BUILDLATTICE_H_
namespace tests {
 namespace lattice {

  const  float3 b2cell1 = { 2.86f, 0.0f, 0.0f };
  const  float3 b2cell2 = { 0.0f, 2.86f, 0.0f };
  const  float3 b2cell3 = { 0.0f, 0.0f, 2.86f };



   float4* cubicL10(int size);
   float4* cubicB2(int size) ;
   float4* hexagonal(int size);
   void test();
 }
}




#endif /* BUILDLATTICE_H_ */
