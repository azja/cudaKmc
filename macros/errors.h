/*
 * errors.h
 *
 *  Created on: 23-03-2013
 *      Author: biborski
 */

#ifndef ERRORS_H_
#define ERRORS_H_

#define CHECK_ERROR(func) do { \
		if( func!=cudaSuccess )\
		printf("Cuda error in %s: %s\n",#func, cudaGetErrorString(func)); \
} while(0)

#endif /* ERRORS_H_ */
