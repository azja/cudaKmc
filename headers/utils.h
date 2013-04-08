/*
 * utils.h
 *
 *  Created on: 23-03-2013
 *      Author: Andrzej Biborski
 */

#ifndef UTILS_H_
#define UTILS_H_
#include "anames.h"
#include "../environment/env.cuh"
#include "../macros/errors.h"
#include <stdio.h>
#include <sys/time.h>
#include <iostream>

namespace utils {

struct GetXFromFloat4 {

	__host__ __device__ float operator()(const float4& f) {
		return f.x;
	}
};

struct GetYFromFloat4 {

	__host__ __device__ float operator()(const float4& f) {
		return f.y;
	}
};

struct GetZFromFloat4 {

	__host__ __device__ float operator()(const float4& f) {
		return f.z;
	}
};

struct GetWFromFloat4 {

	__host__ __device__ float operator()(const float4& f) {
		return f.w;
	}
};

struct IsAtomPredicate {
private:
	const definitions::Atom _atom;

public:
	IsAtomPredicate(definitions::Atom atom) :
		_atom(atom) {

	}

	__host__ __device__ bool operator()(const float4& atom) const {
		return static_cast<uint>(atom.w) == static_cast<uint>(_atom);
	}
};

/*
 * Timers
 */

class Timer {

protected:
	virtual void print() const = 0;

public:
	virtual void start() = 0;
	virtual void stop() = 0;
	virtual int elapsed() = 0;
	virtual void reset() = 0;

	friend std::ostream& operator<<(std::ostream& stream, const Timer& timer) {
		timer.print();
		return stream;
	}

	virtual ~Timer() {
	}
	;

};

class CudaTimer: public Timer {
	cudaEvent_t start_event;
	cudaEvent_t end_event;
	float time;
public:

	void start() {
		cudaEventCreate(&start_event);
		cudaEventCreate(&end_event);
		cudaEventRecord(start_event, 0);
	}
	void stop() {
		cudaEventRecord(end_event, 0);
		cudaEventSynchronize(end_event);
		elapsed();
	}
	int elapsed() {
		cudaEventElapsedTime(&time, start_event, end_event);
		return static_cast<int>(time);
	}
	void reset() {
		cudaEventDestroy(start_event);
		cudaEventDestroy(end_event);
		time = 0.0f;
	}

protected:

	void print() const {
		std::cout << "Time: " << time << "ms" << std::endl;
	}

};

class CpuTimer: public Timer {

	int time;
	timeval start_t;
	timeval stop_t;

	void resetTimeVal(timeval& t) {
		t.tv_sec = 0;
		t.tv_usec = 0;
	}

public:

	void start() {
		if (gettimeofday(&start_t, NULL) < 0)
			std::cout << "Error" << std::endl;
	}

	void stop() {
		gettimeofday(&stop_t, NULL);
		elapsed();
	}

	int elapsed() {
		time = (stop_t.tv_sec * 1000 + stop_t.tv_usec / 1000
				- start_t.tv_usec / 1000 - start_t.tv_sec * 1000);

		return time;
	}

	void reset() {
		resetTimeVal(start_t);
		resetTimeVal(stop_t);
		time = 0;
	}

protected:

	void print() const {
		std::cout << "Time: " << time << "ms" << std::endl;
	}

};



/* Timers */


int MaxNumberOfThreads( cudakmc::verbosity v = cudakmc::silent, int devNumber = 0 );

}
#endif /* UTILS_H_ */

