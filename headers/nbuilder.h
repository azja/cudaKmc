/*
 * Responsible for finding neigbours and building the lists of them for each site
 */
#ifndef NBUILDER_H_
#define NBUILDER_H_

#include <thrust/device_vector.h>

namespace nbuilder {

struct NeigbourFinder {

    virtual void build(const float4 * const sites, uint* const neigbours) = 0;
    virtual void load(const float4 * const sites, uint* const neigbours) = 0;

    virtual ~NeigbourFinder() {
    }
    ;
};

struct XyzNeigbourFinder: public NeigbourFinder {

    XyzNeigbourFinder(const float3& base1, const float3& base2,
            const float3& base3, const float4* const it, const int3& dimensions,
            float radius) {

    }

    XyzNeigbourFinder() {
    }
    void build(const float4 * const sites, uint* const neigbours) {
    }
    ;
    void load(const float4 * const sites, uint* const neigbours) {
    }
    ;

};

}
#endif /* NBUILDER_H_ */
