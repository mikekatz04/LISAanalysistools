#ifndef __ORBITS_VIEW_HPP__
#define __ORBITS_VIEW_HPP__

// OrbitsView — stable-layout POD that downstream sprint wheels consume
// instead of the typed `Orbits*` C++ class. Defined here so that
// GBGPU / BBHx / FEW kernels can read orbit data WITHOUT linking against
// LAT's Detector.cu translation unit and WITHOUT going through pybind11's
// type registry for the Orbits wrapper.
//
// Layout MUST stay in lockstep with class Orbits in Detector.hpp. Any
// field added / reordered / type-changed here OR there:
//   1. Mirror the change on both sides in the same commit.
//   2. Bump LISATOOLS_HEADER_ABI_VERSION in lisatools_header_abi.hpp.
//
// (See "POD-view side-channel" / "OrbitsWrap symbol unification" in the
// sprint reorg plan for rationale.)

#include "gbt_global.h"
#include "lisatools_header_abi.hpp"

// Header-only — no .cu / .cxx counterpart. Compiled freshly into every TU
// that includes this file, on every target backend.

struct OrbitsView {
    double sc_t0;
    double sc_dt;
    int    sc_N;

    double ltt_t0;
    double ltt_dt;
    int    ltt_N;

    // Device pointers (owned by the upstream LAT Orbits instance):
    double *n_arr;
    double *ltt_arr;
    double *x_arr;

    int    nlinks;
    int    nspacecraft;
    double armlength;

    int *links;
    int *sc_r;
    int *sc_e;
};

// make_orbits_view: take any object whose POD layout matches OrbitsView
// (i.e. class Orbits from Detector.hpp) and emit an OrbitsView snapshot.
// Header-only inline so downstream kernels can call it from `__device__`
// code without a translation-unit dependency.
//
// Templated so that this header does NOT depend on Detector.hpp -- LAT
// internal code passes class Orbits, downstream code can pass anything
// that happens to layout-match.
template <typename OrbitsLike>
CUDA_CALLABLE_MEMBER
OrbitsView make_orbits_view(const OrbitsLike *o)
{
    OrbitsView v;
    v.sc_t0       = o->sc_t0;
    v.sc_dt       = o->sc_dt;
    v.sc_N        = o->sc_N;
    v.ltt_t0      = o->ltt_t0;
    v.ltt_dt      = o->ltt_dt;
    v.ltt_N       = o->ltt_N;
    v.n_arr       = o->n_arr;
    v.ltt_arr     = o->ltt_arr;
    v.x_arr       = o->x_arr;
    v.nlinks      = o->nlinks;
    v.nspacecraft = o->nspacecraft;
    v.armlength   = o->armlength;
    v.links       = o->links;
    v.sc_r        = o->sc_r;
    v.sc_e        = o->sc_e;
    return v;
}

#endif // __ORBITS_VIEW_HPP__
