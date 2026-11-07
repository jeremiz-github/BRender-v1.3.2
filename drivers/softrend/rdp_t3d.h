/*
 * N64: hand-over of one model group from the BRender geometry stage to the Tiny3D (RSP) renderer.
 * Plain C types only: shared between softrend (producer) and pentprim/rdprend.c (consumer).
 */
#ifndef RDP_T3D_H
#define RDP_T3D_H
#ifdef N64

#include <stdint.h>

#define T3DG_MAXLIGHTS 3

typedef struct {
    void* key_model;           /* v11model* */
    int group;
    const float* m2v;          /* model->view, br_matrix34 (4x3, row vector convention) */
    const float* v2s;          /* view->screen, br_matrix4 (4x4, row vector convention) */
    const float* pos;          /* 3 floats / vertex */
    const float* uv;           /* 2 floats / vertex */
    const float* nrm;          /* 3 floats / vertex */
    const uint32_t* vcol;      /* br_colour per vertex (alpha byte = prelit intensity index) */
    const uint32_t* fcol;      /* br_colour per face */
    const uint16_t* fv;        /* 3 vertex numbers / face */
    const float* eqn;          /* 4 floats / face (plane equation, normal first) */
    int nverts, nfaces;
    int lighting;              /* surface.lighting */
    int colour_geometry;       /* surface.colour_source == BRT_GEOMETRY */
    int cull;                  /* 1: one sided (cull back faces), 0: both */
    float ci_scale, ci_off;    /* comp_scales/offsets[C_I] */
    float cu_scale, cu_off, cv_scale, cv_off; /* comp_scales/offsets[C_U], [C_V] */
    float ka, kd;
    float ambient;             /* ka * ambient light intensity */
    int nlights;               /* directional lights, view space, unit direction + intensity (already * kd) */
    float ldir[T3DG_MAXLIGHTS][3];
    float lint[T3DG_MAXLIGHTS];
} T3dGroupArgs;

/* returns 1 when the group was drawn by Tiny3D, 0 when the caller must run its normal pipeline */
int RdpT3dGroup(const void* block, const T3dGroupArgs* a);
void RdpT3dInvalidate(const void* v11model);

#endif
#endif
