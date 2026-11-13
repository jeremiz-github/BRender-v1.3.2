/*
 * N64: Tiny3D back-end for BRender model groups.
 *
 * rdprend.c (RdpT3dGroup) prepares the material (texture / RDP mode) and the per-vertex colours, this file
 * turns a model group into Tiny3D vertex batches (cached per model+group) and draws them: the RSP does the
 * transform / clip / projection / fog / triangle setup instead of the CPU.
 *
 * BRender's clip->screen mapping is folded into the model matrix so the screen position, the depth
 * (z01 = (sz + 32767) / 65534, identical to rdprend.c's fill_vertex) and the fog match the CPU path:
 *     total clip = P_viewport * M   with   M = P_viewport^-1 * C   =>   total clip = C
 * where C = (model->clip) with the viewport scale/offset applied.
 *
 * Measured on the console (t3dtest probe): Tiny3D writes z01 = (ndc_z + 1) / 2, ndc_z = clip_z / clip_w.
 */
#ifdef N64
#include <libdragon.h>
#include <t3d/t3d.h>
#include <t3d/t3dmath.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../softrend/rdp_t3d.h"

extern int N64_Rdp_T3dRect(int* ox, int* oy, int* w, int* h);
extern void N64_Rdp_T3dForgetState(void);
extern void N64_Rdp_T3dCovered(int tris);
extern int gN64_fog_on;
extern float gN64_fog_z0, gN64_fog_k;

#define SCR_W 320
#define SCR_H 240
#define MAXB_VERTS 64              /* vertices per batch (Tiny3D cache holds 70) */
#define NCACHE 192
#define BUDGET (256 * 1024)        /* bytes of cached vertex data (the 8 MB heap is full at race start: v31 ran out of memory with 1.25 MB) */
#define NMAT 128

typedef struct {
    uint8_t nslots, ntris;
    T3DVertPacked* vb;
    uint16_t* src;                 /* source vertex per slot */
    uint8_t* tri;                  /* 3 slots per triangle */
} t3d_batch;

typedef struct {
    const void* model;
    int group;
    int used, bad;
    int nverts, nfaces, nbatches;
    float S;                       /* model units -> s16 */
    float uvkey[4];                /* cu/cv scale/off * texture factor used to build the UVs */
    float ubase, vbase;
    unsigned colsum;               /* checksum of the colours currently stored */
    t3d_batch* b;
    int bytes;
} t3d_group;

static t3d_group cache[NCACHE];
static int cache_used_bytes;
static int t3d_ready;
static T3DViewport vp;
static T3DMat4FP* mats;
static int mat_next;
static struct { int groups, tris, fallback_bad, fallback_budget, builds, bad_uv, bad_mat, bad_build, rej[8]; } stat;
void T3dCountReject(int why) { stat.rej[why & 7]++; }

int T3dInit(void) {
    if (t3d_ready) return 1;
    t3d_init((T3DInitParams){});
    vp = t3d_viewport_create();
    t3d_viewport_set_area(&vp, 0, 0, SCR_W, SCR_H);
    mats = malloc_uncached_aligned(16, sizeof(T3DMat4FP) * NMAT);
    if (mats == NULL) return 0;
    t3d_ready = 1;
    fprintf(stderr, "t3d: init ok\n");
    return 1;
}

static void free_group_data(t3d_group* g) {
    int b;
    for (b = 0; b < g->nbatches; b++) {
        free_uncached(g->b[b].vb);
        free(g->b[b].src);
        free(g->b[b].tri);
    }
    free(g->b);
    g->b = NULL;
    g->nbatches = 0;
    cache_used_bytes -= g->bytes;
    g->bytes = 0;
}

void RdpT3dInvalidate(const void* model) {
    int i;
    for (i = 0; i < NCACHE; i++) {
        t3d_group* g = &cache[i];
        if (!g->used || g->model != model) continue;
        free_group_data(g);
        memset(g, 0, sizeof(*g));
    }
}

/* ---- 4x4 helpers (double, row-major r,c) ---- */
static int inv4(const double a[4][4], double o[4][4]) {
    double m[4][8];
    int i, j, k;
    for (i = 0; i < 4; i++)
        for (j = 0; j < 4; j++) {
            m[i][j] = a[i][j];
            m[i][j + 4] = (i == j);
        }
    for (i = 0; i < 4; i++) {
        int p = i;
        double d;
        for (k = i + 1; k < 4; k++)
            if (fabs(m[k][i]) > fabs(m[p][i])) p = k;
        if (fabs(m[p][i]) < 1e-12) return 0;
        if (p != i)
            for (j = 0; j < 8; j++) { double t = m[i][j]; m[i][j] = m[p][j]; m[p][j] = t; }
        d = 1.0 / m[i][i];
        for (j = 0; j < 8; j++) m[i][j] *= d;
        for (k = 0; k < 4; k++) {
            double f;
            if (k == i) continue;
            f = m[k][i];
            for (j = 0; j < 8; j++) m[k][j] -= f * m[i][j];
        }
    }
    for (i = 0; i < 4; i++)
        for (j = 0; j < 4; j++) o[i][j] = m[i][j + 4];
    return 1;
}

static void mul4(const double a[4][4], const double b[4][4], double o[4][4]) {
    int i, j, k;
    for (i = 0; i < 4; i++)
        for (j = 0; j < 4; j++) {
            double s = 0;
            for (k = 0; k < 4; k++) s += a[i][k] * b[k][j];
            o[i][j] = s;
        }
}

/* ---- build the cached batches of a group ---- */
static int build_group(t3d_group* g, const T3dDrawArgs* d) {
    int nv = d->g->nverts, nf = d->g->nfaces, f, v;
    float maxabs = 1e-3f;
    uint32_t* stamp = NULL;
    uint8_t* slot = NULL;
    t3d_batch* batches = NULL;
    int nb = 0, capb = 0, bytes = 0;
    int cur_n = 0, cur_t = 0;
    uint16_t cur_src[MAXB_VERTS];
    uint8_t cur_tri[256 * 3];
    int epoch = 1;
    float umin = 1e30f, vmin = 1e30f, umax = -1e30f, vmax = -1e30f;

    for (v = 0; v < nv; v++) {
        float u = d->textured ? d->g->uv[v * 2] * d->uk_s + d->uk_o : 0.0f, w = d->textured ? d->g->uv[v * 2 + 1] * d->vk_s + d->vk_o : 0.0f;
        int k;
        if (!(u == u) || !(w == w)) return 0;
        if (u < umin) umin = u;
        if (u > umax) umax = u;
        if (w < vmin) vmin = w;
        if (w > vmax) vmax = w;
        for (k = 0; k < 3; k++) {
            float a = fabsf(d->g->pos[v * 3 + k]);
            if (a > maxabs) maxabs = a;
        }
    }
    g->ubase = d->tex_w > 0 ? floorf(umin / d->tex_w) * d->tex_w : 0;
    g->vbase = d->tex_h > 0 ? floorf(vmin / d->tex_h) * d->tex_h : 0;
    if (umax - g->ubase > 1000.0f || vmax - g->vbase > 1000.0f) { stat.bad_uv++; return 0; }   /* UVs hold +-1024 texels (s10.5) */
    /* model units -> s16, capped so the matrix entries stay small (Tiny3D matrices are s15.16) */
    g->S = 30000.0f / maxabs;
    if (g->S > 256.0f) g->S = 256.0f;

    stamp = calloc(nv, sizeof(uint32_t));
    slot = malloc(nv);
    if (!stamp || !slot) goto fail;

    for (f = 0; f <= nf; f++) {
        int need = 0, k;
        uint16_t tv[3] = { 0, 0, 0 };
        if (f < nf) {
            for (k = 0; k < 3; k++) {
                tv[k] = d->g->fv[f * 3 + k];
                if (tv[k] >= nv) goto fail;
            }
        }
        if (f < nf) {
            need = (stamp[tv[0]] != epoch) + (stamp[tv[1]] != epoch && tv[1] != tv[0])
                + (stamp[tv[2]] != epoch && tv[2] != tv[0] && tv[2] != tv[1]);
        }
        if (f == nf || cur_n + need > MAXB_VERTS || cur_t >= 255) {
            if (cur_t > 0) {
                t3d_batch* B;
                int ns = cur_n, i;
                if (nb == capb) {
                    capb = capb ? capb * 2 : 16;
                    batches = realloc(batches, capb * sizeof(t3d_batch));
                    if (!batches) goto fail;
                }
                B = &batches[nb++];
                memset(B, 0, sizeof(*B));
                B->nslots = ns;
                B->ntris = cur_t;
                B->vb = malloc_uncached_aligned(16, sizeof(T3DVertPacked) * ((ns + 1) / 2));
                B->src = malloc(ns * sizeof(uint16_t));
                B->tri = malloc(cur_t * 3);
                if (!B->vb || !B->src || !B->tri) goto fail;
                bytes += sizeof(T3DVertPacked) * ((ns + 1) / 2) + ns * 2 + cur_t * 3 + (int)sizeof(t3d_batch);
                memcpy(B->src, cur_src, ns * sizeof(uint16_t));
                memcpy(B->tri, cur_tri, cur_t * 3);
                for (i = 0; i < ns; i++) {
                    int s2 = cur_src[i];
                    int16_t* p = t3d_vertbuffer_get_pos(B->vb, i);
                    int16_t* uv = t3d_vertbuffer_get_uv(B->vb, i);
                    uint8_t* c = t3d_vertbuffer_get_rgba(B->vb, i);
                    float u = d->textured ? (d->g->uv[s2 * 2] * d->uk_s + d->uk_o - g->ubase) * 32.0f : 0.0f;
                    float w = d->textured ? (d->g->uv[s2 * 2 + 1] * d->vk_s + d->vk_o - g->vbase) * 32.0f : 0.0f;
                    p[0] = (int16_t)lrintf(d->g->pos[s2 * 3] * g->S);
                    p[1] = (int16_t)lrintf(d->g->pos[s2 * 3 + 1] * g->S);
                    p[2] = (int16_t)lrintf(d->g->pos[s2 * 3 + 2] * g->S);
                    uv[0] = (int16_t)lrintf(u);
                    uv[1] = (int16_t)lrintf(w);
                    c[0] = c[1] = c[2] = c[3] = 255;
                }
            }
            if (f == nf) break;
            cur_n = cur_t = 0;
            epoch++;
        }
        for (k = 0; k < 3; k++) {
            if (stamp[tv[k]] != epoch) {
                stamp[tv[k]] = epoch;
                slot[tv[k]] = cur_n;
                cur_src[cur_n++] = tv[k];
            }
            cur_tri[cur_t * 3 + k] = slot[tv[k]];
        }
        cur_t++;
    }
    free(stamp);
    free(slot);
    g->b = batches;
    g->nbatches = nb;
    g->nverts = nv;
    g->nfaces = nf;
    g->bytes = bytes;
    g->colsum = 0xFFFFFFFFu;
    g->uvkey[0] = d->uk_s; g->uvkey[1] = d->uk_o; g->uvkey[2] = d->vk_s; g->uvkey[3] = d->vk_o;
    cache_used_bytes += bytes;
    stat.builds++;
    return 1;
fail:
    free(stamp);
    free(slot);
    {
        int b;
        for (b = 0; b < nb; b++) {
            free_uncached(batches[b].vb);
            free(batches[b].src);
            free(batches[b].tri);
        }
        free(batches);
    }
    return 0;
}

static t3d_group* find_group(const T3dDrawArgs* d) {
    int i, freeslot = -1;
    for (i = 0; i < NCACHE; i++) {
        t3d_group* g = &cache[i];
        if (g->used && g->model == d->g->key_model && g->group == d->g->group) return g;
        if (!g->used && freeslot < 0) freeslot = i;
    }
    if (freeslot < 0) return NULL;
    cache[freeslot].used = 1;
    cache[freeslot].model = d->g->key_model;
    cache[freeslot].group = d->g->group;
    return &cache[freeslot];
}

/* Starts a Tiny3D draw sequence inside the current scene (attached colour + z buffer). Returns 0 when unavailable. */
int T3dBegin(void) {
    int ox, oy, w, h, x0, y0, x1, y1;
    if (!t3d_ready || !N64_Rdp_T3dRect(&ox, &oy, &w, &h)) return 0;
    x0 = ox < 0 ? 0 : ox;
    y0 = oy < 0 ? 0 : oy;
    x1 = ox + w > SCR_W ? SCR_W : ox + w;
    y1 = oy + h > SCR_H ? SCR_H : oy + h;
    if (x1 <= x0 || y1 <= y0) return 0;
    t3d_frame_start();
    rdpq_mode_dithering(DITHER_BAYER_BAYER);
    N64_Rdp_T3dForgetState();   /* t3d_frame_start() replaced the rdpq mode */
    return 1;
}

/* material (texture + rdpq mode) must be set by the caller after T3dBegin(); then T3dDraw() */
int T3dDraw(const T3dDrawArgs* d, const uint32_t* rgba) {
    t3d_group* g;
    int ox, oy, w, h, b, i, ntris = 0;
    double A[4][4], V[4][4], C[4][4], P[4][4], Pi[4][4], Cc[4][4], M[4][4];
    double hw = SCR_W * 0.5, hh = SCR_H * 0.5;
    float near_w, far_w;
    int fog_en = 0;
    float fog_n = 0, fog_f = 1;
    unsigned sum = 0;
    T3DMat4FP* mfp;
    fm_mat4_t mf;
    const T3dGroupArgs* a = d->g;

    g = find_group(d);
    if (g == NULL) { stat.fallback_budget++; return 0; }
    if (g->bad) { stat.fallback_bad++; return 0; }
    if (g->nbatches == 0 || g->uvkey[0] != d->uk_s || g->uvkey[1] != d->uk_o || g->uvkey[2] != d->vk_s || g->uvkey[3] != d->vk_o) {
        if (g->nbatches) {
            free_group_data(g);   /* texture / UV mapping changed: rebuild */
        }
        if (cache_used_bytes > BUDGET || !build_group(g, d)) {
            if (cache_used_bytes <= BUDGET) stat.bad_build++;
            g->bad = 1;
            if (cache_used_bytes > BUDGET) stat.fallback_budget++; else stat.fallback_bad++;
            return 0;
        }
    }

    /* colours (only rewritten when they changed) */
    for (i = 0; i < a->nverts; i++) sum = sum * 31u + rgba[i];
    if (sum != g->colsum) {
        for (b = 0; b < g->nbatches; b++) {
            t3d_batch* B = &g->b[b];
            for (i = 0; i < B->nslots; i++) {
                uint32_t c = rgba[B->src[i]];
                uint8_t* p = t3d_vertbuffer_get_rgba(B->vb, i);
                p[0] = c >> 24; p[1] = c >> 16; p[2] = c >> 8; p[3] = 255;
            }
        }
        g->colsum = sum;
    }

    if (!N64_Rdp_T3dRect(&ox, &oy, &w, &h)) return 0;

    /* ---- matrices ---- */
    {
        const float* m = a->m2v;     /* 4 rows x 3 cols */
        const float* s = a->v2s;     /* 4 x 4 */
        int r, c;
        for (r = 0; r < 4; r++)
            for (c = 0; c < 4; c++) {
                A[r][c] = (c < 3) ? m[r * 3 + c] : (r == 3 ? 1.0 : 0.0);
                V[r][c] = s[r * 4 + c];
            }
        mul4(A, V, C);   /* row-vector convention: clip = [x y z 1] * C */
        /* clip rows in column convention: Cc[j][i] = C[i][j] */
        for (r = 0; r < 4; r++)
            for (c = 0; c < 4; c++) Cc[r][c] = C[c][r];
    }
    {
        double X[4][4], k;
        int c;
        /* x' = sx_scale/hw * cx + (sx_off + ox - 0.5 - hw)/hw * cw ... */
        for (c = 0; c < 4; c++) {
            X[0][c] = (a->sx_scale / hw) * Cc[0][c] + ((a->sx_off + ox - 0.5 - hw) / hw) * Cc[3][c];
            X[1][c] = -((a->sy_scale / hh) * Cc[1][c] + ((a->sy_off + oy - 0.5 - hh) / hh) * Cc[3][c]);
            X[2][c] = (a->sz_scale / 32767.0) * Cc[2][c] + (a->sz_off / 32767.0) * Cc[3][c];
            X[3][c] = Cc[3][c];
        }
        k = 1.0 / g->S;       /* s16 position -> model units */
        for (i = 0; i < 4; i++) { X[i][0] *= k; X[i][1] *= k; X[i][2] *= k; }
        memcpy(Cc, X, sizeof(Cc));
    }
    /* near / far of the clip volume in w, for Tiny3D's w normalisation: ndc_z = -1 / +1 */
    {
        double m22 = a->v2s[2 * 4 + 2], m23 = a->v2s[2 * 4 + 3], m32 = a->v2s[3 * 4 + 2], m33 = a->v2s[3 * 4 + 3];
        double zn, zf, wn, wf, ss = a->sz_scale, so = a->sz_off;
        double tn = (-32767.0 - so) / ss, tf = (32767.0 - so) / ss;
        zn = (tn * m33 - m32) / (m22 - tn * m23);
        zf = (tf * m33 - m32) / (m22 - tf * m23);
        wn = zn * m23 + m33;
        wf = zf * m23 + m33;
        if (wn > wf) { double t = wn; wn = wf; wf = t; }
        if (!(wn > 1e-4)) wn = 1e-2;
        if (!(wf > wn * 1.5)) wf = wn * 100.0;
        near_w = (float)wn;
        far_w = (float)wf;
        {
            T3DMat4* pm = &vp.matProj;
            int r, c;
            t3d_viewport_set_perspective(&vp, T3D_DEG_TO_RAD(60.0f), (float)SCR_W / SCR_H, near_w, far_w);
            t3d_viewport_set_view_matrix(&vp, &(T3DMat4){ .m = { {1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1} } });
            for (r = 0; r < 4; r++)
                for (c = 0; c < 4; c++) P[r][c] = pm->m[c][r];   /* T3DMat4 is column-major m[col][row] */
        }
        if (!inv4(P, Pi)) return 0;
        mul4(Pi, Cc, M);
        for (i = 0; i < 4; i++) {
            int c;
            for (c = 0; c < 4; c++) {
                double v = M[i][c];
                if (!(v == v) || v > 30000.0 || v < -30000.0) { g->bad = 1; stat.bad_mat++; stat.fallback_bad++; return 0; }
                mf.m[c][i] = (float)v;
            }
        }
        if (gN64_fog_on && a->sz_scale != 0) {
            double zs = gN64_fog_z0;
            double ze = zs + (1.0001 - zs) / (gN64_fog_k > 0.01f ? gN64_fog_k : 0.01f);
            double ts = ((zs * 65534.0 - 32767.0) - so) / ss, te = ((ze * 65534.0 - 32767.0) - so) / ss;
            double zvs = (ts * m33 - m32) / (m22 - ts * m23), zve = (te * m33 - m32) / (m22 - te * m23);
            double zcs = (2.0 * zs - 1.0) * (zvs * m23 + m33), zce = (2.0 * ze - 1.0) * (zve * m23 + m33);
            double fa = -P[2][2], fb = P[2][3];   /* proj m22 / m32 (column-major: P[row][col]) */
            if (fabs(fa) > 1e-9 && zce > zcs) {
                fog_n = (float)((zcs - fb) / fa);
                fog_f = (float)((zce - fb) / fa);
                fog_en = fog_f > fog_n;
            }
        }
    }

    t3d_viewport_attach(&vp);
    /* fog needs the viewport attached */
    if (fog_en) {
        t3d_fog_set_range(fog_n, fog_f);
        t3d_fog_set_enabled(true);
    } else {
        t3d_fog_set_enabled(false);
    }
    /* t3d_viewport_attach() set the scissor to the whole viewport: put the scene rectangle back */
    {
        int x0 = ox < 0 ? 0 : ox, y0 = oy < 0 ? 0 : oy;
        int x1 = ox + w > SCR_W ? SCR_W : ox + w, y1 = oy + h > SCR_H ? SCR_H : oy + h;
        rdpq_set_scissor(x0, y0, x1, y1);
    }
    t3d_state_set_drawflags(d->textured ? (T3D_FLAG_TEXTURED | T3D_FLAG_SHADED | T3D_FLAG_DEPTH) : (T3D_FLAG_SHADED | T3D_FLAG_DEPTH));
    t3d_light_set_count(0);

    mfp = &mats[mat_next];
    mat_next = (mat_next + 1) % NMAT;
    t3d_mat4_to_fixed(mfp, &mf);
    t3d_matrix_push(mfp);
    for (b = 0; b < g->nbatches; b++) {
        t3d_batch* B = &g->b[b];
        t3d_vert_load(B->vb, 0, B->nslots);
        for (i = 0; i < B->ntris; i++) t3d_tri_draw(B->tri[i * 3], B->tri[i * 3 + 1], B->tri[i * 3 + 2]);
        t3d_tri_sync();
        ntris += B->ntris;
    }
    t3d_matrix_pop(1);
    stat.groups++;
    stat.tris += ntris;
    N64_Rdp_T3dForgetState();
    N64_Rdp_T3dCovered(ntris);
    {
        static int calls;
        if ((++calls % 600) == 1)
            fprintf(stderr, "t3d: groups %d tris %d builds %d fallback bad %d budget %d cache %d KB\n", stat.groups, stat.tris,
                stat.builds, stat.fallback_bad, stat.fallback_budget, cache_used_bytes / 1024);
    }
    return 1;
}
#endif

#ifdef N64
void T3dStats(int out[20]) {
    out[0] = stat.groups; out[1] = stat.tris; out[2] = stat.builds; out[3] = stat.fallback_bad; out[4] = stat.fallback_budget;
    out[5] = cache_used_bytes / 1024;
    out[6] = stat.bad_uv; out[7] = stat.bad_mat; out[8] = stat.bad_build;
    { int i; for (i = 0; i < 8; i++) { out[9 + i] = stat.rej[i]; stat.rej[i] = 0; } }
    stat.groups = stat.tris = 0;
}
#endif
