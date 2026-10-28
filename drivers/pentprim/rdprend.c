/*
 * N64 RDP triangle renderer.
 *
 * Replaces the software rasterisers for Z-buffered triangle primitives: the vertices that BRender
 * hands to a primitive block are converted to rdpq vertices and drawn by the RDP (textured with the
 * game's 8-bit textures through a TLUT, perspective correct, bilinear filtered, z-buffered).
 * The palette/shade-table lighting of the original renderer is approximated with a per-vertex grey
 * level derived from the shade table.
 */
#ifdef N64

#include "drv.h"
#include "shortcut.h"
#include "brassert.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* harness side (src/harness/platforms/n64_rdp.c) */
extern int gN64_use_rdp;
extern int N64_Rdp_BeginScene(void* pixels, int row_bytes, int w, int h);
extern void N64_Rdp_EndScene(uint8_t* pixels, int row_bytes, int w, int h);
extern int N64_Rdp_SetTexture(const uint8_t* px, int w, int h, int row_bytes, int* ew, int* eh);
extern void N64_Rdp_SetMode(int kind);
extern void N64_Rdp_TriFlat(const float* a, const float* b, const float* c);
extern void N64_Rdp_TriTex(const float* a, const float* b, const float* c);
extern uint16_t N64_Rdp_PaletteEntry(int i);
extern int N64_Rdp_OffsetX(void);
extern int N64_Rdp_OffsetY(void);
extern unsigned N64_Rdp_PaletteSerial(void);

/* vertex: x y z r g b a  s*iw t*iw iw   (attributes linear in screen space) */
#define VF 10
typedef struct {
    float f[VF];
} rv;

typedef struct {
    brp_block blk;
    const struct local_block* orig;
    int textured, persp, blend, decal;
    int has_i_vertex, has_i_const;
    int shade_h;
    const uint8_t* shade_table;
    int index_base;
    float tex_w, tex_h;
    int unscaled;
    const uint8_t* tex_base;
    int tex_wi, tex_hi, tex_stride;
    int mode;
} rdp_blk;

#define POOL 96
static rdp_blk pool[POOL];
static int pool_next;

/* per-scene state + the block currently being rendered (copied from its pool entry) */
static struct {
    int scene_open;
    int scene_rejected;
    uint8_t* pixels;
    int row_bytes, w, h;
    int textured, persp, blend, decal;
    int has_i_vertex, has_i_const;
    int shade_h;
    const uint8_t* shade_table;
    int index_base;
    float tex_w, tex_h;
    int unscaled;
    float uk, vk;
    int tex_ok;
    int mode;
} R;

/* ---- statistics (logged at the end of each scene) ---- */
static struct {
    int tex, flat, drop_tex, drop_bad, sub_notri, sub_noz, sub_notex, sub_nocpm, split_tex_types[4];
    int dumped;
} ST;

/* ---- luminance ramp for the current shade table ---- */
static float ramp[256];
static const uint8_t* ramp_table;
static int ramp_h;
static unsigned ramp_serial;

static int lum5(uint16_t c) {
    return ((c >> 11) & 31) * 77 + ((c >> 6) & 31) * 150 + ((c >> 1) & 31) * 29;
}

static void build_ramp(const uint8_t* table, int h) {
    int r, c;
    long base = 0;
    if (h > 256) {
        h = 256;
    }
    for (c = 1; c < 256; c++) {
        base += lum5(N64_Rdp_PaletteEntry(c));
    }
    if (base < 1) {
        base = 1;
    }
    for (r = 0; r < h; r++) {
        long sum = 0;
        for (c = 1; c < 256; c++) {
            sum += lum5(N64_Rdp_PaletteEntry(table[r * 256 + c]));
        }
        ramp[r] = (float)sum / (float)base;
        if (ramp[r] > 1.0f) {
            ramp[r] = 1.0f;
        }
    }
    ramp_table = table;
    ramp_h = h;
    ramp_serial = N64_Rdp_PaletteSerial();
}

static float shade_of(float i) {
    int r;
    float f;
    if (R.shade_table == NULL || R.shade_h < 2) {
        return 1.0f;
    }
    if (ramp_table != R.shade_table || ramp_h != R.shade_h || ramp_serial != N64_Rdp_PaletteSerial()) {
        build_ramp(R.shade_table, R.shade_h);
    }
    i -= 0.5f;
    if (i < 0) {
        i = 0;
    }
    if (i > (float)(ramp_h - 1)) {
        i = (float)(ramp_h - 1);
    }
    r = (int)i;
    f = i - (float)r;
    if (r >= ramp_h - 1) {
        return ramp[ramp_h - 1];
    }
    return ramp[r] + (ramp[r + 1] - ramp[r]) * f;
}

/* ---- primitive emission ---- */
#define GUARD_X0 (-200.0f)
#define GUARD_X1 (520.0f)
#define GUARD_Y0 (-200.0f)
#define GUARD_Y1 (400.0f)

static int finite_f(float v) {
    return v == v && v < 1e9f && v > -1e9f;
}

static void emit_one(const rv* a, const rv* b, const rv* c) {
    float va[10], vb[10], vc[10];
    float* out[3] = { va, vb, vc };
    const rv* in[3] = { a, b, c };
    int k, j;
    float umin, vmin, bu, bv;

    for (k = 0; k < 3; k++) {
        for (j = 0; j < 7; j++) {
            out[k][j] = in[k]->f[j];
        }
        if (R.textured) {
            float iw = in[k]->f[9];
            out[k][7] = in[k]->f[7] / iw;
            out[k][8] = in[k]->f[8] / iw;
            out[k][9] = iw;
        }
    }
    if (R.textured) {
        /* bring the texture coordinates close to 0 (the RDP only holds +-1024 texels) */
        umin = va[7] < vb[7] ? va[7] : vb[7];
        if (vc[7] < umin) {
            umin = vc[7];
        }
        vmin = va[8] < vb[8] ? va[8] : vb[8];
        if (vc[8] < vmin) {
            vmin = vc[8];
        }
        bu = floorf(umin / R.tex_w) * R.tex_w;
        bv = floorf(vmin / R.tex_h) * R.tex_h;
        for (k = 0; k < 3; k++) {
            out[k][7] -= bu;
            out[k][8] -= bv;
        }
        N64_Rdp_TriTex(va, vb, vc);
    } else {
        N64_Rdp_TriFlat(va, vb, vc);
    }
}

static void mid(rv* o, const rv* a, const rv* b) {
    int j;
    for (j = 0; j < VF; j++) {
        o->f[j] = (a->f[j] + b->f[j]) * 0.5f;
    }
}

/* split triangles whose texture coordinates span too many texels */
static void emit_split(const rv* a, const rv* b, const rv* c, int depth) {
    if (R.textured && depth < 5) {
        float u[3], v[3], umin, umax, vmin, vmax;
        const rv* t[3] = { a, b, c };
        int k;
        for (k = 0; k < 3; k++) {
            u[k] = t[k]->f[7] / t[k]->f[9];
            v[k] = t[k]->f[8] / t[k]->f[9];
        }
        umin = umax = u[0];
        vmin = vmax = v[0];
        for (k = 1; k < 3; k++) {
            if (u[k] < umin) umin = u[k];
            if (u[k] > umax) umax = u[k];
            if (v[k] < vmin) vmin = v[k];
            if (v[k] > vmax) vmax = v[k];
        }
        if (umax - umin > 700.0f || vmax - vmin > 700.0f) {
            rv ab, bc, ca;
            mid(&ab, a, b);
            mid(&bc, b, c);
            mid(&ca, c, a);
            emit_split(a, &ab, &ca, depth + 1);
            emit_split(&ab, b, &bc, depth + 1);
            emit_split(&ca, &bc, c, depth + 1);
            emit_split(&ab, &bc, &ca, depth + 1);
            return;
        }
    }
    emit_one(a, b, c);
}

/* Sutherland-Hodgman against one guard-band plane */
static int clip_plane(const rv* in, int n, rv* out, int axis, float limit, int keep_greater) {
    int i, m = 0;
    for (i = 0; i < n; i++) {
        const rv* p = &in[i];
        const rv* q = &in[(i + 1) % n];
        float dp = keep_greater ? p->f[axis] - limit : limit - p->f[axis];
        float dq = keep_greater ? q->f[axis] - limit : limit - q->f[axis];
        if (dp >= 0) {
            out[m++] = *p;
        }
        if ((dp >= 0) != (dq >= 0)) {
            float t = dp / (dp - dq);
            int j;
            for (j = 0; j < VF; j++) {
                out[m].f[j] = p->f[j] + (q->f[j] - p->f[j]) * t;
            }
            m++;
        }
        if (m > 12) {
            break;
        }
    }
    return m;
}

static void draw_tri(const rv* a, const rv* b, const rv* c) {
    int outside = 0, k;
    const rv* t[3] = { a, b, c };
    for (k = 0; k < 3; k++) {
        float x = t[k]->f[0], y = t[k]->f[1];
        if (x < GUARD_X0 || x > GUARD_X1 || y < GUARD_Y0 || y > GUARD_Y1) {
            outside = 1;
        }
    }
    if (!outside) {
        emit_split(a, b, c, 0);
        return;
    }
    {
        rv p0[16], p1[16];
        int n = 3, i;
        p0[0] = *a;
        p0[1] = *b;
        p0[2] = *c;
        n = clip_plane(p0, n, p1, 0, GUARD_X0, 1);
        if (n < 3) return;
        n = clip_plane(p1, n, p0, 0, GUARD_X1, 0);
        if (n < 3) return;
        n = clip_plane(p0, n, p1, 1, GUARD_Y0, 1);
        if (n < 3) return;
        n = clip_plane(p1, n, p0, 1, GUARD_Y1, 0);
        if (n < 3) return;
        for (i = 1; i + 1 < n; i++) {
            emit_split(&p0[0], &p0[i], &p0[i + 1], 0);
        }
    }
}

static void fill_vertex(rv* o, const brp_vertex* v, const brp_vertex* v0) {
    float sz = v->comp[C_SZ];
    float z = (sz + 32767.0f) * (1.0f / 65534.0f);
    float lum = 1.0f, r, g, b, a = 1.0f;
    int idx;

    if (z < 0) z = 0;
    if (z > 1) z = 1;
    if (R.decal) {
        z -= 0.002f;
        if (z < 0) z = 0;
    }
    o->f[0] = v->comp[C_SX] - 0.5f + (float)N64_Rdp_OffsetX();
    o->f[1] = v->comp[C_SY] - 0.5f + (float)N64_Rdp_OffsetY();
    o->f[2] = z;

    if (R.textured) {
        if (R.has_i_vertex) {
            lum = shade_of(v->comp[C_I]);
        } else if (R.has_i_const) {
            lum = shade_of(v0->comp[C_I]);
        }
        r = g = b = lum;
    } else {
        float iv = R.has_i_vertex ? v->comp[C_I] : (R.has_i_const ? v0->comp[C_I] : (float)R.index_base + 0.5f);
        if (R.shade_table != NULL && R.shade_h > 1 && (R.has_i_vertex || R.has_i_const)) {
            int row = (int)iv;
            if (row < 0) row = 0;
            if (row >= R.shade_h) row = R.shade_h - 1;
            idx = R.shade_table[row * 256 + (R.index_base & 255)];
        } else {
            idx = (int)iv;
            if (idx < 0) idx = 0;
            if (idx > 255) idx = 255;
        }
        {
            uint16_t c = N64_Rdp_PaletteEntry(idx);
            r = (float)((c >> 11) & 31) * (1.0f / 31.0f);
            g = (float)((c >> 6) & 31) * (1.0f / 31.0f);
            b = (float)((c >> 1) & 31) * (1.0f / 31.0f);
        }
    }
    if (R.blend) {
        a = 0.5f;
    }
    o->f[3] = r;
    o->f[4] = g;
    o->f[5] = b;
    o->f[6] = a;
    if (R.textured) {
        float iw = 1.0f;
        if (R.persp) {
            float w = v->comp[C_W];
            iw = w > 1e-6f ? 1.0f / w : 1.0f;
        }
        float us = R.unscaled ? R.tex_w : R.uk, vs = R.unscaled ? R.tex_h : R.vk;
        o->f[7] = v->comp[C_U] * us * iw;
        o->f[8] = v->comp[C_V] * vs * iw;
        o->f[9] = iw;
    } else {
        o->f[7] = o->f[8] = 0;
        o->f[9] = 1;
    }
}

void BR_ASM_CALL RdpTriangleRender(brp_block* block, brp_vertex* v0, brp_vertex* v1, brp_vertex* v2,
    br_uint_16* fp_vertices, br_uint_16* fp_edges, br_vector4* fp_eqn, struct temp_face* tfp) {
    rv a, b, c;
    int k;
    const brp_vertex* t[3];
    const rdp_blk* pb = (const rdp_blk*)block;

    if (v2 == NULL || !R.scene_open) {
        return;
    }
    R.textured = pb->textured;
    R.persp = pb->persp;
    R.blend = pb->blend;
    R.decal = pb->decal;
    R.has_i_vertex = pb->has_i_vertex;
    R.has_i_const = pb->has_i_const;
    R.shade_h = pb->shade_h;
    R.shade_table = pb->shade_table;
    R.index_base = pb->index_base;
    R.tex_w = pb->tex_w;
    R.tex_h = pb->tex_h;
    R.unscaled = pb->unscaled;
    R.tex_ok = 1;
    if (pb->textured) {
        int ew = pb->tex_wi, eh = pb->tex_hi;
        R.tex_ok = N64_Rdp_SetTexture(pb->tex_base, pb->tex_wi, pb->tex_hi, pb->tex_stride, &ew, &eh);
        R.tex_w = (float)ew;
        R.tex_h = (float)eh;
        R.uk = ew / pb->tex_w;
        R.vk = eh / pb->tex_h;
    }
    N64_Rdp_SetMode(pb->mode);
    t[0] = v0;
    t[1] = v1;
    t[2] = v2;
    for (k = 0; k < 3; k++) {
        if (!finite_f(t[k]->comp[C_SX]) || !finite_f(t[k]->comp[C_SY]) || !finite_f(t[k]->comp[C_SZ])) {
            return;
        }
        if (R.textured && (!finite_f(t[k]->comp[C_U]) || !finite_f(t[k]->comp[C_V]))) {
            return;
        }
    }
    if (R.textured && !R.tex_ok) {
        ST.drop_tex++;
        return;
    }
    if (R.textured) {
        ST.tex++;
    } else {
        ST.flat++;
    }
    fill_vertex(&a, v0, v0);
    fill_vertex(&b, v1, v0);
    fill_vertex(&c, v2, v0);
    if (R.textured) {
        /* only perspective blocks have a valid C_W; others keep iw = 1 */
        if (!(a.f[9] > 0 && b.f[9] > 0 && c.f[9] > 0)) {
            return;
        }
    }
    if (R.textured && ST.dumped < 12) {
        ST.dumped++;
        fprintf(stderr, "rdp tri tex %dx%d persp %d iw %.4f %.4f %.4f uv0 %.2f,%.2f uv1 %.2f,%.2f uv2 %.2f,%.2f rawUV0 %.3f,%.3f W0 %.3f lum %.2f xy %.1f,%.1f %.1f,%.1f %.1f,%.1f z %.3f\n",
            (int)R.tex_w, (int)R.tex_h, R.persp, a.f[9], b.f[9], c.f[9],
            a.f[7] / a.f[9], a.f[8] / a.f[9], b.f[7] / b.f[9], b.f[8] / b.f[9], c.f[7] / c.f[9], c.f[8] / c.f[9],
            v0->comp[C_U], v0->comp[C_V], v0->comp[C_W], a.f[3], a.f[0], a.f[1], b.f[0], b.f[1], c.f[0], c.f[1], a.f[2]);
    }
    draw_tri(&a, &b, &c);
}

/*
 * Called by renderBegin after a block has been selected. May replace *rpb by an RDP block.
 */
void RdpSubstitute(struct br_primitive_state* self, struct local_block* lb, struct brp_block** rpb) {
    const char* id;
    br_pixelmap* cpm;
    int textured;

    if (!gN64_use_rdp || lb == NULL || lb->p.type != BRT_TRIANGLE) {
        ST.sub_notri++;
        return;
    }
    id = lb->p.identifier;
    if (id == NULL || strstr(id, "Z Buffered") == NULL) {
        ST.sub_noz++;
        return;
    }
    cpm = self->out.colour.pixelmap;
    if (cpm == NULL || work.colour.base == NULL) {
        ST.sub_nocpm++;
        return;
    }
    if (!R.scene_open) {
        if (R.scene_rejected) {
            return;
        }
        if (!N64_Rdp_BeginScene(work.colour.base, work.colour.stride_b, work.colour.width_p, work.colour.height)) {
            R.scene_rejected = 1;
            return;
        }
        R.scene_open = 1;
        R.pixels = (uint8_t*)work.colour.base;
        R.row_bytes = work.colour.stride_b;
        R.w = work.colour.width_p;
        R.h = work.colour.height;
    }

    {
        rdp_blk* e = NULL;
        int n;
        const uint8_t* tbase = (const uint8_t*)work.texture.base;
        const uint8_t* stab = self->prim.index_shade.buffer ? (const uint8_t*)work.shade_table : NULL;
        int ibase = self->prim.index_base;
        textured = strstr(id, "Textured") != NULL;
        if (textured && (tbase == NULL || work.texture.type != BR_PMT_INDEX_8)) {
            ST.sub_notex++;
            return;
        }
        for (n = 0; n < POOL; n++) {
            if (pool[n].orig == lb && pool[n].tex_base == (textured ? tbase : NULL) && pool[n].shade_table == stab
                && pool[n].index_base == ibase && pool[n].tex_wi == (textured ? (int)work.texture.width_p : 0)) {
                e = &pool[n];
                break;
            }
        }
        if (e == NULL) {
            e = &pool[pool_next];
            pool_next = (pool_next + 1) % POOL;
            memset(e, 0, sizeof(*e));
            e->orig = lb;
            e->textured = textured;
            e->persp = 1; /* C_W (view depth) is always valid: always perspective-correct */
            e->blend = strstr(id, "Blend") != NULL;
            e->decal = strstr(id, "Decal") != NULL;
            e->has_i_vertex = (lb->p.vertex_components & CM_I) != 0;
            e->has_i_const = (lb->p.constant_components & CM_I) != 0;
            e->shade_h = stab ? self->prim.index_shade.height : 0;
            e->shade_table = stab;
            e->index_base = ibase;
            if (textured) {
                e->tex_base = tbase;
                e->tex_wi = work.texture.width_p;
                e->tex_hi = work.texture.height;
                e->tex_stride = work.texture.stride_b;
                e->tex_w = (float)e->tex_wi;
                e->tex_h = (float)e->tex_hi;
                e->unscaled = (lb->range_flags & RF_UNSCALED_TEXTURE_COORDS) != 0;
                e->mode = e->blend ? 3 : 1;
            } else {
                e->mode = e->blend ? 2 : 0;
            }
            if (textured) {
                static int sublogs = 0;
                if (sublogs < 8) {
                    sublogs++;
                    fprintf(stderr, "rdp sub: scales U %g V %g | colour_map %dx%d | work tex %dx%d | tex_type %d range_flags %x\n",
                        (double)self->cache.comp_scales[C_U], (double)self->cache.comp_scales[C_V],
                        (int)self->prim.colour_map.width,
                        (int)self->prim.colour_map.height,
                        e->tex_wi, e->tex_hi, (int)lb->texture_type, (unsigned)lb->range_flags);
                }
            }
            {
                static int plogs = 0;
                if (plogs < 60) {
                    plogs++;
                    fprintf(stderr, "rdp blk #%d '%s' tex %d blend %d decal %d vcomp %x ccomp %x tex %dx%d\n", plogs, id, textured, e->blend, e->decal,
                        (unsigned)lb->p.vertex_components, (unsigned)lb->p.constant_components, e->tex_wi, e->tex_hi);
                }
            }
            e->blk = lb->p;
            e->blk.render = (brp_render_fn*)RdpTriangleRender;
            e->blk.chain = NULL;
            e->blk.convert_mask_f = e->blk.convert_mask_f | e->blk.convert_mask_x | e->blk.convert_mask_i;
            e->blk.convert_mask_x = 0;
            e->blk.convert_mask_i = 0;
        }
        *rpb = &e->blk;
    }
}

/* End of scene (primitive library flush) */
void RdpFlush(void) {
    static int logs = 0;
    if (R.scene_open && logs < 40) {
        logs++;
        fprintf(stderr, "rdp scene: tex tris %d, flat tris %d, dropped(tex) %d | sw blocks: non-tri %d, non-zbuf %d, no-8bit-tex %d | pool %d\n",
            ST.tex, ST.flat, ST.drop_tex, ST.sub_notri, ST.sub_noz, ST.sub_notex, pool_next);
    }
    ST.tex = ST.flat = ST.drop_tex = ST.sub_notri = ST.sub_noz = ST.sub_notex = ST.sub_nocpm = 0;
    if (R.scene_open) {
        N64_Rdp_EndScene(R.pixels, R.row_bytes, R.w, R.h);
    }
    R.scene_open = 0;
    R.scene_rejected = 0;
}

#endif /* N64 */
