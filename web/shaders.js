// WGSL shader sources.
//
// Simulation: kick-drift-kick leapfrog, same scheme as the CPU core. The
// force pass is the classic tiled all-pairs kernel: each 256-thread
// workgroup marches over the body list in 256-body tiles staged through
// workgroup memory, so every global position is read once per workgroup
// instead of once per thread. Out-of-range slots carry mass 0 and Plummer
// softening makes the self-interaction term exactly zero, so the inner loop
// has no branches at all.
//
// vel.w is repurposed as a population tag (0/1 = which galaxy, 2 = mass the
// visitor flung in); the integrator only ever adds to vel.xyz, so the tag
// survives. The renderer uses it to tint the populations so you can watch
// the galaxies interleave during the merger.

export const SIM_WGSL = /* wgsl */ `
struct SimU {
  dt:    f32,
  eps2:  f32,
  count: u32,
  _pad:  u32,
}

@group(0) @binding(0) var<storage, read_write> posMass: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> vel:     array<vec4f>;
@group(0) @binding(2) var<storage, read_write> accel:   array<vec4f>;
@group(0) @binding(3) var<uniform> U: SimU;

const TILE = 256u;
var<workgroup> tile: array<vec4f, TILE>;

@compute @workgroup_size(256)
fn kick_drift(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.count) { return; }
  var v = vel[i];
  v += vec4f(accel[i].xyz, 0.0) * (0.5 * U.dt);
  vel[i] = v;
  posMass[i] += vec4f(v.xyz, 0.0) * U.dt;
}

@compute @workgroup_size(256)
fn forces(@builtin(global_invocation_id) gid: vec3u,
          @builtin(local_invocation_id) lid: vec3u) {
  let i = gid.x;
  var p = vec3f(0.0);
  if (i < U.count) { p = posMass[i].xyz; }

  var acc = vec3f(0.0);
  let ntiles = (U.count + TILE - 1u) / TILE;
  for (var t = 0u; t < ntiles; t++) {
    let j = t * TILE + lid.x;
    tile[lid.x] = select(vec4f(0.0), posMass[j], j < U.count);
    workgroupBarrier();
    for (var k = 0u; k < TILE; k++) {
      let q = tile[k];
      let d = q.xyz - p;
      let r2 = dot(d, d) + U.eps2;
      let inv = inverseSqrt(r2);
      acc += (q.w * inv * inv * inv) * d;
    }
    workgroupBarrier();
  }
  if (i < U.count) { accel[i] = vec4f(acc, 0.0); }
}

@compute @workgroup_size(256)
fn kick(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.count) { return; }
  vel[i] += vec4f(accel[i].xyz, 0.0) * (0.5 * U.dt);
}
`;

// Rendering: one camera-facing quad per body, additive-blended gaussian
// sprites into an rgba16float target. Color comes from three layered cues:
// speed (slow halo stars cool, fast core stars hot), population tint (each
// galaxy and any flung mass get their own cast), and a per-star hash that
// varies size, brightness and hue so the field reads as stars rather than
// uniform dots — including a sprinkling of rare bright ones.

export const RENDER_WGSL = /* wgsl */ `
struct RenderU {
  viewProj: mat4x4f,
  right:    vec4f,   // xyz camera right, w = base sprite half-size
  up:       vec4f,   // xyz camera up,    w = mean body mass
  params:   vec4f,   // x = vmax for color ramp
}

@group(0) @binding(0) var<storage, read> posMass: array<vec4f>;
@group(0) @binding(1) var<storage, read> vel:     array<vec4f>;
@group(0) @binding(2) var<uniform> R: RenderU;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv:    vec2f,
  @location(1) color: vec3f,
}

fn hash1(n: u32) -> f32 {
  var x = n * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  x = (x >> 22u) ^ x;
  return f32(x) / 4294967295.0;
}

@vertex
fn vs(@builtin(vertex_index) vi: u32,
      @builtin(instance_index) inst: u32) -> VSOut {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let c = corners[vi];

  let pm = posMass[inst];
  let v = vel[inst];
  let speed = length(v.xyz);

  let h1 = hash1(inst);
  let h2 = hash1(inst ^ 0x9e3779b9u);

  // Heavier bodies get modestly larger sprites; the central masses are
  // thousands of particle masses, so clamp hard or they fill the screen.
  let szScale = clamp(sqrt(pm.w / R.up.w), 0.8, 3.0) * mix(0.7, 1.5, h1);
  let hs = R.right.w * szScale;

  let world = pm.xyz + (R.right.xyz * c.x + R.up.xyz * c.y) * hs;

  // Speed ramp, hue-jittered per star.
  let t = clamp(speed / R.params.x + (h1 - 0.5) * 0.16, 0.0, 1.0);
  let cool = vec3f(0.45, 0.58, 1.00);
  let mid  = vec3f(1.00, 0.96, 0.92);
  let hot  = vec3f(1.00, 0.62, 0.28);
  var col: vec3f;
  if (t < 0.5) { col = mix(cool, mid, t * 2.0); }
  else         { col = mix(mid, hot, (t - 0.5) * 2.0); }

  // Population tint: galaxy 0 warm, galaxy 1 cool, flung mass violet.
  var tints = array<vec3f, 3>(
    vec3f(1.05, 0.95, 0.82),
    vec3f(0.80, 0.92, 1.12),
    vec3f(1.05, 0.78, 1.15));
  let tint = tints[min(u32(v.w + 0.5), 2u)];
  col = col * mix(vec3f(1.0), tint, 0.5);

  // Brightness lottery: most stars dim, a few blaze.
  var lum = mix(0.6, 1.3, h2 * h2);
  if (h2 > 0.985) { lum *= 3.5; }
  // Central masses read as galactic cores.
  if (pm.w > R.up.w * 100.0) { lum *= 4.0; }

  var out: VSOut;
  out.pos = R.viewProj * vec4f(world, 1.0);
  out.uv = c;
  out.color = col * lum;
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let r2 = dot(in.uv, in.uv);
  // Bright core + wide faint skirt; the skirt is what reads as glow once
  // thousands of sprites add up.
  let w = 1.6 * exp(-14.0 * r2) + 0.22 * exp(-3.5 * r2);
  return vec4f(in.color * w, w);
}
`;

// Post-processing. Bloom runs at half resolution: bright-pass, then two
// horizontal+vertical gaussian iterations (the second with wider taps) —
// cheap next to the n-body compute, and it's what turns dense cores into
// light sources instead of white blobs.

export const POST_WGSL = /* wgsl */ `
struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn fullscreen(@builtin(vertex_index) vi: u32) -> VSOut {
  var out: VSOut;
  let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  out.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(xy.x, 1.0 - xy.y);
  return out;
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSamp: sampler;

@fragment
fn brightpass(in: VSOut) -> @location(0) vec4f {
  let c = textureSample(srcTex, srcSamp, in.uv).rgb;
  // Soft knee around the threshold so bloom fades in rather than pops.
  let l = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  let k = smoothstep(0.55, 1.4, l);
  return vec4f(c * k, 1.0);
}

struct BlurU { dir: vec2f, _pad: vec2f }
@group(0) @binding(2) var<uniform> B: BlurU;

@fragment
fn blur(in: VSOut) -> @location(0) vec4f {
  var w = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
  var c = textureSample(srcTex, srcSamp, in.uv).rgb * w[0];
  for (var i = 1; i < 5; i++) {
    let o = B.dir * f32(i);
    c += textureSample(srcTex, srcSamp, in.uv + o).rgb * w[i];
    c += textureSample(srcTex, srcSamp, in.uv - o).rgb * w[i];
  }
  return vec4f(c, 1.0);
}
`;

// Volumetric nebula, stage 1: deposit particle mass into a 96^3 density
// grid. Float atomics don't exist in WGSL, so the grid is fixed-point u32
// accumulated with atomicAdd, then a second pass converts it into a
// filterable 3D texture for the ray marcher. Central masses are capped so a
// single body can't nuke one cell.

export const VOLUME_WGSL = /* wgsl */ `
struct VolU {
  boxMin: vec3f,
  count: u32,
  invBoxSize: vec3f,   // 1 / boxSize, per axis
  dim: u32,
  depositCap: f32,     // max mass deposited by one body
  fixedScale: f32,     // mass -> fixed-point
  densNorm: f32,       // cell mass -> O(1) density for the marcher
  _pad: f32,
}

@group(0) @binding(0) var<storage, read> posMass: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> grid: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> U: VolU;
@group(0) @binding(3) var volOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(256)
fn splat(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.count) { return; }
  let p = posMass[i];
  let g = (p.xyz - U.boxMin) * U.invBoxSize;
  if (any(g < vec3f(0.0)) || any(g >= vec3f(1.0))) { return; }
  let c = min(vec3u(g * f32(U.dim)), vec3u(U.dim - 1u));
  let idx = (c.z * U.dim + c.y) * U.dim + c.x;
  atomicAdd(&grid[idx], u32(min(p.w, U.depositCap) * U.fixedScale));
}

@compute @workgroup_size(4, 4, 4)
fn to_tex(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= vec3u(U.dim))) { return; }
  let idx = (gid.z * U.dim + gid.y) * U.dim + gid.x;
  let mass = f32(atomicLoad(&grid[idx])) / U.fixedScale;
  textureStore(volOut, gid, vec4f(mass * U.densNorm, 0.0, 0.0, 0.0));
}
`;

// Stage 2: march camera rays through the density volume (emission +
// absorption, front-to-back). Runs at half resolution; the result is
// composited under the star layer. Start jittered per pixel to trade
// banding for noise.

export const VOLMARCH_WGSL = /* wgsl */ `
struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var out: VSOut;
  let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  out.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(xy.x, 1.0 - xy.y);
  return out;
}

struct MarchU {
  eye:   vec4f,   // xyz, w = tan(fov/2)
  right: vec4f,   // xyz, w = aspect
  up:    vec4f,   // xyz, w = nebula density multiplier
  fwd:   vec4f,   // xyz
  box:   vec4f,   // xyz = boxMin, w = box side
}

@group(0) @binding(0) var vol: texture_3d<f32>;
@group(0) @binding(1) var volSamp: sampler;
@group(0) @binding(2) var<uniform> M: MarchU;

fn hash12(p: vec2f) -> f32 {
  let q = fract(p * vec2f(0.1031, 0.0973));
  let r = q + dot(q, q.yx + 33.33);
  return fract((r.x + r.y) * r.x);
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let dens = M.up.w;
  if (dens <= 0.0) { return vec4f(0.0); }

  let tanF = M.eye.w;
  let ndc = vec2f(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
  let dir = normalize(M.fwd.xyz +
                      M.right.xyz * ndc.x * tanF * M.right.w +
                      M.up.xyz * ndc.y * tanF);

  // Slab intersection with the volume cube.
  let bmin = M.box.xyz;
  let bmax = bmin + vec3f(M.box.w);
  let inv = 1.0 / dir;
  let t0v = (bmin - M.eye.xyz) * inv;
  let t1v = (bmax - M.eye.xyz) * inv;
  let tn = max(max(min(t0v.x, t1v.x), min(t0v.y, t1v.y)), min(t0v.z, t1v.z));
  let tf = min(min(max(t0v.x, t1v.x), max(t0v.y, t1v.y)), max(t0v.z, t1v.z));
  if (tf <= max(tn, 0.0)) { return vec4f(0.0); }

  const STEPS = 64;
  let t0 = max(tn, 0.0);
  let dt = (tf - t0) / f32(STEPS);
  var t = t0 + dt * hash12(in.pos.xy);

  var acc = vec3f(0.0);
  var trans = 1.0;
  for (var s = 0; s < STEPS; s++) {
    let p = M.eye.xyz + dir * t;
    let q = (p - bmin) / M.box.w;
    let d = textureSampleLevel(vol, volSamp, q, 0.0).r * dens;
    if (d > 1e-4) {
      // Emissive palette: thin gas deep indigo, denser teal, cores warm.
      let s1 = 1.0 - exp(-d * 1.5);
      var col = mix(vec3f(0.10, 0.13, 0.38), vec3f(0.22, 0.50, 0.62), s1);
      col = mix(col, vec3f(0.95, 0.80, 0.55), s1 * s1);
      let a = 1.0 - exp(-d * 2.2 * dt);
      acc += trans * col * a * 1.8;
      trans *= 1.0 - a;
      if (trans < 0.02) { break; }
    }
    t += dt;
  }
  return vec4f(acc, 1.0 - trans);
}
`;

export const COMPOSITE_WGSL = /* wgsl */ `
struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var out: VSOut;
  let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  out.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(xy.x, 1.0 - xy.y);
  return out;
}

struct CompU {
  lens0:  vec4f,   // xy = core screen uv, z = thetaE^2, w = enabled
  lens1:  vec4f,
  params: vec4f,   // x = aspect
}

@group(0) @binding(0) var hdrTex: texture_2d<f32>;
@group(0) @binding(1) var hdrSamp: sampler;
@group(0) @binding(2) var bloomTex: texture_2d<f32>;
@group(0) @binding(3) var nebulaTex: texture_2d<f32>;
@group(0) @binding(4) var<uniform> C: CompU;

fn hash2(p: vec2f) -> f32 {
  let q = fract(p * vec2f(0.1031, 0.0973));
  let r = q + dot(q, q.yx + 33.33);
  return fract((r.x + r.y) * r.x);
}

// Point-mass gravitational lens, thin-lens equation beta = theta -
// thetaE^2 / theta: to draw image position theta we sample the unlensed
// scene at beta. Inverse mapping means rings and arcs come out for free.
fn lensWarp(uv: vec2f) -> vec2f {
  var w = uv;
  for (var i = 0; i < 2; i++) {
    let L = select(C.lens1, C.lens0, i == 0);
    if (L.w > 0.5) {
      var d = w - L.xy;
      d.x *= C.params.x;
      let r2 = dot(d, d) + 1e-5;
      var shift = d * (L.z / r2);
      shift.x /= C.params.x;
      w -= shift;
    }
  }
  return w;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let wuv = lensWarp(in.uv);
  let hdr = textureSample(hdrTex, hdrSamp, wuv).rgb;
  let bloom = textureSample(bloomTex, hdrSamp, wuv).rgb;
  let nebula = textureSample(nebulaTex, hdrSamp, wuv).rgb;
  var c = vec3f(1.0) - exp(-(hdr + bloom * 0.9 + nebula) * 1.25);

  // Sparse static background starfield, post-tonemap so exposure changes
  // don't swallow it — sampled through the lens warp, so the cores bend the
  // background into arcs.
  let h = hash2(floor(wuv * 900.0));
  if (h > 0.9988) {
    let b = (h - 0.9988) / 0.0012;
    c += vec3f(0.55, 0.6, 0.75) * b * b * 0.30;
  }

  // A whisper of blue in the blacks so space reads deep rather than dead,
  // plus a soft vignette.
  c += vec3f(0.010, 0.012, 0.020);
  let v = 1.0 - 0.35 * smoothstep(0.45, 1.25, length(in.uv - 0.5));
  c *= v;
  return vec4f(pow(c, vec3f(1.0 / 2.2)), 1.0);
}
`;

// Multiplies the HDR accumulation buffer by a constant each frame (set via
// blend constant): 0 clears it, values near 1 leave light trails.
export const FADE_WGSL = /* wgsl */ `
struct VSOut { @builtin(position) pos: vec4f }

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var out: VSOut;
  let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  out.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
  return out;
}

@fragment
fn fs() -> @location(0) vec4f {
  return vec4f(0.0);
}
`;
