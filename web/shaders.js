// WGSL shader sources.
//
// Simulation: kick-drift-kick leapfrog, same scheme as the CPU core. The
// force pass is the classic tiled all-pairs kernel: each 256-thread
// workgroup marches over the body list in 256-body tiles staged through
// workgroup memory, so every global position is read once per workgroup
// instead of once per thread. Out-of-range slots carry mass 0 and Plummer
// softening makes the self-interaction term exactly zero, so the inner loop
// has no branches at all.

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
// sprites into an rgba16float target. Stars are colored by speed (slow halo
// stars cool blue, fast core stars hot amber); brightness accumulates in HDR
// and a final pass tonemaps, so dense regions bloom out naturally.

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

@vertex
fn vs(@builtin(vertex_index) vi: u32,
      @builtin(instance_index) inst: u32) -> VSOut {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let c = corners[vi];

  let pm = posMass[inst];
  let speed = length(vel[inst].xyz);

  // Heavier bodies get modestly larger sprites; the central masses are
  // thousands of particle masses, so clamp hard or they fill the screen.
  let szScale = clamp(sqrt(pm.w / R.up.w), 0.8, 3.0);
  let hs = R.right.w * szScale;

  let world = pm.xyz + (R.right.xyz * c.x + R.up.xyz * c.y) * hs;

  let t = clamp(speed / R.params.x, 0.0, 1.0);
  let cool = vec3f(0.45, 0.58, 1.00);
  let mid  = vec3f(1.00, 0.96, 0.92);
  let hot  = vec3f(1.00, 0.62, 0.28);
  var col: vec3f;
  if (t < 0.5) { col = mix(cool, mid, t * 2.0); }
  else         { col = mix(mid, hot, (t - 0.5) * 2.0); }

  var out: VSOut;
  out.pos = R.viewProj * vec4f(world, 1.0);
  out.uv = c;
  out.color = col;
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

export const COMPOSITE_WGSL = /* wgsl */ `
@group(0) @binding(0) var hdrTex: texture_2d<f32>;
@group(0) @binding(1) var hdrSamp: sampler;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  // Fullscreen triangle.
  var out: VSOut;
  let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  out.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(xy.x, 1.0 - xy.y);
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let hdr = textureSample(hdrTex, hdrSamp, in.uv).rgb;
  // Exponential tonemap + a whisper of blue in the blacks so space reads as
  // deep rather than dead, then gamma for the non-srgb swapchain.
  var c = vec3f(1.0) - exp(-hdr * 1.25);
  c += vec3f(0.012, 0.014, 0.022);
  let v = 1.0 - 0.35 * smoothstep(0.45, 1.25, length(in.uv - 0.5));
  c *= v;
  return vec4f(pow(c, vec3f(1.0 / 2.2)), 1.0);
}
`;
