// Initial conditions, generated on the CPU in JS (G = 1 units, same physics
// conventions as the C++ core). Returns interleaved Float32Arrays matching
// the GPU buffer layout: posMass = [x y z m]*, vel = [vx vy vz 0]*.

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function basisFrom(axis) {
  const n = norm(axis);
  const h = Math.abs(n[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const e1 = norm(cross(h, n));
  const e2 = cross(n, e1);
  return [e1, e2, n];
}

const norm = (a) => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1],
                         a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0]];

// One rotating disk galaxy: a dominant central mass plus an exponential disk
// (surface density ~ exp(-r/Rd), sampled exactly as r = -Rd ln(u1*u2)) on
// near-circular orbits. v_circ comes from the spherically-enclosed mass —
// approximate for a flat disk, but the slight imbalance just feeds the spiral
// structure that makes the collision pretty.
function disk({ posMass, vel, offset, n, center, bulk, axis, radius, mass, rng }) {
  const Rd = radius / 3;
  const bulgeFrac = 0.3;
  const mc = mass * bulgeFrac;          // central point mass
  const md = mass * (1 - bulgeFrac);    // disk total
  const mp = md / (n - 1);              // per disk particle
  const [e1, e2, nrm] = basisFrom(axis);

  // Central mass first.
  posMass.set([center[0], center[1], center[2], mc], offset * 4);
  vel.set([bulk[0], bulk[1], bulk[2], 0], offset * 4);

  for (let i = 1; i < n; i++) {
    let r;
    do { r = -Rd * Math.log(Math.max(rng() * rng(), 1e-12)); } while (r > radius * 1.6);
    r = Math.max(r, 0.04 * radius);
    const ph = rng() * Math.PI * 2;
    const z = (rng() + rng() + rng() - 1.5) * 0.05 * radius; // ~gaussian thin disk

    const cx = Math.cos(ph), sx = Math.sin(ph);
    const pos = [
      center[0] + (e1[0] * cx + e2[0] * sx) * r + nrm[0] * z,
      center[1] + (e1[1] * cx + e2[1] * sx) * r + nrm[1] * z,
      center[2] + (e1[2] * cx + e2[2] * sx) * r + nrm[2] * z,
    ];

    const menc = mc + md * (1 - (1 + r / Rd) * Math.exp(-r / Rd));
    const vc = Math.sqrt(menc / r) * (1 + (rng() - 0.5) * 0.06);
    // Tangential direction: nrm x radial.
    const rad = [e1[0] * cx + e2[0] * sx, e1[1] * cx + e2[1] * sx, e1[2] * cx + e2[2] * sx];
    const tan = cross(nrm, rad);

    const k = (offset + i) * 4;
    posMass[k] = pos[0]; posMass[k + 1] = pos[1]; posMass[k + 2] = pos[2];
    posMass[k + 3] = mp;
    vel[k] = bulk[0] + tan[0] * vc;
    vel[k + 1] = bulk[1] + tan[1] * vc;
    vel[k + 2] = bulk[2] + tan[2] * vc;
    vel[k + 3] = 0;
  }
}

// Plummer sphere (same sampling as the C++ core), used for the "cloud" scene
// and for flung mass blobs.
export function plummerBlob({ posMass, vel, offset, n, center, bulk, mass, scale, rng }) {
  const mp = mass / n;
  for (let i = 0; i < n; i++) {
    let r;
    do {
      const u = Math.max(rng(), 1e-7);
      r = scale / Math.sqrt(Math.pow(u, -2 / 3) - 1);
    } while (r > scale * 8);

    const sphere = () => {
      const cz = rng() * 2 - 1;
      const ph = rng() * Math.PI * 2;
      const s = Math.sqrt(Math.max(0, 1 - cz * cz));
      return [s * Math.cos(ph), s * Math.sin(ph), cz];
    };

    let q, g;
    do { q = rng(); g = rng() * 0.1; } while (g > q * q * Math.pow(1 - q * q, 3.5));
    const vesc = Math.sqrt(2 * mass / scale) * Math.pow(1 + (r / scale) ** 2, -0.25);
    const v = q * vesc;

    const dp = sphere(), dv = sphere();
    const k = (offset + i) * 4;
    posMass[k] = center[0] + r * dp[0];
    posMass[k + 1] = center[1] + r * dp[1];
    posMass[k + 2] = center[2] + r * dp[2];
    posMass[k + 3] = mp;
    vel[k] = bulk[0] + v * dv[0];
    vel[k + 1] = bulk[1] + v * dv[1];
    vel[k + 2] = bulk[2] + v * dv[2];
    vel[k + 3] = 0;
  }
}

// Builds a scene into fresh arrays of `capacity` slots, with `n` active.
export function makeScene(name, n, capacity, seed = 12345) {
  const rng = mulberry32(seed);
  const posMass = new Float32Array(capacity * 4);
  const vel = new Float32Array(capacity * 4);

  if (name === "collision") {
    const n1 = Math.floor(n * 0.6), n2 = n - n1;
    // Two disks on a slightly offset approach so the encounter is a grazing
    // one — head-on mergers are over too fast to be interesting.
    disk({ posMass, vel, offset: 0, n: n1, rng,
           center: [-1.5, 0.12, -0.3], bulk: [0.32, 0, 0.06],
           axis: [0.25, 1, 0.12], radius: 1.25, mass: 1.0 });
    disk({ posMass, vel, offset: n1, n: n2, rng,
           center: [1.5, -0.12, 0.3], bulk: [-0.45, 0, -0.08],
           axis: [-0.45, 1, 0.35], radius: 1.0, mass: 0.65 });
  } else if (name === "single") {
    disk({ posMass, vel, offset: 0, n, rng,
           center: [0, 0, 0], bulk: [0, 0, 0],
           axis: [0.18, 1, 0.1], radius: 1.4, mass: 1.2 });
  } else {  // cloud: cold-ish Plummer sphere that collapses and virializes
    plummerBlob({ posMass, vel, offset: 0, n, rng,
                  center: [0, 0, 0], bulk: [0, 0, 0], mass: 1.2, scale: 0.9 });
    // Cool it slightly so it visibly collapses first.
    for (let i = 0; i < n; i++) {
      vel[i * 4] *= 0.6; vel[i * 4 + 1] *= 0.6; vel[i * 4 + 2] *= 0.6;
    }
  }

  // Mean particle mass for sprite sizing (exclude the heavy central masses).
  let msum = 0, mcount = 0;
  for (let i = 0; i < n; i++) {
    const m = posMass[i * 4 + 3];
    if (m < 0.01) { msum += m; mcount++; }
  }
  const meanMass = mcount ? msum / mcount : 1;
  return { posMass, vel, meanMass };
}
