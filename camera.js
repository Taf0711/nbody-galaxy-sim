// Orbit camera + the handful of vec/mat helpers the demo needs. Column-major
// mat4 to match WGSL.

export function vec3(x = 0, y = 0, z = 0) { return [x, y, z]; }

export function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
export function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
export function scale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
export function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
export function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1],
          a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]];
}
export function norm(a) {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}

export function perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  const k = far / (near - far);
  // prettier-ignore
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, k, -1,
    0, 0, k * near, 0,
  ]);
}

export function lookAt(eye, target, upHint) {
  const f = norm(sub(target, eye));        // forward
  const r = norm(cross(f, upHint));        // right
  const u = cross(r, f);                   // true up
  // prettier-ignore
  return new Float32Array([
    r[0], u[0], -f[0], 0,
    r[1], u[1], -f[1], 0,
    r[2], u[2], -f[2], 0,
    -dot(r, eye), -dot(u, eye), dot(f, eye), 1,
  ]);
}

export function mul4(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  return out;
}

export class OrbitCamera {
  constructor() {
    this.target = vec3(0, 0, 0);
    this.yaw = 0.6;
    this.pitch = 0.35;
    this.dist = 5.6;
    this.fovY = (50 * Math.PI) / 180;
  }

  eye() {
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    return add(this.target,
               scale([cp * cy, sp, cp * sy], this.dist));
  }

  forward() { return norm(sub(this.target, this.eye())); }
  right() { return norm(cross(this.forward(), [0, 1, 0])); }
  up() { return cross(this.right(), this.forward()); }

  rotate(dx, dy) {
    this.yaw += dx * 0.005;
    this.pitch = Math.min(1.5, Math.max(-1.5, this.pitch + dy * 0.005));
  }

  zoom(dy) {
    this.dist = Math.min(30, Math.max(0.8, this.dist * Math.exp(dy * 0.0012)));
  }

  viewProj(aspect) {
    return mul4(perspective(this.fovY, aspect, 0.05, 200),
                lookAt(this.eye(), this.target, [0, 1, 0]));
  }

  // World-space ray through a canvas pixel (for unprojecting drags onto the
  // plane through the origin perpendicular to the view direction).
  pixelRay(px, py, width, height) {
    const ndcX = (px / width) * 2 - 1;
    const ndcY = 1 - (py / height) * 2;
    const tanF = Math.tan(this.fovY / 2);
    const aspect = width / height;
    const f = this.forward(), r = this.right(), u = this.up();
    return norm(add(f, add(scale(r, ndcX * tanF * aspect),
                           scale(u, ndcY * tanF))));
  }
}
