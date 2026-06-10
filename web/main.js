// WebGPU n-body demo: tiled all-pairs gravity in compute, KDK leapfrog,
// additive HDR sprites + bloom + tonemap. No build step, no dependencies.

import { OrbitCamera, add, scale, sub, dot } from "./camera.js";
import { SIM_WGSL, RENDER_WGSL, POST_WGSL, COMPOSITE_WGSL, FADE_WGSL,
         VOLUME_WGSL, VOLMARCH_WGSL } from "./shaders.js";
import { makeScene, plummerBlob } from "./galaxy.js";

const DT0 = 1 / 240;       // base timestep (sim units)
const EPS2 = 0.03 * 0.03;
const RESERVE = 16384;     // extra buffer slots for flung mass
const BLOB_N = 1536;       // particles per fling
const VMAX_COLOR = 1.3;    // speed -> color ramp ceiling
const BASE_SPRITE = 0.0085;

// Volumetric nebula grid: fixed cube so the density texture doesn't swim
// when the camera moves.
const GRID_DIM = 96;
const BOX_MIN = -5;
const BOX_SIZE = 10;
const VOL_FIXED = 2e7;     // mass -> fixed-point for atomic accumulation

const canvas = document.getElementById("gpu");
const overlay = document.getElementById("overlay");
const stats = document.getElementById("stats");

function showFallback(detail) {
  document.getElementById("fallback").hidden = false;
  document.getElementById("fallback-detail").textContent = detail || "";
}

function fail(detail) {
  showFallback(detail);
  throw new Error(detail);
}

async function init() {
  if (!navigator.gpu) fail("navigator.gpu is undefined in this browser.");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) fail("No WebGPU adapter found.");
  const device = await adapter.requestDevice();
  device.lost.then((info) => {
    if (info.reason !== "destroyed") showFallback(`GPU device lost: ${info.message}`);
  });

  const ctx = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });

  // --- pipelines -----------------------------------------------------------
  const simModule = device.createShaderModule({ code: SIM_WGSL });
  const renderModule = device.createShaderModule({ code: RENDER_WGSL });
  const postModule = device.createShaderModule({ code: POST_WGSL });
  const compositeModule = device.createShaderModule({ code: COMPOSITE_WGSL });
  const fadeModule = device.createShaderModule({ code: FADE_WGSL });
  const volumeModule = device.createShaderModule({ code: VOLUME_WGSL });
  const marchModule = device.createShaderModule({ code: VOLMARCH_WGSL });

  // One explicit layout shared by all three entry points. layout:"auto"
  // would derive a per-entry-point layout containing only the bindings that
  // entry point touches (forces never reads vel), and a bind group built for
  // one would be invalid for the others.
  const simBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const simLayout = device.createPipelineLayout({ bindGroupLayouts: [simBGL] });
  const simPipelines = Object.fromEntries(
    ["kick_drift", "forces", "kick"].map((entry) => [
      entry,
      device.createComputePipeline({
        layout: simLayout,
        compute: { module: simModule, entryPoint: entry },
      }),
    ]),
  );

  const spritePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: renderModule, entryPoint: "vs" },
    fragment: {
      module: renderModule,
      entryPoint: "fs",
      targets: [{
        format: "rgba16float",
        blend: {
          color: { srcFactor: "one", dstFactor: "one" },
          alpha: { srcFactor: "one", dstFactor: "one" },
        },
      }],
    },
    primitive: { topology: "triangle-list" },
  });

  // Multiplies the HDR buffer by the blend constant — the trails control.
  const fadePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: fadeModule, entryPoint: "vs" },
    fragment: {
      module: fadeModule,
      entryPoint: "fs",
      targets: [{
        format: "rgba16float",
        blend: {
          color: { srcFactor: "zero", dstFactor: "constant" },
          alpha: { srcFactor: "zero", dstFactor: "constant" },
        },
      }],
    },
    primitive: { topology: "triangle-list" },
  });

  const brightPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: postModule, entryPoint: "fullscreen" },
    fragment: { module: postModule, entryPoint: "brightpass",
                targets: [{ format: "rgba16float" }] },
    primitive: { topology: "triangle-list" },
  });

  const blurPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: postModule, entryPoint: "fullscreen" },
    fragment: { module: postModule, entryPoint: "blur",
                targets: [{ format: "rgba16float" }] },
    primitive: { topology: "triangle-list" },
  });

  const compositePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: compositeModule, entryPoint: "vs" },
    fragment: { module: compositeModule, entryPoint: "fs",
                targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const splatPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: volumeModule, entryPoint: "splat" },
  });
  const toTexPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: volumeModule, entryPoint: "to_tex" },
  });
  const marchPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: marchModule, entryPoint: "vs" },
    fragment: { module: marchModule, entryPoint: "fs",
                targets: [{ format: "rgba16float" }] },
    primitive: { topology: "triangle-list" },
  });

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  // --- volumetric nebula resources (camera-independent, so static) --------
  const gridBuf = device.createBuffer({
    size: GRID_DIM ** 3 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const volTex = device.createTexture({
    size: [GRID_DIM, GRID_DIM, GRID_DIM],
    dimension: "3d",
    format: "rgba16float",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const volTexView = volTex.createView();
  const volU = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const marchU = device.createBuffer({
    size: 80,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const compU = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const toTexBind = device.createBindGroup({
    layout: toTexPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 1, resource: { buffer: gridBuf } },
      { binding: 2, resource: { buffer: volU } },
      { binding: 3, resource: volTexView },
    ],
  });
  const marchBind = device.createBindGroup({
    layout: marchPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: volTexView },
      { binding: 1, resource: sampler },
      { binding: 2, resource: { buffer: marchU } },
    ],
  });
  // Lens positions: the two central masses, read back from the GPU with a
  // frame of latency.
  const coreStaging = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // --- mutable sim state ---------------------------------------------------
  const state = {
    n: 32768,
    scene: "collision",
    capacity: 0,
    count: 0,
    spawnCursor: 0,        // next reserve slot for flung mass
    meanMass: 1,
    paused: false,
    timescale: 1,
    trails: 0.3,
    nebula: 0.55,
    lensing: 0.6,
    buffers: null,         // { posMass, vel, accel, simU, renderU }
    simBind: null,
    spriteBind: null,
    splatBind: null,
    coreSlots: [],         // buffer indices of the heavy central masses
    cores: [],             // their world positions (readback, 1 frame stale)
    readbackBusy: false,
    seed: 1,
  };

  const camera = new OrbitCamera();

  function createBuffers() {
    if (state.buffers) {
      for (const b of Object.values(state.buffers)) b.destroy();
    }
    const cap = state.n + RESERVE;
    const bytes = cap * 16;
    // COPY_SRC: the lens effect reads the core positions back each frame.
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
                  GPUBufferUsage.COPY_SRC;
    state.capacity = cap;
    state.buffers = {
      posMass: device.createBuffer({ size: bytes, usage }),
      vel: device.createBuffer({ size: bytes, usage }),
      accel: device.createBuffer({ size: bytes, usage }),
      simU: device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      renderU: device.createBuffer({
        size: 112,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
    };
    const b = state.buffers;
    state.simBind = device.createBindGroup({
      layout: simBGL,
      entries: [
        { binding: 0, resource: { buffer: b.posMass } },
        { binding: 1, resource: { buffer: b.vel } },
        { binding: 2, resource: { buffer: b.accel } },
        { binding: 3, resource: { buffer: b.simU } },
      ],
    });
    state.spriteBind = device.createBindGroup({
      layout: spritePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: b.posMass } },
        { binding: 1, resource: { buffer: b.vel } },
        { binding: 2, resource: { buffer: b.renderU } },
      ],
    });
    state.splatBind = device.createBindGroup({
      layout: splatPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: b.posMass } },
        { binding: 1, resource: { buffer: gridBuf } },
        { binding: 2, resource: { buffer: volU } },
      ],
    });
  }

  function writeVolUniform() {
    const buf = new ArrayBuffer(48);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    f.set([BOX_MIN, BOX_MIN, BOX_MIN], 0);
    u[3] = state.count;
    f.set([1 / BOX_SIZE, 1 / BOX_SIZE, 1 / BOX_SIZE], 4);
    u[7] = GRID_DIM;
    f[8] = state.meanMass * 20;        // deposit cap per body
    f[9] = VOL_FIXED;
    f[10] = 1 / (state.meanMass * 400); // cell mass -> O(1) density
    device.queue.writeBuffer(volU, 0, buf);
  }

  // High speeds split into two substeps so dt stays integrable.
  function substeps() {
    return state.timescale >= 2 ? 2 : 1;
  }

  function writeSimUniform() {
    const u = new ArrayBuffer(16);
    const f = new Float32Array(u);
    const i = new Uint32Array(u);
    f[0] = (DT0 * state.timescale) / substeps();
    f[1] = EPS2;
    i[2] = state.count;
    device.queue.writeBuffer(state.buffers.simU, 0, u);
  }

  function dispatchSim(encoder, entries) {
    const wgs = Math.ceil(state.count / 256);
    const pass = encoder.beginComputePass();
    for (const e of entries) {
      pass.setPipeline(simPipelines[e]);
      pass.setBindGroup(0, state.simBind);
      pass.dispatchWorkgroups(wgs);
    }
    pass.end();
  }

  function reset() {
    createBuffers();
    const { posMass, vel, meanMass } = makeScene(
      state.scene, state.n, state.capacity, state.seed++);
    state.count = state.n;
    state.spawnCursor = 0;
    state.meanMass = meanMass;
    // The heavy central masses act as gravitational lenses; their initial
    // positions are known, then the readback keeps them fresh.
    if (state.scene === "collision") {
      const n1 = Math.floor(state.n * 0.6);
      state.coreSlots = [0, n1];
      state.cores = [[-1.9, 0.15, -0.4], [1.9, -0.15, 0.4]];
    } else if (state.scene === "single") {
      state.coreSlots = [0];
      state.cores = [[0, 0, 0]];
    } else {
      state.coreSlots = [];
      state.cores = [];
    }
    device.queue.writeBuffer(state.buffers.posMass, 0, posMass);
    device.queue.writeBuffer(state.buffers.vel, 0, vel);
    device.queue.writeBuffer(
      state.buffers.accel, 0, new Float32Array(state.capacity * 4));
    writeSimUniform();
    writeVolUniform();
    // Prime accelerations so the first half-kick uses real forces.
    const encoder = device.createCommandEncoder();
    dispatchSim(encoder, ["forces"]);
    device.queue.submit([encoder.finish()]);
  }

  // --- render targets (HDR + half-res bloom chain + nebula) ----------------
  let hdrTex = null, hdrView = null;
  let bloomA = null, bloomB = null, bloomAView = null, bloomBView = null;
  let nebTex = null, nebView = null;
  let brightBind = null, blurBinds = null, compositeBind = null;
  const blurU = [0, 1, 2, 3].map(() =>
    device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }));

  function configureSize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width === w && canvas.height === h && hdrTex) return;
    canvas.width = w;
    canvas.height = h;
    overlay.width = w;
    overlay.height = h;

    for (const t of [hdrTex, bloomA, bloomB, nebTex]) t?.destroy();
    const mk = (tw, th) => device.createTexture({
      size: [tw, th],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    hdrTex = mk(w, h);
    const hw = Math.max(1, w >> 1), hh = Math.max(1, h >> 1);
    bloomA = mk(hw, hh);
    bloomB = mk(hw, hh);
    nebTex = mk(hw, hh);
    hdrView = hdrTex.createView();
    bloomAView = bloomA.createView();
    bloomBView = bloomB.createView();
    nebView = nebTex.createView();

    brightBind = device.createBindGroup({
      layout: brightPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: hdrView },
        { binding: 1, resource: sampler },
      ],
    });
    // Ping-pong: A -h-> B -v-> A -h(wide)-> B -v(wide)-> A.
    const dirs = [
      [1 / hw, 0], [0, 1 / hh],
      [2.5 / hw, 0], [0, 2.5 / hh],
    ];
    blurBinds = dirs.map((d, i) => {
      device.queue.writeBuffer(blurU[i], 0, new Float32Array([d[0], d[1], 0, 0]));
      return {
        bind: device.createBindGroup({
          layout: blurPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: i % 2 === 0 ? bloomAView : bloomBView },
            { binding: 1, resource: sampler },
            { binding: 2, resource: { buffer: blurU[i] } },
          ],
        }),
        target: i % 2 === 0 ? bloomBView : bloomAView,
      };
    });
    compositeBind = device.createBindGroup({
      layout: compositePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: hdrView },
        { binding: 1, resource: sampler },
        { binding: 2, resource: bloomAView },
        { binding: 3, resource: nebView },
        { binding: 4, resource: { buffer: compU } },
      ],
    });
  }
  new ResizeObserver(configureSize).observe(canvas);

  // --- interaction ---------------------------------------------------------
  const octx = overlay.getContext("2d");
  const pointers = new Map();   // pointerId -> {x, y}
  let drag = null;              // { mode: "orbit"|"fling", x0, y0, x, y }
  let pinchDist = 0;

  // Point on the plane through the origin perpendicular to the view, under
  // the cursor — where flung mass appears.
  function unproject(px, py) {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const ray = camera.pixelRay(px * dpr, py * dpr, canvas.width, canvas.height);
    const eye = camera.eye();
    const n = camera.forward();
    const t = dot(scale(eye, -1), n) / dot(ray, n);
    return add(eye, scale(ray, t));
  }

  function fling(x0, y0, x1, y1) {
    const a = unproject(x0, y0);
    const b = unproject(x1, y1);
    const v = scale(sub(b, a), 0.55);

    const blob = {
      posMass: new Float32Array(BLOB_N * 4),
      vel: new Float32Array(BLOB_N * 4),
    };
    plummerBlob({
      ...blob, offset: 0, n: BLOB_N, center: a, bulk: v,
      mass: 0.05, scale: 0.07, rng: Math.random, tag: 2,
    });

    // Reserve slots wrap around: the oldest flung blob gets overwritten.
    const slot = state.n + (state.spawnCursor % RESERVE);
    const fit = Math.min(BLOB_N, state.capacity - slot);
    device.queue.writeBuffer(state.buffers.posMass, slot * 16,
                             blob.posMass, 0, fit * 4);
    device.queue.writeBuffer(state.buffers.vel, slot * 16, blob.vel, 0, fit * 4);
    state.spawnCursor = (state.spawnCursor + BLOB_N) % RESERVE;
    state.count = Math.max(state.count, slot + fit);
    writeSimUniform();
    writeVolUniform();
  }

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 2) {
      const [p, q] = [...pointers.values()];
      pinchDist = Math.hypot(p.x - q.x, p.y - q.y);
      drag = null;
      return;
    }
    const mode = e.button === 2 || e.shiftKey ? "fling" : "orbit";
    drag = { mode, x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY };
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch {
      // Pointer may already be gone (e.g. pen lifted); drag still works.
    }
  });
  canvas.addEventListener("pointermove", (e) => {
    const p = pointers.get(e.pointerId);
    if (p) {
      p.x = e.clientX;
      p.y = e.clientY;
    }
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinchDist > 0) camera.zoom((pinchDist - d) * 4);
      pinchDist = d;
      return;
    }
    if (!drag) return;
    if (drag.mode === "orbit") {
      camera.rotate(e.clientX - drag.x, e.clientY - drag.y);
    }
    drag.x = e.clientX;
    drag.y = e.clientY;
  });
  const endPointer = (e) => {
    pointers.delete(e.pointerId);
    if (pointers.size < 2) pinchDist = 0;
    if (drag?.mode === "fling") {
      fling(drag.x0, drag.y0, e.clientX, e.clientY);
    }
    drag = null;
  };
  canvas.addEventListener("pointerup", endPointer);
  canvas.addEventListener("pointercancel", endPointer);
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    camera.zoom(e.deltaY);
  }, { passive: false });

  function drawFlingArrow() {
    octx.clearRect(0, 0, overlay.width, overlay.height);
    if (!drag || drag.mode !== "fling") return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    octx.strokeStyle = "rgba(138, 180, 255, 0.9)";
    octx.fillStyle = "rgba(138, 180, 255, 0.9)";
    octx.lineWidth = 1.5 * dpr;
    octx.beginPath();
    octx.moveTo(drag.x0 * dpr, drag.y0 * dpr);
    octx.lineTo(drag.x * dpr, drag.y * dpr);
    octx.stroke();
    octx.beginPath();
    octx.arc(drag.x0 * dpr, drag.y0 * dpr, 4 * dpr, 0, Math.PI * 2);
    octx.fill();
  }

  // --- UI ------------------------------------------------------------------
  const ui = {
    scene: document.getElementById("scene"),
    count: document.getElementById("count"),
    speed: document.getElementById("speed"),
    trails: document.getElementById("trails"),
    nebula: document.getElementById("nebula"),
    lensing: document.getElementById("lensing"),
    pause: document.getElementById("pause"),
    reset: document.getElementById("reset"),
  };
  ui.scene.addEventListener("change", () => {
    state.scene = ui.scene.value;
    reset();
  });
  ui.count.addEventListener("change", () => {
    state.n = parseInt(ui.count.value, 10);
    reset();
  });
  ui.speed.addEventListener("input", () => {
    state.timescale = parseFloat(ui.speed.value);
    writeSimUniform();
  });
  ui.trails.addEventListener("input", () => {
    state.trails = parseFloat(ui.trails.value);
  });
  ui.nebula.addEventListener("input", () => {
    state.nebula = parseFloat(ui.nebula.value);
  });
  ui.lensing.addEventListener("input", () => {
    state.lensing = parseFloat(ui.lensing.value);
  });
  function togglePause() {
    state.paused = !state.paused;
    ui.pause.textContent = state.paused ? "resume" : "pause";
  }
  ui.pause.addEventListener("click", togglePause);
  ui.reset.addEventListener("click", reset);
  addEventListener("keydown", (e) => {
    if (e.code === "Space" && e.target.tagName !== "BUTTON") {
      e.preventDefault();
      togglePause();
    } else if (e.key === "r" || e.key === "R") {
      reset();
    }
  });

  // --- frame loop ----------------------------------------------------------
  let lastT = performance.now();
  let fpsAvg = 0;

  function frame(now) {
    const dtMs = now - lastT;
    lastT = now;
    fpsAvg = fpsAvg * 0.95 + (1000 / Math.max(dtMs, 0.1)) * 0.05;

    configureSize();
    const aspect = canvas.width / canvas.height;

    // Render uniforms.
    const viewProj = camera.viewProj(aspect);
    const u = new Float32Array(28);
    u.set(viewProj, 0);
    const r = camera.right(), up = camera.up();
    u.set([r[0], r[1], r[2], BASE_SPRITE * camera.dist / 4.6], 16);
    u.set([up[0], up[1], up[2], state.meanMass], 20);
    u.set([VMAX_COLOR, 0, 0, 0], 24);
    device.queue.writeBuffer(state.buffers.renderU, 0, u);

    // Ray-march uniforms (camera basis + nebula density).
    const eye = camera.eye(), fwd = camera.forward();
    const tanF = Math.tan(camera.fovY / 2);
    const m = new Float32Array(20);
    m.set([eye[0], eye[1], eye[2], tanF], 0);
    m.set([r[0], r[1], r[2], aspect], 4);
    m.set([up[0], up[1], up[2], state.nebula], 8);
    m.set([fwd[0], fwd[1], fwd[2], 0], 12);
    m.set([BOX_MIN, BOX_MIN, BOX_MIN, BOX_SIZE], 16);
    device.queue.writeBuffer(marchU, 0, m);

    // Lens uniforms: project the (frame-stale) core positions to screen uv.
    const cu = new Float32Array(12);
    const projectLens = (slot, p) => {
      const w = viewProj[3] * p[0] + viewProj[7] * p[1] + viewProj[11] * p[2] + viewProj[15];
      if (w < 0.1) return;
      const cx = viewProj[0] * p[0] + viewProj[4] * p[1] + viewProj[8] * p[2] + viewProj[12];
      const cy = viewProj[1] * p[0] + viewProj[5] * p[1] + viewProj[9] * p[2] + viewProj[13];
      cu.set([cx / w * 0.5 + 0.5, 0.5 - cy / w * 0.5,
              0.0011 * state.lensing, 1], slot * 4);
    };
    if (state.lensing > 0) state.cores.forEach((p, i) => i < 2 && projectLens(i, p));
    cu[8] = aspect;
    device.queue.writeBuffer(compU, 0, cu);

    const encoder = device.createCommandEncoder();
    if (!state.paused) {
      for (let s = 0; s < substeps(); s++) {
        dispatchSim(encoder, ["kick_drift", "forces", "kick"]);
      }
    }

    // Volumetric nebula: deposit mass density, convert to a filterable 3D
    // texture, then ray-march it at half resolution.
    encoder.clearBuffer(gridBuf);
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(splatPipeline);
      pass.setBindGroup(0, state.splatBind);
      pass.dispatchWorkgroups(Math.ceil(state.count / 256));
      pass.setPipeline(toTexPipeline);
      pass.setBindGroup(0, toTexBind);
      const wg = Math.ceil(GRID_DIM / 4);
      pass.dispatchWorkgroups(wg, wg, wg);
      pass.end();
    }
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: nebView, loadOp: "clear", storeOp: "store" }],
      });
      pass.setPipeline(marchPipeline);
      pass.setBindGroup(0, marchBind);
      pass.draw(3);
      pass.end();
    }

    // Queue the core-position readback for the lenses.
    const wantReadback = !state.readbackBusy && state.coreSlots.length > 0;
    if (wantReadback) {
      state.coreSlots.forEach((slot, i) => {
        encoder.copyBufferToBuffer(
          state.buffers.posMass, slot * 16, coreStaging, i * 16, 16);
      });
    }

    // Fade (trails) then accumulate sprites into HDR.
    const fadePass = encoder.beginRenderPass({
      colorAttachments: [{ view: hdrView, loadOp: "load", storeOp: "store" }],
    });
    fadePass.setPipeline(fadePipeline);
    const k = state.paused ? 1 : state.trails;
    fadePass.setBlendConstant({ r: k, g: k, b: k, a: k });
    fadePass.draw(3);
    fadePass.end();

    const spritePass = encoder.beginRenderPass({
      colorAttachments: [{ view: hdrView, loadOp: "load", storeOp: "store" }],
    });
    spritePass.setPipeline(spritePipeline);
    spritePass.setBindGroup(0, state.spriteBind);
    spritePass.draw(6, state.count);
    spritePass.end();

    // Bloom chain: brightpass to half-res, then ping-pong blurs.
    const bright = encoder.beginRenderPass({
      colorAttachments: [{ view: bloomAView, loadOp: "clear", storeOp: "store" }],
    });
    bright.setPipeline(brightPipeline);
    bright.setBindGroup(0, brightBind);
    bright.draw(3);
    bright.end();

    for (const { bind, target } of blurBinds) {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: target, loadOp: "clear", storeOp: "store" }],
      });
      pass.setPipeline(blurPipeline);
      pass.setBindGroup(0, bind);
      pass.draw(3);
      pass.end();
    }

    const compositePass = encoder.beginRenderPass({
      colorAttachments: [{
        view: ctx.getCurrentTexture().createView(),
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    compositePass.setPipeline(compositePipeline);
    compositePass.setBindGroup(0, compositeBind);
    compositePass.draw(3);
    compositePass.end();

    device.queue.submit([encoder.finish()]);

    if (wantReadback) {
      state.readbackBusy = true;
      const slots = state.coreSlots;
      coreStaging.mapAsync(GPUMapMode.READ).then(() => {
        const f = new Float32Array(coreStaging.getMappedRange().slice(0));
        coreStaging.unmap();
        if (state.coreSlots === slots) {  // scene unchanged meanwhile
          state.cores = slots.map((_, i) => [f[i * 4], f[i * 4 + 1], f[i * 4 + 2]]);
        }
        state.readbackBusy = false;
      }).catch(() => { state.readbackBusy = false; });
    }

    drawFlingArrow();
    const gpairs = state.paused
      ? 0
      : (state.count ** 2 * substeps() * fpsAvg) / 1e9;
    stats.textContent =
      `${fpsAvg.toFixed(0).padStart(3)} fps\n` +
      `${state.count.toLocaleString()} bodies\n` +
      `${gpairs.toFixed(1)} Gpair/s`;
    requestAnimationFrame(frame);
  }

  // Browsers may restore form values across reloads; trust the controls.
  state.scene = ui.scene.value;
  state.n = parseInt(ui.count.value, 10);
  state.timescale = parseFloat(ui.speed.value);
  state.trails = parseFloat(ui.trails.value);
  state.nebula = parseFloat(ui.nebula.value);
  state.lensing = parseFloat(ui.lensing.value);

  configureSize();
  reset();
  requestAnimationFrame(frame);
}

init().catch((e) => {
  console.error(e);
  showFallback(String(e));
});
