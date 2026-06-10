// WebGPU n-body demo: tiled all-pairs gravity in compute, KDK leapfrog,
// additive HDR sprites + bloom + tonemap. No build step, no dependencies.

import { OrbitCamera, add, scale, sub, dot } from "./camera.js";
import { SIM_WGSL, RENDER_WGSL, POST_WGSL, COMPOSITE_WGSL, FADE_WGSL } from "./shaders.js";
import { makeScene, plummerBlob } from "./galaxy.js";

const DT0 = 1 / 240;       // base timestep (sim units)
const EPS2 = 0.03 * 0.03;
const RESERVE = 16384;     // extra buffer slots for flung mass
const BLOB_N = 1536;       // particles per fling
const VMAX_COLOR = 1.3;    // speed -> color ramp ceiling
const BASE_SPRITE = 0.0085;

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

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
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
    buffers: null,         // { posMass, vel, accel, simU, renderU }
    simBind: null,
    spriteBind: null,
    seed: 1,
  };

  const camera = new OrbitCamera();

  function createBuffers() {
    if (state.buffers) {
      for (const b of Object.values(state.buffers)) b.destroy();
    }
    const cap = state.n + RESERVE;
    const bytes = cap * 16;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
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
    device.queue.writeBuffer(state.buffers.posMass, 0, posMass);
    device.queue.writeBuffer(state.buffers.vel, 0, vel);
    device.queue.writeBuffer(
      state.buffers.accel, 0, new Float32Array(state.capacity * 4));
    writeSimUniform();
    // Prime accelerations so the first half-kick uses real forces.
    const encoder = device.createCommandEncoder();
    dispatchSim(encoder, ["forces"]);
    device.queue.submit([encoder.finish()]);
  }

  // --- render targets (HDR + half-res bloom chain) -------------------------
  let hdrTex = null, hdrView = null;
  let bloomA = null, bloomB = null, bloomAView = null, bloomBView = null;
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

    for (const t of [hdrTex, bloomA, bloomB]) t?.destroy();
    const mk = (tw, th) => device.createTexture({
      size: [tw, th],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    hdrTex = mk(w, h);
    const hw = Math.max(1, w >> 1), hh = Math.max(1, h >> 1);
    bloomA = mk(hw, hh);
    bloomB = mk(hw, hh);
    hdrView = hdrTex.createView();
    bloomAView = bloomA.createView();
    bloomBView = bloomB.createView();

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
    const u = new Float32Array(28);
    u.set(camera.viewProj(aspect), 0);
    const r = camera.right(), up = camera.up();
    u.set([r[0], r[1], r[2], BASE_SPRITE * camera.dist / 4.6], 16);
    u.set([up[0], up[1], up[2], state.meanMass], 20);
    u.set([VMAX_COLOR, 0, 0, 0], 24);
    device.queue.writeBuffer(state.buffers.renderU, 0, u);

    const encoder = device.createCommandEncoder();
    if (!state.paused) {
      for (let s = 0; s < substeps(); s++) {
        dispatchSim(encoder, ["kick_drift", "forces", "kick"]);
      }
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

  configureSize();
  reset();
  requestAnimationFrame(frame);
}

init().catch((e) => {
  console.error(e);
  showFallback(String(e));
});
