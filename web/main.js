// WebGPU n-body demo: tiled all-pairs gravity in compute, KDK leapfrog,
// additive HDR sprites + tonemap. No build step, no dependencies.

import { OrbitCamera, add, scale, sub, dot } from "./camera.js";
import { SIM_WGSL, RENDER_WGSL, COMPOSITE_WGSL } from "./shaders.js";
import { makeScene, plummerBlob } from "./galaxy.js";

const DT0 = 1 / 240;       // base timestep (sim units)
const EPS2 = 0.015 * 0.015;
const RESERVE = 16384;     // extra buffer slots for flung mass
const BLOB_N = 1536;       // particles per fling
const VMAX_COLOR = 1.3;    // speed -> color ramp ceiling
const BASE_SPRITE = 0.0085;

const canvas = document.getElementById("gpu");
const overlay = document.getElementById("overlay");
const stats = document.getElementById("stats");

function fail(detail) {
  document.getElementById("fallback").hidden = false;
  document.getElementById("fallback-detail").textContent = detail || "";
  throw new Error(detail);
}

async function init() {
  if (!navigator.gpu) fail("navigator.gpu is undefined in this browser.");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) fail("No WebGPU adapter found.");
  const device = await adapter.requestDevice();
  device.lost.then((info) => {
    if (info.reason !== "destroyed") fail(`GPU device lost: ${info.message}`);
  });

  const ctx = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });

  // --- pipelines -----------------------------------------------------------
  const simModule = device.createShaderModule({ code: SIM_WGSL });
  const renderModule = device.createShaderModule({ code: RENDER_WGSL });
  const compositeModule = device.createShaderModule({ code: COMPOSITE_WGSL });

  const simPipelines = Object.fromEntries(
    ["kick_drift", "forces", "kick"].map((entry) => [
      entry,
      device.createComputePipeline({
        layout: "auto",
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

  const compositePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: compositeModule, entryPoint: "vs" },
    fragment: {
      module: compositeModule,
      entryPoint: "fs",
      targets: [{ format }],
    },
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
      layout: simPipelines.forces.getBindGroupLayout(0),
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

  function writeSimUniform() {
    const u = new ArrayBuffer(16);
    const f = new Float32Array(u);
    const i = new Uint32Array(u);
    f[0] = DT0 * state.timescale;
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

  // --- HDR target ----------------------------------------------------------
  let hdrTex = null, hdrView = null, compositeBind = null;

  function configureSize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width === w && canvas.height === h && hdrTex) return;
    canvas.width = w;
    canvas.height = h;
    overlay.width = w;
    overlay.height = h;
    if (hdrTex) hdrTex.destroy();
    hdrTex = device.createTexture({
      size: [w, h],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    hdrView = hdrTex.createView();
    compositeBind = device.createBindGroup({
      layout: compositePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: hdrView },
        { binding: 1, resource: sampler },
      ],
    });
  }
  new ResizeObserver(configureSize).observe(canvas);

  // --- interaction ---------------------------------------------------------
  const octx = overlay.getContext("2d");
  let drag = null; // { mode: "orbit"|"fling", x0, y0, x, y }

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
      mass: 0.05, scale: 0.07, rng: Math.random,
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
    const mode = e.button === 2 || e.shiftKey ? "fling" : "orbit";
    drag = { mode, x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    if (drag.mode === "orbit") {
      camera.rotate(e.clientX - drag.x, e.clientY - drag.y);
    }
    drag.x = e.clientX;
    drag.y = e.clientY;
  });
  canvas.addEventListener("pointerup", (e) => {
    if (drag?.mode === "fling") {
      fling(drag.x0, drag.y0, e.clientX, e.clientY);
    }
    drag = null;
  });
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
      dispatchSim(encoder, ["kick_drift", "forces", "kick"]);
    }

    const spritePass = encoder.beginRenderPass({
      colorAttachments: [{
        view: hdrView,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    spritePass.setPipeline(spritePipeline);
    spritePass.setBindGroup(0, state.spriteBind);
    spritePass.draw(6, state.count);
    spritePass.end();

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
      : (state.count ** 2 * fpsAvg) / 1e9;
    stats.textContent =
      `${fpsAvg.toFixed(0).padStart(3)} fps\n` +
      `${state.count.toLocaleString()} bodies\n` +
      `${gpairs.toFixed(1)} Gpair/s`;
    requestAnimationFrame(frame);
  }

  configureSize();
  reset();
  requestAnimationFrame(frame);
}

init().catch((e) => {
  console.error(e);
  const fb = document.getElementById("fallback");
  if (fb.hidden) {
    fb.hidden = false;
    document.getElementById("fallback-detail").textContent = String(e);
  }
});
