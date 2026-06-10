#!/usr/bin/env python3
"""Render benchmark charts (SVG, no dependencies) and a markdown table from
bench/results.csv.

Usage: python3 tools/plot_bench.py [bench/results.csv]

Outputs:
  assets/bench_sweep.svg     ms/step vs N, log-log, one line per method
  assets/bench_scaling.svg   speedup vs thread count at fixed N
  bench/results.md           the full table, markdown
"""

import csv
import math
import os
import sys

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "bench/results.csv"
ASSETS = "assets"

COLORS = {
    "naive": "#c0392b",
    "naive_mt": "#e67e22",
    "simd": "#2980b9",
    "simd_mt": "#16a085",
    "bh_mt": "#8e44ad",
    "bh_quad_mt": "#d4488e",
}
LABELS = {
    "naive": "naive O(n²), 1 thread",
    "naive_mt": "naive O(n²), threaded",
    "simd": "NEON SIMD, 1 thread",
    "simd_mt": "NEON SIMD, threaded",
    "bh_mt": "Barnes-Hut θ=0.5, threaded",
    "bh_quad_mt": "Barnes-Hut quadrupole θ=0.65",
}

W, H = 760, 480
ML, MR, MT, MB = 70, 24, 40, 56  # margins


def read_rows():
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def svg_header(title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="-apple-system, Segoe UI, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2}" y="24" text-anchor="middle" font-size="16" '
        f'font-weight="600" fill="#222">{title}</text>',
    ]


def polyline(pts, color):
    p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    out = [f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="2.2"/>']
    for x, y in pts:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{color}"/>')
    return out


def sweep_chart(rows):
    rows = [r for r in rows if r["section"] == "sweep"]
    if not rows:
        return
    xs = sorted({int(r["n"]) for r in rows})
    ys = [float(r["ms_step"]) for r in rows]
    x0, x1 = math.log2(min(xs)), math.log2(max(xs))
    y0, y1 = math.log10(min(ys)) - 0.15, math.log10(max(ys)) + 0.15

    def X(n):
        return ML + (math.log2(n) - x0) / (x1 - x0) * (W - ML - MR)

    def Y(ms):
        return H - MB - (math.log10(ms) - y0) / (y1 - y0) * (H - MT - MB)

    s = svg_header("Force computation cost vs N (log–log, lower is better)")
    # gridlines: y decades, x powers of two present in data
    d = math.ceil(y0)
    while d <= y1:
        y = Y(10 ** d)
        lab = f"{10 ** d:g}"
        s.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{W - MR}" y2="{y:.1f}" '
                 f'stroke="#e3e3e3"/>')
        s.append(f'<text x="{ML - 8}" y="{y + 4:.1f}" text-anchor="end" '
                 f'font-size="11" fill="#555">{lab}</text>')
        d += 1
    for n in xs:
        x = X(n)
        lab = f"{n // 1024}k" if n >= 1024 else str(n)
        s.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{H - MB}" '
                 f'stroke="#f0f0f0"/>')
        s.append(f'<text x="{x:.1f}" y="{H - MB + 18}" text-anchor="middle" '
                 f'font-size="11" fill="#555">{lab}</text>')
    s.append(f'<text x="{ML - 52}" y="{(MT + H - MB) / 2}" font-size="12" fill="#333" '
             f'transform="rotate(-90 {ML - 52} {(MT + H - MB) / 2})" '
             f'text-anchor="middle">ms / step</text>')
    s.append(f'<text x="{(ML + W - MR) / 2}" y="{H - 12}" text-anchor="middle" '
             f'font-size="12" fill="#333">bodies (N)</text>')

    ly = MT + 14
    for method in ["naive", "naive_mt", "simd", "simd_mt", "bh_mt", "bh_quad_mt"]:
        pts = sorted(
            [(int(r["n"]), float(r["ms_step"])) for r in rows if r["method"] == method])
        if not pts:
            continue
        s += polyline([(X(n), Y(ms)) for n, ms in pts], COLORS[method])
        s.append(f'<rect x="{ML + 14}" y="{ly - 9}" width="12" height="12" '
                 f'fill="{COLORS[method]}"/>')
        s.append(f'<text x="{ML + 32}" y="{ly + 2}" font-size="12" fill="#333">'
                 f'{LABELS[method]}</text>')
        ly += 20
    s.append("</svg>")
    with open(os.path.join(ASSETS, "bench_sweep.svg"), "w") as f:
        f.write("\n".join(s))


def scaling_chart(rows):
    rows = [r for r in rows if r["section"] == "scaling"]
    if not rows:
        return
    pts = sorted([(int(r["threads"]), float(r["ms_step"])) for r in rows])
    base = dict(pts)[1]
    data = [(t, base / ms) for t, ms in pts]
    tmax = max(t for t, _ in data)
    smax = max(max(sp for _, sp in data), tmax) * 1.08

    def X(t):
        return ML + (t - 1) / (tmax - 1) * (W - ML - MR)

    def Y(sp):
        return H - MB - sp / smax * (H - MT - MB)

    n = rows[0]["n"]
    s = svg_header(f"Thread scaling, NEON kernel at N = {int(n):,}")
    for sp in range(1, int(smax) + 1):
        y = Y(sp)
        s.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{W - MR}" y2="{y:.1f}" '
                 f'stroke="#e3e3e3"/>')
        s.append(f'<text x="{ML - 8}" y="{y + 4:.1f}" text-anchor="end" '
                 f'font-size="11" fill="#555">{sp}×</text>')
    for t, _ in data:
        x = X(t)
        s.append(f'<text x="{x:.1f}" y="{H - MB + 18}" text-anchor="middle" '
                 f'font-size="11" fill="#555">{t}</text>')
    # ideal scaling reference
    s.append(f'<line x1="{X(1):.1f}" y1="{Y(1):.1f}" x2="{X(tmax):.1f}" '
             f'y2="{Y(tmax):.1f}" stroke="#bbb" stroke-dasharray="5,4"/>')
    s.append(f'<text x="{X(tmax) - 6:.1f}" y="{Y(tmax) + 16:.1f}" text-anchor="end" '
             f'font-size="11" fill="#888">ideal</text>')
    s += polyline([(X(t), Y(sp)) for t, sp in data], COLORS["simd_mt"])
    s.append(f'<text x="{ML - 52}" y="{(MT + H - MB) / 2}" font-size="12" fill="#333" '
             f'transform="rotate(-90 {ML - 52} {(MT + H - MB) / 2})" '
             f'text-anchor="middle">speedup vs 1 thread</text>')
    s.append(f'<text x="{(ML + W - MR) / 2}" y="{H - 12}" text-anchor="middle" '
             f'font-size="12" fill="#333">threads</text>')
    s.append("</svg>")
    with open(os.path.join(ASSETS, "bench_scaling.svg"), "w") as f:
        f.write("\n".join(s))


def markdown_table(rows):
    lines = [
        "# Benchmark results",
        "",
        "Generated from `bench/results.csv` by `tools/plot_bench.py`.",
        "",
        "| section | method | N | threads | θ | ms/step | tree build ms | traverse ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        theta = r["theta"] if float(r["theta"]) > 0 else ""
        bh = f'{float(r["build_ms"]):.2f}' if float(r["build_ms"]) > 0 else ""
        tr = f'{float(r["traverse_ms"]):.2f}' if float(r["traverse_ms"]) > 0 else ""
        lines.append(
            f'| {r["section"]} | {r["method"]} | {int(r["n"]):,} | {r["threads"]} '
            f'| {theta} | {float(r["ms_step"]):.2f} | {bh} | {tr} |')
    with open("bench/results.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(ASSETS, exist_ok=True)
    rows = read_rows()
    sweep_chart(rows)
    scaling_chart(rows)
    markdown_table(rows)
    print("wrote assets/bench_sweep.svg assets/bench_scaling.svg bench/results.md")


if __name__ == "__main__":
    main()
