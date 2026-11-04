import html
from pathlib import Path

import numpy as np

from . import beatmap, labeler, rating

DIR_LABELS = ["↑ Up", "↓ Down", "← Left", "→ Right", "↖ Up-L", "↗ Up-R", "↙ Dn-L", "↘ Dn-R", "• Dot"]
DIR_ARROWS = ["↑", "↓", "←", "→", "↖", "↗", "↙", "↘", "•"]
DIFF_COLORS = {
    "Easy": "#22c55e",
    "Normal": "#06b6d4",
    "Hard": "#f59e0b",
    "Expert": "#ec4899",
    "ExpertPlus": "#a855f7",
}


def analyze_map_zip(zip_path: str | Path) -> dict:
    with beatmap.open_map(zip_path) as z:
        info = beatmap.read_info(z)
        levels = []
        for dm in info.difficulties:
            if dm["characteristic"] != "Standard":
                continue
            d = beatmap.read_difficulty(z, info, "Standard", dm["difficulty"])
            notes = d.notes
            lab = labeler.parity_rule(notes)
            feats = rating.features(notes, d.njs or 16.0, lab)
            est = rating.estimate(notes, d.njs or 16.0, notes["time"])
            red_n = int((notes["color"] == 0).sum()) if len(notes) else 0
            blue_n = int((notes["color"] == 1).sum()) if len(notes) else 0

            grid = np.zeros((3, 4), dtype=int)
            if len(notes):
                x = np.clip(notes["line"], 0, 3)
                y = np.clip(notes["layer"], 0, 2)
                for xi, yi in zip(x, y):
                    grid[int(yi), int(xi)] += 1

            dirs = np.zeros(9, dtype=int)
            if len(notes):
                cdir = np.clip(notes["cut_dir"], 0, 8)
                for cd in cdir:
                    dirs[int(cd)] += 1

            max_t = float(notes["time"].max()) if len(notes) else 60.0
            bins = np.arange(0.0, max(max_t, 8.0) + 4.0, 4.0)
            counts, _ = np.histogram(notes["time"], bins=bins) if len(notes) else (np.zeros(len(bins) - 1), bins)
            nps_curve = (counts / 4.0).tolist()
            bin_times = bins[:-1].tolist()

            # Extract up to 24 notes around the peak density window for the Note Highway Slice visualizer
            slice_notes = []
            if len(notes):
                peak_idx = int(np.argmax(counts)) if len(counts) else 0
                t_start = float(bins[peak_idx])
                mask = (notes["time"] >= t_start) & (notes["time"] <= t_start + 8.0)
                sub = notes[mask][:24] if mask.any() else notes[:24]
                for n in sub:
                    slice_notes.append({
                        "beat": round(float(n["beat"]), 3),
                        "line": int(np.clip(n["line"], 0, 3)),
                        "layer": int(np.clip(n["layer"], 0, 2)),
                        "color": int(n["color"]),
                        "cut_dir": int(np.clip(n["cut_dir"], 0, 8)),
                    })

            levels.append({
                "difficulty": dm["difficulty"],
                "njs": d.njs or 16.0,
                "notes": len(notes),
                "red": red_n,
                "blue": blue_n,
                "bombs": len(d.bombs),
                "walls": len(d.walls),
                "arcs": len(d.arcs),
                "nps": float(feats["notes/s"]),
                "peak_2s": float(feats["peak nps 2s max"]),
                "peak_8s": float(feats["peak nps 8s max"]),
                "swings_s": float(feats["swings/s"]),
                "resets_pct": float(feats["resets/swing"]) * 100.0,
                "dots_pct": float(feats["dots"]) * 100.0,
                "stars": float(est[0]) if est else None,
                "tech": float(est[1]) if est else None,
                "pass": float(est[2]) if est else None,
                "grid": grid.tolist(),
                "dirs": dirs.tolist(),
                "nps_curve": nps_curve,
                "bin_times": bin_times,
                "slice_notes": slice_notes,
            })

    return {
        "song_name": info.song_name,
        "song_author": info.song_author,
        "mapper": info.mapper,
        "bpm": info.bpm,
        "levels": levels,
    }


def _render_highway_svg(slice_notes: list[dict], diff_name: str) -> str:
    if not slice_notes:
        return ""
    W, H = 560, 124
    pad_x, pad_y = 36, 18
    usable_w = W - pad_x * 2
    b_min = min(n["beat"] for n in slice_notes)
    b_max = max(max(n["beat"] for n in slice_notes), b_min + 2.0)
    span = max(b_max - b_min, 1.0)

    elems = [
        f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;background:rgba(10,13,28,0.9);border:1px solid rgba(139,92,246,0.28);border-radius:10px;">'
    ]
    # Draw 4 horizontal lane tracks
    for lane in range(4):
        ly = pad_y + lane * 24 + 12
        elems.append(f'<line x1="{pad_x}" y1="{ly}" x2="{W - pad_x}" y2="{ly}" stroke="rgba(148,163,184,0.16)" stroke-width="1" />')
        elems.append(f'<text x="8" y="{ly + 4}" fill="#94a3b8" font-size="10" font-family="monospace">L{lane + 1}</text>')

    # Draw beat markers
    for b in range(int(np.floor(b_min)), int(np.ceil(b_max)) + 1):
        if b < b_min:
            continue
        bx = pad_x + ((b - b_min) / span) * usable_w
        elems.append(f'<line x1="{bx:.1f}" y1="{pad_y}" x2="{bx:.1f}" y2="{H - 16}" stroke="rgba(139,92,246,0.18)" stroke-dasharray="2,2" />')
        elems.append(f'<text x="{bx:.1f}" y="{H - 4}" text-anchor="middle" fill="#64748b" font-size="9" font-family="monospace">b{b}</text>')

    # Draw note blocks
    for n in slice_notes:
        nx = pad_x + ((n["beat"] - b_min) / span) * usable_w
        ny = pad_y + n["line"] * 24 + 12 - (n["layer"] - 1) * 4
        fill = "#c084fc" if n["color"] == 0 else "#22d3ee"
        stroke = "#f472b6" if n["color"] == 0 else "#67e8f9"
        arrow = DIR_ARROWS[n["cut_dir"]]
        elems.append(
            f'<rect x="{nx - 9:.1f}" y="{ny - 9:.1f}" width="18" height="18" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="1.2" />'
            f'<text x="{nx:.1f}" y="{ny + 4:.1f}" text-anchor="middle" fill="#080a14" font-size="11" font-weight="800">{arrow}</text>'
        )

    elems.append("</svg>")
    return (
        f'<div style="margin:10px 0;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;font-size:12.5px;font-weight:600;margin-bottom:4px;">'
        f'<span>⚡ Peak Pattern Highway Slice ({html.escape(diff_name)} · 4 Lanes × Beat Timeline)</span>'
        f'<span style="opacity:0.7;font-size:11.5px;">🟣 Left Saber · 🔵 Right Saber · Layer offset shown vertically</span>'
        f'</div>'
        + "".join(elems)
        + "</div>"
    )


def render_analysis_html(report: dict, selected_diff: str | None = None) -> str:
    levels = report.get("levels", [])
    if not levels:
        return '<div class="sum">No Standard difficulties found in this map.</div>'

    title = html.escape(report.get("song_name") or "Untitled Song")
    artist = html.escape(report.get("song_author") or "Unknown Artist")
    mapper = html.escape(report.get("mapper") or "NebulaMapper")
    bpm = report.get("bpm", 120.0)

    parts = [
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;'
        f'padding:14px 16px;border-radius:12px;background:rgba(139,92,246,0.12);border:1px solid rgba(139,92,246,0.3);margin-bottom:12px;">'
        f'<div>'
        f'<div style="font-size:18px;font-weight:800;color:#f8fafc;">{title}</div>'
        f'<div style="font-size:13px;color:#cbd5e1;">{artist} · Mapped by <b>{mapper}</b> · <code>{bpm:g} BPM</code></div>'
        f'</div>'
        f'<a href="https://allpoland.github.io/ArcViewer/" target="_blank" rel="noopener" '
        f'style="padding:6px 13px;border-radius:999px;background:linear-gradient(90deg,#8b5cf6,#06b6d4);'
        f'color:#fff !important;font-size:12px;font-weight:700;text-decoration:none !important;">'
        f'🌐 Open 3D Web Preview (ArcViewer)</a>'
        f'</div>'
    ]

    rows_html = []
    for lv in levels:
        dname = lv["difficulty"]
        col = DIFF_COLORS.get(dname, "#06b6d4")
        star_str = f"{lv['stars']:.2f}★" if lv["stars"] is not None else "—"
        tech_str = f"{lv['tech']:.1f}" if lv["tech"] is not None else "—"
        pass_str = f"{lv['pass']:.1f}" if lv["pass"] is not None else "—"
        lr_bal = f"{lv['red'] / max(lv['notes'], 1) * 100:.0f}/{lv['blue'] / max(lv['notes'], 1) * 100:.0f}%"
        rows_html.append(
            f'<tr style="border-bottom:1px solid rgba(148,163,184,0.14);">'
            f'<td style="padding:7px 8px;font-weight:700;color:{col};">{html.escape(dname)}</td>'
            f'<td style="padding:7px 8px;font-weight:700;">{star_str}</td>'
            f'<td style="padding:7px 8px;">{lv["notes"]:,}</td>'
            f'<td style="padding:7px 8px;">{lv["nps"]:.2f}/s (peak {lv["peak_2s"]:.1f})</td>'
            f'<td style="padding:7px 8px;">{lv["njs"]:g}</td>'
            f'<td style="padding:7px 8px;">{tech_str} / {pass_str}</td>'
            f'<td style="padding:7px 8px;">{lv["bombs"]}/{lv["walls"]}/{lv["arcs"]}</td>'
            f'<td style="padding:7px 8px;">{lr_bal} ({lv["resets_pct"]:.1f}% resets)</td>'
            f'</tr>'
        )

    parts.append(
        '<div style="overflow-x:auto;margin-bottom:14px;">'
        '<table style="width:100%;border-collapse:collapse;font-size:13px;text-align:left;">'
        '<thead><tr style="border-bottom:1px solid rgba(148,163,184,0.3);color:#94a3b8;">'
        '<th style="padding:6px 8px;">Difficulty</th>'
        '<th style="padding:6px 8px;">Est ★</th>'
        '<th style="padding:6px 8px;">Notes</th>'
        '<th style="padding:6px 8px;">NPS</th>'
        '<th style="padding:6px 8px;">NJS</th>'
        '<th style="padding:6px 8px;">Tech/Pass</th>'
        '<th style="padding:6px 8px;">B/W/A</th>'
        '<th style="padding:6px 8px;">L/R & Parity</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody></table></div>'
    )

    # NPS Timeline SVG
    W, H = 560, 136
    pad_l, pad_r, pad_t, pad_b = 36, 14, 16, 22
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b
    max_nps = max([max(lv["nps_curve"] or [1.0]) for lv in levels] + [4.0])
    svg_lines = []
    for lv in levels:
        curve = lv["nps_curve"]
        if not curve:
            continue
        col = DIFF_COLORS.get(lv["difficulty"], "#06b6d4")
        pts = []
        for idx, val in enumerate(curve):
            x = pad_l + (idx / max(len(curve) - 1, 1)) * pw
            y = pad_t + ph - (val / max_nps) * ph
            pts.append(f"{x:.1f},{y:.1f}")
        svg_lines.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.4" points="{" ".join(pts)}" />')

    parts.append(
        f'<div style="margin-bottom:12px;">'
        f'<div style="font-size:12.5px;font-weight:600;margin-bottom:4px;">📈 Note Density Timeline (NPS over Song Duration)</div>'
        f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;background:rgba(10,13,28,0.9);border:1px solid rgba(148,163,184,0.16);border-radius:10px;">'
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + ph}" stroke="rgba(148,163,184,0.3)" />'
        f'<line x1="{pad_l}" y1="{pad_t + ph}" x2="{pad_l + pw}" y2="{pad_t + ph}" stroke="rgba(148,163,184,0.3)" />'
        f'<text x="4" y="{pad_t + 10}" fill="#94a3b8" font-size="10">{max_nps:.1f}/s</text>'
        f'<text x="4" y="{pad_t + ph}" fill="#94a3b8" font-size="10">0/s</text>'
        f'{"".join(svg_lines)}'
        f'</svg></div>'
    )

    top = next((lv for lv in levels if lv["difficulty"] == selected_diff), levels[-1])
    parts.append(_render_highway_svg(top.get("slice_notes", []), top["difficulty"]))

    grid = np.array(top["grid"], dtype=float)
    gmax = max(float(grid.max()), 1.0)
    cells_html = []
    for r in (2, 1, 0):
        for c in range(4):
            count = int(grid[r, c])
            pct = count / max(top["notes"], 1) * 100.0
            alpha = 0.14 + 0.68 * (count / gmax)
            cells_html.append(
                f'<div style="background:rgba(139,92,246,{alpha:.2f});border:1px solid rgba(192,132,252,0.28);'
                f'border-radius:8px;padding:8px 4px;text-align:center;">'
                f'<div style="font-weight:700;font-size:13px;color:#f8fafc;">{pct:.1f}%</div>'
                f'<div style="font-size:11px;color:#cbd5e1;">{count} notes</div>'
                f'</div>'
            )

    dirs = top["dirs"]
    dmax = max(max(dirs), 1)
    dir_bars = []
    for label, cnt in zip(DIR_LABELS, dirs):
        pct = cnt / max(top["notes"], 1) * 100.0
        w_pct = cnt / dmax * 100.0
        dir_bars.append(
            f'<div style="display:grid;grid-template-columns:58px 1fr 42px;gap:6px;align-items:center;font-size:12px;margin:3px 0;">'
            f'<span>{html.escape(label)}</span>'
            f'<div style="height:8px;background:rgba(148,163,184,0.16);border-radius:999px;overflow:hidden;">'
            f'<div style="height:100%;width:{w_pct:.0f}%;background:linear-gradient(90deg,#8b5cf6,#ec4899,#06b6d4);"></div></div>'
            f'<span style="text-align:right;opacity:0.8;font-family:monospace;">{pct:.0f}%</span>'
            f'</div>'
        )

    parts.append(
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start;margin-top:8px;">'
        f'<div><div style="font-size:12.5px;font-weight:600;margin-bottom:6px;">🎯 4×3 Grid Heatmap ({html.escape(top["difficulty"])})</div>'
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;">{"".join(cells_html)}</div></div>'
        f'<div><div style="font-size:12.5px;font-weight:600;margin-bottom:6px;">🧭 9-Direction Cut Distribution ({html.escape(top["difficulty"])})</div>'
        f'{"".join(dir_bars)}</div>'
        f'</div>'
    )

    return "".join(parts)
