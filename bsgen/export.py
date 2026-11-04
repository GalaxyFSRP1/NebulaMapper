import json
import re
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np


DIFF_RANK = {"Easy": 1, "Normal": 3, "Hard": 5, "Expert": 7, "ExpertPlus": 9}
NJS_STARS = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
NJS_MEDIAN = [10.5, 10, 12, 14, 15, 16, 16, 17, 18, 19, 20, 20.5, 20.1, 20.2, 20]

COLOR_SCHEMES = {
    "Default (Game Colors)": None,
    "Nebula (Purple & Cyan)": {
        "_colorLeft": {"r": 0.68, "g": 0.24, "b": 0.98},
        "_colorRight": {"r": 0.08, "g": 0.85, "b": 0.98},
        "_envColorLeft": {"r": 0.58, "g": 0.18, "b": 0.92},
        "_envColorRight": {"r": 0.05, "g": 0.75, "b": 0.95},
        "_obstacleColor": {"r": 0.35, "g": 0.10, "b": 0.65},
    },
    "Phantom Void (Crimson & Violet)": {
        "_colorLeft": {"r": 0.95, "g": 0.12, "b": 0.32},
        "_colorRight": {"r": 0.52, "g": 0.22, "b": 0.96},
        "_envColorLeft": {"r": 0.90, "g": 0.08, "b": 0.28},
        "_envColorRight": {"r": 0.45, "g": 0.18, "b": 0.92},
        "_obstacleColor": {"r": 0.22, "g": 0.05, "b": 0.38},
    },
    "Cyberpunk (Neon Pink & Electric Gold)": {
        "_colorLeft": {"r": 1.0, "g": 0.16, "b": 0.58},
        "_colorRight": {"r": 0.98, "g": 0.85, "b": 0.12},
        "_envColorLeft": {"r": 0.95, "g": 0.12, "b": 0.52},
        "_envColorRight": {"r": 0.12, "g": 0.92, "b": 0.88},
        "_obstacleColor": {"r": 0.85, "g": 0.10, "b": 0.45},
    },
    "Solar Ice (Amber & Glacier Blue)": {
        "_colorLeft": {"r": 0.98, "g": 0.52, "b": 0.10},
        "_colorRight": {"r": 0.18, "g": 0.68, "b": 0.98},
        "_envColorLeft": {"r": 0.95, "g": 0.45, "b": 0.08},
        "_envColorRight": {"r": 0.15, "g": 0.62, "b": 0.95},
        "_obstacleColor": {"r": 0.12, "g": 0.35, "b": 0.65},
    },
    "Emerald Matrix (Toxic Green & Cyber Teal)": {
        "_colorLeft": {"r": 0.13, "g": 0.88, "b": 0.38},
        "_colorRight": {"r": 0.08, "g": 0.78, "b": 0.72},
        "_envColorLeft": {"r": 0.10, "g": 0.82, "b": 0.34},
        "_envColorRight": {"r": 0.06, "g": 0.72, "b": 0.68},
        "_obstacleColor": {"r": 0.04, "g": 0.38, "b": 0.24},
    },
    "Blood Moon (Scarlet & Obsidian Silver)": {
        "_colorLeft": {"r": 0.92, "g": 0.10, "b": 0.14},
        "_colorRight": {"r": 0.75, "g": 0.82, "b": 0.92},
        "_envColorLeft": {"r": 0.86, "g": 0.06, "b": 0.10},
        "_envColorRight": {"r": 0.62, "g": 0.70, "b": 0.85},
        "_obstacleColor": {"r": 0.42, "g": 0.04, "b": 0.08},
    },
    "Supernova (Hot Magenta & Hyper Blue)": {
        "_colorLeft": {"r": 0.92, "g": 0.22, "b": 0.96},
        "_colorRight": {"r": 0.22, "g": 0.52, "b": 0.98},
        "_envColorLeft": {"r": 0.86, "g": 0.18, "b": 0.92},
        "_envColorRight": {"r": 0.18, "g": 0.46, "b": 0.95},
        "_obstacleColor": {"r": 0.38, "g": 0.10, "b": 0.55},
    },
    "Sakura Drift (Cherry Blossom & Mint)": {
        "_colorLeft": {"r": 0.98, "g": 0.44, "b": 0.58},
        "_colorRight": {"r": 0.20, "g": 0.88, "b": 0.64},
        "_envColorLeft": {"r": 0.94, "g": 0.38, "b": 0.52},
        "_envColorRight": {"r": 0.16, "g": 0.82, "b": 0.58},
        "_obstacleColor": {"r": 0.52, "g": 0.18, "b": 0.32},
    },
    "Custom Hex Colors (Pick Below)": "custom",
}


def hex_to_rgb01(hex_str: str, fallback=(0.68, 0.24, 0.98)) -> dict:
    s = (hex_str or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    try:
        r, g, b = int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0
    except Exception:
        r, g, b = fallback
    return {"r": round(r, 3), "g": round(g, 3), "b": round(b, 3)}


def rgb01_to_hex(d: dict) -> str:
    r = int(np.clip(d.get("r", 0.5) * 255, 0, 255))
    g = int(np.clip(d.get("g", 0.5) * 255, 0, 255))
    b = int(np.clip(d.get("b", 0.5) * 255, 0, 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def make_custom_palette(left_hex="#a855f7", right_hex="#06b6d4", wall_hex="#4c1d95") -> dict:
    c_l = hex_to_rgb01(left_hex, (0.68, 0.24, 0.98))
    c_r = hex_to_rgb01(right_hex, (0.08, 0.85, 0.98))
    c_w = hex_to_rgb01(wall_hex, (0.35, 0.10, 0.65))
    return {
        "_colorLeft": c_l,
        "_colorRight": c_r,
        "_envColorLeft": c_l,
        "_envColorRight": c_r,
        "_obstacleColor": c_w,
    }


def resolve_palette(color_scheme: str | None = None, custom_colors=None) -> dict | None:
    if custom_colors:
        if isinstance(custom_colors, dict):
            return custom_colors
        if isinstance(custom_colors, (list, tuple)) and len(custom_colors) >= 2:
            w = custom_colors[2] if len(custom_colors) >= 3 else "#4c1d95"
            return make_custom_palette(custom_colors[0], custom_colors[1], w)
        if isinstance(custom_colors, str) and "," in custom_colors:
            parts = [p.strip() for p in custom_colors.split(",") if p.strip()]
            if len(parts) >= 2:
                return make_custom_palette(parts[0], parts[1], parts[2] if len(parts) >= 3 else "#4c1d95")
    val = COLOR_SCHEMES.get(color_scheme or "")
    if val == "custom":
        return make_custom_palette()
    return val


def render_palette_preview_html(color_scheme: str, custom_left="#a855f7", custom_right="#06b6d4", custom_wall="#4c1d95") -> str:
    if color_scheme == "Custom Hex Colors (Pick Below)":
        pal = make_custom_palette(custom_left, custom_right, custom_wall)
    else:
        pal = COLOR_SCHEMES.get(color_scheme)
    if not pal or not isinstance(pal, dict):
        l_hex, r_hex, w_hex = "#f03030", "#2064a8", "#f03030"
    else:
        l_hex = rgb01_to_hex(pal["_colorLeft"])
        r_hex = rgb01_to_hex(pal["_colorRight"])
        w_hex = rgb01_to_hex(pal.get("_obstacleColor", pal["_colorLeft"]))
    return (
        f'<div style="display:flex;align-items:center;gap:12px;padding:7px 12px;border-radius:10px;'
        f'background:rgba(10,13,26,0.75);border:1px solid rgba(148,163,184,0.16);font-size:12px;">'
        f'  <span style="display:inline-flex;align-items:center;gap:6px;">'
        f'    <span style="width:14px;height:14px;border-radius:4px;background:{l_hex};box-shadow:0 0 8px {l_hex};"></span>'
        f'    <b>Left Saber:</b> <code>{l_hex}</code>'
        f'  </span>'
        f'  <span style="display:inline-flex;align-items:center;gap:6px;">'
        f'    <span style="width:14px;height:14px;border-radius:4px;background:{r_hex};box-shadow:0 0 8px {r_hex};"></span>'
        f'    <b>Right Saber:</b> <code>{r_hex}</code>'
        f'  </span>'
        f'  <span style="display:inline-flex;align-items:center;gap:6px;">'
        f'    <span style="width:14px;height:14px;border-radius:4px;background:{w_hex};box-shadow:0 0 8px {w_hex};"></span>'
        f'    <b>Walls:</b> <code>{w_hex}</code>'
        f'  </span>'
        f'</div>'
    )


def typical_njs(stars):
    return round(float(np.interp(stars, NJS_STARS, NJS_MEDIAN)) * 2) / 2


REACTION_STARS = [2, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 13]
REACTION_MS = [1009, 879, 771, 714, 643, 600, 558, 540, 500, 475, 464]


def spawn_offset(stars, njs, bpm):
    beat = 60 / bpm
    half = 4.0
    while njs * beat * half > 17.999 and half > 0.25:
        half /= 2
    want = float(np.interp(stars, REACTION_STARS, REACTION_MS)) / 1000 / beat
    return round(max(want, 0.25) - half, 3)


def difficulty_for(stars):
    for name, top in (("Easy", 4.1), ("Normal", 5.0), ("Hard", 6.5), ("Expert", 9.0)):
        if stars < top:
            return name
    return "ExpertPlus"


def difficulty_names(stars):
    names = list(DIFF_RANK)
    order = sorted(range(len(stars)), key=lambda i: stars[i])
    want = [names.index(difficulty_for(stars[i])) for i in order]
    best = min(combinations(range(len(names)), len(stars)), key=lambda c: sum(abs(x - w) for x, w in zip(c, want)))
    out = [None] * len(stars)
    for i, k in zip(order, best):
        out[i] = names[k]
    return out


def safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", s).strip()


def v2_walls(walls):
    out = []
    for w in walls:
        full = w["layer"] == 0 and w["height"] >= 3
        line, width = int(w["line"]), int(w["width"])
        out.append({"_time": round(float(w["beat"]), 5), "_lineIndex": line, "_type": 0 if full else 1,
                    "_duration": round(float(w["duration"]), 5), "_width": width})
    return out


def v3_difficulty(notes, bombs, walls, arcs, tempo_changes, lights=()):
    r = lambda x: round(float(x), 5)
    return {
        "version": "3.3.0",
        "bpmEvents": [{"b": r(b), "m": float(m)} for b, m in tempo_changes],
        "rotationEvents": [],
        "colorNotes": sorted(({"b": r(n["beat"]), "x": int(n["line"]), "y": int(n["layer"]), "c": int(n["color"]),
                               "d": int(n["cut_dir"]), "a": 0} for n in notes), key=lambda o: o["b"]),
        "bombNotes": sorted(({"b": r(b["beat"]), "x": int(b["line"]), "y": int(b["layer"])} for b in bombs),
                            key=lambda o: o["b"]),
        "obstacles": sorted(({"b": r(w["beat"]), "x": int(w["line"]), "y": int(w["layer"]), "d": r(w["duration"]),
                              "w": int(w["width"]), "h": int(w["height"])} for w in walls), key=lambda o: o["b"]),
        "sliders": sorted(({"c": int(a["color"]), "b": r(a["beat"]), "x": int(a["line"]), "y": int(a["layer"]),
                            "d": int(a["dir"]), "mu": float(a["mult"]), "tb": r(a["tail_beat"]), "tx": int(a["tail_line"]),
                            "ty": int(a["tail_layer"]), "tc": int(a["tail_dir"]), "tmu": float(a["tail_mult"]),
                            "m": int(a["mid"])} for a in arcs), key=lambda o: o["b"]),
        "burstSliders": [], "waypoints": [],
        "basicBeatmapEvents": [{"b": r(b), "et": t, "i": v, "f": 1.0} for b, t, v in lights], "colorBoostBeatmapEvents": [],
        "lightColorEventBoxGroups": [], "lightRotationEventBoxGroups": [], "lightTranslationEventBoxGroups": [],
        "vfxEventBoxGroups": [], "_fxEventsCollection": {"_fl": [], "_il": []},
        "basicEventTypesWithKeywords": {"d": []}, "useNormalEventsAsCompatibleEvents": True,
    }


def write_level(folder: Path, audio: bytes, notes, bpm: float, title: str, label: str,
                njs: float = 18.0, difficulty="ExpertPlus", tempo_changes=(), bombs=None, walls=None, arcs=None,
                description="Generated by NebulaMapper neural model.", fmt="v2", lights=()):
    level = dict(difficulty=difficulty, notes=notes, label=label, njs=njs, bombs=bombs, walls=walls, arcs=arcs,
                 lights=lights)
    return write_pack(folder, audio, bpm, title, [level], tempo_changes, description, fmt)


def write_pack(folder: Path, audio: bytes, bpm: float, title: str, levels, tempo_changes=(),
               description="Generated by NebulaMapper neural model.", fmt="v2", cover=None, artist="",
               environment="DefaultEnvironment", color_scheme: str | None = None,
               level_author: str | None = None, custom_colors=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "song.egg").write_bytes(audio)
    levels = sorted(levels, key=lambda lv: DIFF_RANK[lv["difficulty"]])
    names = [lv["difficulty"] for lv in levels]
    assert len(set(names)) == len(names), f"each difficulty once: {names}"
    palette = resolve_palette(color_scheme, custom_colors)
    author = level_author or "NebulaMapper (PhantomVoidTTV)"

    def make_custom_data(lv):
        cd = {"_difficultyLabel": lv["label"], "_information": [description]}
        if palette:
            cd.update(palette)
        return cd

    info = {
        "_version": "2.1.0", "_songName": title, "_songSubName": "", "_songAuthorName": artist,
        "_levelAuthorName": author, "_beatsPerMinute": bpm, "_shuffle": 0, "_shufflePeriod": 0.5,
        "_previewStartTime": 30, "_previewDuration": 10, "_songFilename": "song.egg",
        "_coverImageFilename": "cover.jpg" if cover else "", "_environmentName": environment, "_songTimeOffset": 0,
        "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": [{
                "_difficulty": lv["difficulty"], "_difficultyRank": DIFF_RANK[lv["difficulty"]],
                "_beatmapFilename": f"{lv['difficulty']}Standard.dat",
                "_noteJumpMovementSpeed": lv["njs"],
                "_noteJumpStartBeatOffset": spawn_offset(lv["stars"], lv["njs"], bpm) if lv.get("stars") else 0,
                "_customData": make_custom_data(lv),
            } for lv in levels],
        }],
    }
    (folder / "Info.dat").write_text(json.dumps(info, indent=1), encoding="utf-8")
    files = ["Info.dat"]
    if cover:
        (folder / "cover.jpg").write_bytes(cover)
        files.append("cover.jpg")
    for lv in levels:
        fname = f"{lv['difficulty']}Standard.dat"
        (folder / fname).write_text(json.dumps(_difficulty(lv["notes"], lv.get("bombs"), lv.get("walls"),
                                                           lv.get("arcs"), tempo_changes, lv.get("lights", ()), fmt)), encoding="utf-8")
        files.append(fname)
    with zipfile.ZipFile(folder.parent / (folder.name + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for f in files + ["song.egg"]:
            z.write(folder / f, f)
    return folder


def _difficulty(notes, bombs, walls, arcs, tempo_changes, lights, fmt):
    bombs = bombs if bombs is not None else []
    walls = walls if walls is not None else []
    arcs = arcs if arcs is not None else []
    if fmt == "v3":
        diff = v3_difficulty(notes, bombs, walls, arcs, tempo_changes, lights)
    else:
        objs = [{"_time": round(float(n["beat"]), 5), "_lineIndex": int(n["line"]), "_lineLayer": int(n["layer"]),
                 "_type": int(n["color"]), "_cutDirection": int(n["cut_dir"])} for n in notes]
        objs += [{"_time": round(float(b["beat"]), 5), "_lineIndex": int(b["line"]), "_lineLayer": int(b["layer"]),
                  "_type": 3, "_cutDirection": 0} for b in bombs]
        objs.sort(key=lambda o: o["_time"])
        events = [{"_time": round(float(b), 5), "_type": 100, "_value": 0, "_floatValue": float(m)}
                  for b, m in tempo_changes]
        events += [{"_time": round(float(b), 5), "_type": t, "_value": v, "_floatValue": 1.0} for b, t, v in lights]
        events.sort(key=lambda e: e["_time"])
        diff = {"_version": "2.2.0", "_events": events, "_notes": objs, "_obstacles": v2_walls(walls)}
        if len(arcs):
            diff["_sliders"] = [{"_colorType": int(a["color"]), "_headTime": round(float(a["beat"]), 5),
                                 "_headLineIndex": int(a["line"]), "_headLineLayer": int(a["layer"]),
                                 "_headControlPointLengthMultiplier": float(a["mult"]),
                                 "_headCutDirection": int(a["dir"]), "_tailTime": round(float(a["tail_beat"]), 5),
                                 "_tailLineIndex": int(a["tail_line"]), "_tailLineLayer": int(a["tail_layer"]),
                                 "_tailControlPointLengthMultiplier": float(a["tail_mult"]),
                                 "_tailCutDirection": int(a["tail_dir"]),
                                 "_sliderMidAnchorMode": int(a["mid"])} for a in arcs]
    return diff
