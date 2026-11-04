import argparse
import contextlib
import functools
import html
import io
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate
from bsgen import analyzer, artwork, audio_fx, beatmap, export, hf_sync, hub, lightmodel, modelgen, rhythm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "web"
STYLES = {"Auto": None, "Acc": "acc", "Tech": "tech", "Midspeed": "midspeed", "Speed": "speed"}
ARCS = {"Off": None, "Rare": 0.0, "Normal": 0.025, "Lots": 0.08}
DRY, XTC = 1.0, 0.3
RHYTHM = [f"Auto ({rhythm.AUTO_STARS:g}★ and up)" if rhythm.AUTO_STARS > 0 else "Auto (every difficulty)", "Always", "Never"]
DIFFS = {"Easy": (3.5, "Easy"), "Normal": (4.5, "Normal"), "Hard": (5.5, "Hard"), "Expert": (7.5, "Expert"),
         "Expert+": (10.0, "ExpertPlus")}
ALL_DIFF_CHOICES = [("Easy", "Easy"), ("Normal", "Normal"), ("Hard", "Hard"), ("Expert", "Expert"), ("Expert+", "ExpertPlus")]
MAX_LEVELS = 5
SCENES = {"Auto": "auto"} | dict(zip(["The First (default)", "Origins", "Triangle", "Nice", "Big Mirror", "Dragons", "K/DA", "Monstercat",
                   "Crab Rave", "Panic", "Rocket League", "Green Day", "Green Day Grenade", "Timbaland", "FitBeat",
                   "Linkin Park", "BTS", "Kaleidoscope", "Interscope", "Skrillex", "Billie Eilish", "Spooky",
                   "Lady Gaga"], lightmodel.ENVS))
START_STARS = (10, 7.5, 5.5, 12.5, 4)
SHOWN = {"ExpertPlus": "Expert+"}
HIGH_STAR = 12.0
LEAD_CHECK, LEAD_IN, LEAD_DB = 1.0, 1.0, -45.0

VIOLET, MAGENTA, CYAN = "#8b5cf6", "#ec4899", "#06b6d4"

QUICK_PRESETS = {
    "Custom / Manual": None,
    "Nebula Signature (PhantomVoid)": {"style": "Auto", "arcs": "Normal", "mapper": "phantomvoid", "color": "Nebula (Purple & Cyan)", "scene": "Kaleidoscope"},
    "Ranked Standard": {"style": "Auto", "arcs": "Off", "mapper": "nebulamapper", "color": "Default (Game Colors)", "scene": "Auto"},
    "Tech & Angles": {"style": "Tech", "arcs": "Normal", "mapper": "nebula_tech", "color": "Phantom Void (Crimson & Violet)", "scene": "Triangle"},
    "Speed / Stream": {"style": "Speed", "arcs": "Off", "mapper": "nebula_speed", "color": "Cyberpunk (Neon Pink & Electric Gold)", "scene": "Skrillex"},
    "Acc / Precision": {"style": "Acc", "arcs": "Rare", "mapper": "nebula_acc", "color": "Solar Ice (Amber & Glacier Blue)", "scene": "Nice"},
    "Emerald Overdrive (Speed + Arcs)": {"style": "Speed", "arcs": "Lots", "mapper": "nebula_speed", "color": "Emerald Matrix (Toxic Green & Cyber Teal)", "scene": "Monstercat"},
    "Supernova Showpiece (Lights + Arcs)": {"style": "Midspeed", "arcs": "Lots", "mapper": "phantomvoid", "color": "Supernova (Hot Magenta & Hyper Blue)", "scene": "Interscope"},
}

HF_URL = f"https://huggingface.co/{hub.DEFAULT_REPO}/tree/main"
STAR_ASK = (
    '<div class="star"><span>⚡ Neural weights synced with Hugging Face Hub</span>'
    f'<a href="{HF_URL}" target="_blank" rel="noopener">🤗 {html.escape(hub.DEFAULT_REPO)}</a></div>'
)

MAP_LINE = re.compile(r"(?:(\w+): )?([\d.]+)★, NJS ([\d.]+): (\d+) notes, ([\d.]+) notes/s")
RATED_LINE = re.compile(r"(?:(\w+): )?(?:estimated from the notes: |kept the one made at [\d.]+★, rated )([\d.]+)★")
REPLAY_LINE = re.compile(r"(\d+) of (\d+) notes cut, (\d+) bad cuts?, (\d+) missed, score (\d+)")

CSS = f"""
.gradio-container {{
    width: 100% !important;
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 16px 22px 40px !important;
    font-family: 'Space Grotesk', system-ui, -apple-system, sans-serif !important;
}}

body, .dark {{
    background: radial-gradient(circle at 12% 0%, rgba(139, 92, 246, 0.16), transparent 44%),
                radial-gradient(circle at 88% 8%, rgba(6, 182, 212, 0.14), transparent 42%),
                radial-gradient(circle at 50% 95%, rgba(236, 72, 153, 0.08), transparent 45%),
                #070913 !important;
}}

#nebula-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
    padding: 18px 24px;
    margin-bottom: 16px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(18, 22, 46, 0.94), rgba(10, 13, 28, 0.96));
    border: 1px solid rgba(139, 92, 246, 0.34);
    box-shadow: 0 14px 36px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}}

#nebula-brand {{
    display: flex;
    align-items: center;
    gap: 16px;
}}

#nebula-emblem {{
    width: 54px;
    height: 54px;
    flex-shrink: 0;
    filter: drop-shadow(0 0 14px rgba(139, 92, 246, 0.5));
}}

#nebula-brand h1 {{
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 0.6px;
    margin: 0;
    line-height: 1.1;
    color: #f8fafc;
}}

#nebula-brand h1 .grad {{
    background: linear-gradient(90deg, #c084fc, #ec4899, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

#nebula-brand p {{
    margin: 4px 0 0;
    font-size: 13px;
    color: #94a3b8;
    letter-spacing: 0.3px;
}}

#nebula-badges {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}}

.nebula-pill {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 13px;
    border-radius: 999px;
    font-size: 12.5px;
    font-weight: 600;
    background: rgba(139, 92, 246, 0.14);
    border: 1px solid rgba(139, 92, 246, 0.38);
    color: #e2e8f0 !important;
    text-decoration: none !important;
    transition: all 0.18s ease;
}}

.nebula-pill:hover {{
    border-color: {CYAN};
    background: rgba(6, 182, 212, 0.18);
    transform: translateY(-1px);
}}

.nebula-pill .dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 8px #22c55e;
}}

.studio-card {{
    background: rgba(14, 18, 36, 0.82) !important;
    border: 1px solid rgba(139, 92, 246, 0.20) !important;
    border-radius: 14px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.32) !important;
}}

.deck-heading {{
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: #c084fc;
    margin: 0 0 8px 2px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
}}

#go, #play, #gen-models-btn {{
    background: linear-gradient(100deg, {VIOLET} 0%, {MAGENTA} 52%, {CYAN} 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    letter-spacing: 0.7px !important;
    text-transform: uppercase !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 26px rgba(139, 92, 246, 0.42) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}}

#go:hover, #play:hover, #gen-models-btn:hover {{
    transform: translateY(-1px);
    box-shadow: 0 12px 30px rgba(236, 72, 153, 0.52) !important;
}}

footer {{ display: none !important; }}
#map-zip {{ border-style: solid !important; }}

.status {{
    background: rgba(10, 13, 28, 0.90) !important;
    border: 1px solid rgba(139, 92, 246, 0.28) !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
}}

.status .bar {{
    height: 9px;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.14);
    overflow: hidden;
}}

.status .fill {{
    height: 100%;
    background: linear-gradient(90deg, {VIOLET}, {MAGENTA}, {CYAN});
    box-shadow: 0 0 12px rgba(139, 92, 246, 0.6);
    transition: width .4s;
}}

.status .busy .fill {{
    width: 35% !important;
    animation: slide 1.2s ease-in-out infinite alternate;
}}

@keyframes slide {{ from {{ margin-left: 0; }} to {{ margin-left: 65%; }} }}

.status .head {{
    display: flex;
    justify-content: space-between;
    font-weight: 700;
    margin: 2px 0 8px;
    color: #f1f5f9;
}}

.status .row {{
    display: grid;
    grid-template-columns: minmax(120px, max-content) 1fr 48px;
    gap: 10px;
    align-items: center;
    font-size: 13.5px;
    margin: 6px 0;
}}

.status .row span:first-child {{ white-space: nowrap; font-weight: 600; }}
.status .row span:last-child {{ text-align: right; opacity: .75; font-family: monospace; }}
.status .sum {{ font-size: 13.5px; color: #94a3b8; margin: 2px 0; }}
.status .err {{ color: #f43f5e; font-weight: 700; }}
.status .meta {{ font-size: 13px; color: #94a3b8; margin: 8px 0 0; font-family: monospace; }}
.status .lvs {{
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 4px 14px;
    margin: 10px 0 0;
    font-size: 13.5px;
    font-family: monospace;
}}
.status .lvs .dim {{ opacity: .55; }}
.status .star {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin: 14px 0 0;
    padding-top: 10px;
    border-top: 1px solid rgba(148, 163, 184, 0.12);
    font-size: 12.5px;
}}
.status .star > span {{ opacity: .7; }}
.status .star a {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 4px 11px;
    border-radius: 999px;
    border: 1px solid rgba(139, 92, 246, 0.4);
    background: rgba(139, 92, 246, 0.12);
    color: #e2e8f0 !important;
    text-decoration: none !important;
    font-size: 12px;
    white-space: nowrap;
}}
.status .star a:hover {{ border-color: {CYAN}; }}

#youtube-row {{
    flex-wrap: nowrap;
    align-items: center;
    gap: 6px;
    padding: 5px 6px 5px 12px;
    min-height: 0;
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 10px;
}}
#youtube-row .form, #youtube-link, #youtube-link * {{ background: transparent !important; border: none !important; box-shadow: none !important; }}
#youtube-link, #youtube-link .block, #youtube-link label {{ background: transparent !important; border: none !important;
               box-shadow: none !important; padding: 0 !important; min-width: 0 !important; }}
#youtube-link textarea, #youtube-link input {{ background: transparent !important; border: none !important;
               box-shadow: none !important; padding: 6px 0 !important; height: auto !important; }}
#youtube-go {{
    flex: 0 0 auto !important;
    min-width: 0 !important;
    width: auto !important;
    height: 32px;
    padding: 0 14px;
    border-radius: 8px;
    background: rgba(139, 92, 246, 0.22) !important;
    border: 1px solid rgba(139, 92, 246, 0.45) !important;
    color: #e2e8f0 !important;
    font-size: 13px;
    font-weight: 600;
    line-height: 1;
}}
#youtube-go:hover {{ border-color: {CYAN} !important; color: {CYAN} !important; }}
#song-details {{ flex-wrap: nowrap; align-items: center; }}
#cover-picker {{ flex: 0 0 124px; min-width: 124px !important; max-width: 124px; }}
#cover-picker .upload-container {{ min-height: 0; }}
#cover-picker .prompt, #cover-picker .prompt p, #cover-picker .prompt span {{
    white-space: normal; text-overflow: clip; overflow: visible; font-size: 12px; text-align: center;
}}
"""

SCROLL = """
(...args) => {
  document.getElementById('status')?.scrollIntoView({behavior: 'smooth', block: 'center'});
  return args;
}
"""

DARK = """
() => {
  const typed = (files) => {
    if (![...files].some((f) => !f.type)) return null;
    const dt = new DataTransfer();
    for (const f of files) {
      const ext = (f.name.match(/\\.(\\w+)$/) || [, ''])[1].toLowerCase();
      dt.items.add(f.type ? f : new File([f], ext === 'egg' ? f.name.replace(/\\.egg$/i, '.ogg') : f.name,
                                         {type: 'audio/' + (ext === 'egg' ? 'ogg' : ext)}));
    }
    return dt;
  };
  const inSong = (el) => el instanceof Element && el.closest('#song');
  document.addEventListener('drop', (e) => {
    if (!inSong(e.target) || !e.dataTransfer || e.nebulamapper) return;
    const dt = typed(e.dataTransfer.files);
    if (!dt) return;
    e.stopImmediatePropagation(); e.preventDefault();
    const again = new DragEvent('drop', {bubbles: true, cancelable: true, dataTransfer: dt});
    again.nebulamapper = true;
    e.target.dispatchEvent(again);
  }, true);
  document.addEventListener('change', (e) => {
    if (inSong(e.target) && e.target.files) { const dt = typed(e.target.files); if (dt) e.target.files = dt.files; }
  }, true);
  document.addEventListener('click', (e) => {
    if (inSong(e.target)) document.querySelectorAll('#song input[type=file]').forEach((i) => {
      if (!i.accept.includes('.egg')) i.accept += (i.accept ? ',' : '') + '.egg';
    });
  }, true);
  document.documentElement.classList.add('dark');
  document.body.classList.add('dark');
}
"""

state = {"model": None}
lock = threading.Lock()
stopping = threading.Event()


class Stopped(Exception):
    pass


def lead_in(song, folder):
    import librosa
    import numpy as np
    try:
        head, _ = librosa.load(song, sr=22050, mono=True, duration=LEAD_CHECK)
        if not len(head) or 20 * np.log10(librosa.feature.rms(y=head).max() + 1e-10) < LEAD_DB:
            return None
        title, artist, _, picture = artwork.song_info(song)
        out = folder / "lead_in" / f"{Path(song).stem}.ogg"
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(song), "-vn", "-af", f"adelay={LEAD_IN * 1000:g}:all=1",
                        "-c:a", "libvorbis", "-q:a", "6", "-map_metadata", "0", str(out)], check=True)
        cover = None
        if picture:
            cover = out.parent / "cover"
            cover.write_bytes(picture)
        return out, title, artist, cover
    except Exception:
        return None


def number(text, name):
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise gr.Error(f"{name} should be a number, like 174.")


class Status:

    NAMES = {"ExpertPlus": "Expert+"}
    REPLAY_RATE = 0.06

    def __init__(self, names=None, levels=1, replay=False, doing="Forging beatmap", stars=None):
        self.t0 = time.time()
        self.stars = stars or {}
        self.doing = doing
        self.phase, self.beats, self.found, self.scene = "Initializing neural pipeline", {}, "", ""
        self.notes = []
        self.rated, self.maps, self.replays = {}, {}, {}
        for n in names or []:
            self.beats[n] = (0, 1)
        self.started = {}
        self.levels, self.replay, self.song = levels, replay, None

    def update(self, msg):
        if not msg:
            return
        if msg[0] == "phase":
            self.phase = msg[1]
        elif msg[0] == "beat":
            _, name, done, total = msg
            self.beats[name] = (done, total)
            self.phase = "Simulating AI Replay" if name.endswith(" replay") else self.doing
            self.started.setdefault(name, (time.time(), done / total))
        elif msg[0] == "log":
            line = msg[1]
            if " BPM, " in line:
                self.found = line.split(", ", 1)[0]
                m = re.search(r",\s*([\d.]+)s\b", line)
                if m:
                    self.song = float(m.group(1))
            elif line.startswith(("the song starts right away", "that's really high", "on the first try")):
                if line not in self.notes:
                    self.notes.append(line)
            elif line == "making the lights":
                self.phase = "Making the lights"
            elif line.startswith("song:"):
                self.phase = "Making the map"
                try:
                    self.song = float(line.split()[1])
                except Exception:
                    pass
            elif line.startswith(("environment: ", "scene: ")):
                self.scene = line.split(": ", 1)[1]
            elif m := RATED_LINE.match(line):
                self.rated[m[1] or "map"] = m[2]
            elif m := MAP_LINE.match(line):
                self.maps[m[1] or "map"] = (self.rated.get(m[1] or "map", m[2]), m[3], m[4], m[5])
            elif m := REPLAY_LINE.search(line):
                name = line.split(" replay:", 1)[0] if " replay:" in line else "map"
                cut, total, bad, missed, _ = map(int, m.groups())
                good = cut - bad
                problems = [f"{n} {w}" for n, w in [(bad, "bad cut" if bad == 1 else "bad cuts"), (missed, "missed")] if n]
                self.replays[name] = f"{100 * good / max(1, total):.1f}% cut" + (f" ({', '.join(problems)})" if problems else "")
            elif line.startswith("wrote "):
                self.phase = "Packaging Beat Saber .zip archive"

    def label(self, name):
        replay = name.endswith(" replay")
        base = name.removesuffix(" replay")
        shown = self.NAMES.get(base, base)
        star = self.stars.get(base)
        return (f"{shown} ({star:g}★)" if star else shown) + (" AI Replay" if replay else "")

    def eta(self):
        if self.phase == "Packaging Beat Saber .zip archive":
            return "almost done"
        if self.phase == "Making the lights":
            return f"{int(time.time() - self.t0)}s"
        now = time.time()
        parts = []
        for k, (done, total) in self.beats.items():
            if not k.endswith(" replay") and k in self.started:
                t0, f0 = self.started[k]
                df = done / total - f0
                if df > 0.03 and now - t0 > 1.0:
                    parts.append((now - t0) * (1 - done / total) / df)
        if parts:
            left = max(parts)
            if self.replay and not any(k.endswith(" replay") for k in self.beats):
                left += (self.song or 180) * self.REPLAY_RATE
        elif any(k.endswith(" replay") for k in self.beats):
            rates = [(now - self.started[k][0]) / (done / total - self.started[k][1])
                     for k, (done, total) in self.beats.items()
                     if k.endswith(" replay") and k in self.started and done / total - self.started[k][1] > 0.05]
            rate = max(rates) if rates else (self.song or 180) * self.REPLAY_RATE
            left = max((1 - done / total) * rate for k, (done, total) in self.beats.items() if k.endswith(" replay"))
        else:
            return f"{int(now - self.t0)}s"
        if left < 5:
            return "a few seconds left"
        if left < 55:
            return f"about {int(round(left / 5) * 5)}s left"
        m = int(round(left / 60))
        return "about a minute left" if m == 1 else f"about {m} minutes left"

    def html(self, done=False, error=None):
        if error:
            return f'<div class="head"><span class="err">{html.escape(error)}</span></div>'
        rows = [f'<div class="row"><span>{html.escape(self.label(k))}</span>'
                f'<div class="bar"><div class="fill" style="width:{100 * d / t:.1f}%"></div></div>'
                f'<span>{int(100 * d / t)}%</span></div>' for k, (d, t) in self.beats.items()]
        meta = " · ".join(x for x in [self.found, self.scene and f"Scene: {self.scene}"] if x)
        info = ([f'<div class="meta">{html.escape(meta)}</div>'] if meta else []) + [
            f'<div class="meta">{html.escape(n)}</div>' for n in self.notes]
        if done:
            summary = []
            for k, (stars, njs, notes, nps) in self.maps.items():
                label = f'<span>{html.escape(self.NAMES.get(k, k))}</span>' if len(self.maps) > 1 or k != "map" else '<span></span>'
                rep = f' <span class="dim">· AI Replay: {html.escape(self.replays[k])}</span>' if k in self.replays else ""
                summary.append(f'{label}<span>{stars}★ · {notes} notes ({nps}/s) · NJS {njs}{rep}</span>')
            if not summary:
                for k, rep in self.replays.items():
                    label = f'<span>{html.escape(self.NAMES.get(k, k))}</span>' if len(self.replays) > 1 else '<span></span>'
                    summary.append(f'{label}<span>{html.escape(rep)}</span>')
            body = f'<div class="lvs">{"".join(summary)}</div>' if summary else ""
            took = int(time.time() - self.t0)
            return (f'<div class="head"><span style="color:#4ade80;">✓ Complete</span><span>{took}s</span></div>'
                    f'<div class="bar"><div class="fill" style="width:100%"></div></div>' + "".join(info) + body + STAR_ASK)
        head = f'<div class="head"><span>{html.escape(self.phase)}…</span><span>{self.eta()}</span></div>'
        if rows:
            return head + "".join(rows) + "".join(info)
        return head + '<div class="bar busy"><div class="fill"></div></div>' + "".join(info)


def follow(q, view):
    while True:
        try:
            msg = q.get(timeout=0.25)
        except queue.Empty:
            yield gr.update(), view.html()
            continue
        if msg is None:
            break
        view.update(msg)
        while not q.empty():
            msg = q.get_nowait()
            if msg is None:
                return
            view.update(msg)
        yield gr.update(), view.html()


def map_levels(path):
    """Read Standard difficulty tiers from a Beat Saber .zip archive using beatmap.read_info."""
    try:
        with beatmap.open_map(path) as z:
            info = beatmap.read_info(z)
        names = [d["difficulty"] for d in info.difficulties if d.get("characteristic") == "Standard"]
        if names:
            return [(SHOWN.get(n, n), n) for n in names]
    except Exception:
        pass
    return []


def player(device):
    from bsgen import botplay
    if "player" not in state or state["player"] is None:
        state["player"] = botplay.load_player(hub.player_model(), device)
    return state["player"]


def prefetch(device):
    try:
        from bsgen import rating, songstars
        with lock:
            if state["model"] is None:
                state["model"] = generate.load_map_model(hub.NEWEST, None, device)
                state["version"] = hub.NEWEST
            if state.get("lighter") is None:
                state["lighter"] = lightmodel.load(hub.light_model(), device)
        hub.timing_model()
        rhythm.loaded(hub.rhythm_model(), device)
        rating.available()
        songstars.load()
        print("NebulaMapper neural models ready.", flush=True)
    except Exception as e:
        print(f"could not pre-load models ({e}); will try again when you click Forge", flush=True)


NO_GPU = ("No GPU detected — running in CPU Turbo Mode (1-pass fast generation). "
          "On Colab: Runtime → Change runtime type → T4 GPU, then run both cells again.")


def show_link(local, public):
    BOLD, CYAN_C, DIM, RESET = "\033[1m", "\033[96m", "\033[2m", "\033[0m"
    shown = re.sub(r"://(0\.0\.0\.0|127\.0\.0\.1)\b", "://localhost", local)
    if public:
        print(f"\n  {BOLD}NebulaMapper Studio is live — open this link in your browser:{RESET}", flush=True)
        print(f"  {BOLD}{CYAN_C}➜  {public}{RESET}\n", flush=True)
        print(f"  {DIM}(local link if on this machine: {shown}){RESET}\n", flush=True)
    else:
        print(f"\n{BOLD}NebulaMapper Studio is running — open {CYAN_C}{shown}{RESET}\n", flush=True)


def star_labels(n, *vals):
    slots = export.difficulty_names(sorted(set(vals[:n])))
    by_star = dict(zip(sorted(set(vals[:n])), [SHOWN.get(s, s) for s in slots]))
    warn = f" · over {HIGH_STAR:g}★ can get chaotic"
    return [gr.update(label=f"Target Stars → {by_star.get(v, '')}" + (warn if v > HIGH_STAR else ""))
            if i < n else gr.update() for i, v in enumerate(vals)]


def diff_choices(by_name, version):
    if not by_name or version.split()[0] in hub.LEGACY:
        return list(DIFFS)
    return [(f"{d} ({by_name[name]:.1f}★)", d) if name in by_name else (d, d) for d, (_, name) in DIFFS.items()]


def scan_vault_files():
    """Find all .zip maps and .bsor replays in out/ directories."""
    files = []
    for base in (OUT, ROOT / "out"):
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in (".zip", ".bsor"):
                files.append(p)
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    seen = set()
    uniq = []
    for p in files:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            uniq.append(p)
    return uniq[:30]


def render_vault_html():
    items = scan_vault_files()
    if not items:
        return '<div class="sum">No maps or AI replays generated in <code>out/</code> yet. Forge a map first!</div>'
    rows = []
    for p in items:
        kind = "📦 Beatmap (.zip)" if p.suffix == ".zip" else "🤖 AI Replay (.bsor)"
        sz = p.stat().st_size
        sz_str = f"{sz / (1024 * 1024):.2f} MB" if sz >= 1024 * 1024 else f"{sz / 1024:.1f} KB"
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
        rows.append(
            f'<tr style="border-bottom:1px solid rgba(148,163,184,0.14);">'
            f'<td style="padding:7px 8px;font-weight:700;color:#e2e8f0;">{html.escape(p.name)}</td>'
            f'<td style="padding:7px 8px;color:#a78bfa;">{kind}</td>'
            f'<td style="padding:7px 8px;font-family:monospace;">{sz_str}</td>'
            f'<td style="padding:7px 8px;opacity:0.75;font-size:12px;">{mtime}</td>'
            f'</tr>'
        )
    return (
        '<div style="overflow-x:auto;">'
        '<table style="width:100%;border-collapse:collapse;font-size:13px;text-align:left;">'
        '<thead><tr style="border-bottom:1px solid rgba(148,163,184,0.3);color:#94a3b8;">'
        '<th style="padding:6px 8px;">Filename</th>'
        '<th style="padding:6px 8px;">Type</th>'
        '<th style="padding:6px 8px;">Size</th>'
        '<th style="padding:6px 8px;">Created</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def render_hf_status_html(repo_id=None, token=None):
    st = hf_sync.get_repo_status(repo_id=repo_id, token=token)
    present_count = len(hf_sync.ALL_MODEL_FILES) - len(st.get("missing", []))
    badge_color = "#22c55e" if st["complete"] else ("#f59e0b" if st["ok"] else "#f43f5e")
    badge_text = "All Neural Artifacts Synced" if st["complete"] else (
        f"{present_count}/{len(hf_sync.ALL_MODEL_FILES)} Files Present" if st["ok"] else "Offline / Auth Error"
    )
    rows = []
    for fname in hf_sync.ALL_MODEL_FILES:
        meta = st["files"].get(fname)
        sz = meta.get("size") if isinstance(meta, dict) else meta
        if sz is not None:
            mb = f"{sz / (1024 * 1024):.2f} MB" if sz >= 1024 * 1024 else f"{sz / 1024:.1f} KB"
            rows.append(
                f'<div style="display:flex;justify-content:space-between;padding:4px 8px;border-bottom:1px solid rgba(148,163,184,0.1);font-size:12.5px;">'
                f'<span>✅ <code>{html.escape(fname)}</code></span><span style="opacity:0.75;font-family:monospace;">{mb}</span></div>'
            )
        else:
            rows.append(
                f'<div style="display:flex;justify-content:space-between;padding:4px 8px;border-bottom:1px solid rgba(148,163,184,0.1);font-size:12.5px;">'
                f'<span>❌ <code>{html.escape(fname)}</code></span><span style="color:#f43f5e;">Missing</span></div>'
            )
    branches = ", ".join(st.get("branches", [])) or "main"
    tags = ", ".join(st.get("tags", [])) or "none"
    return f"""
    <div style="border:1px solid rgba(139,92,246,0.3);border-radius:12px;padding:14px;background:rgba(15,18,36,0.8);">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
        <div>
          <span style="font-weight:700;font-size:15px;">🤗 Hugging Face Model Hub: </span>
          <a href="{html.escape(st['url'])}" target="_blank" style="color:#22d3ee;font-weight:700;text-decoration:none;">
            {html.escape(st['repo_id'])}
          </a>
        </div>
        <span style="background:{badge_color};color:#fff;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:700;">
          {html.escape(badge_text)}
        </span>
      </div>
      <div style="font-size:12px;opacity:0.72;margin-bottom:8px;">
        Branches: <code>{html.escape(branches)}</code> · Tags: <code>{html.escape(tags)}</code>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:2px 16px;">
        {''.join(rows)}
      </div>
    </div>
    """


def get_mapper_choices(ckpt=None, force_download=False):
    mappers = ["Any"]
    try:
        raw = generate.mapper_names(ckpt, force_download=force_download)
        custom = [m for m in modelgen.DEFAULT_CUSTOM_MAPPERS if m in raw]
        others = sorted(m for m in raw if m not in custom)
        mappers += custom + others
    except Exception:
        mappers += list(modelgen.DEFAULT_CUSTOM_MAPPERS)
    return mappers


def build(ckpt, device):
    cfg_init = hub.load_config()
    hw_label = f"CUDA GPU ({device})" if str(device).startswith("cuda") else f"CPU Engine ({device})"

    def pick_levels(path):
        if not path:
            return (
                gr.update(choices=ALL_DIFF_CHOICES, value=["ExpertPlus"], visible=True),
                '<div class="sum">Upload any Beat Saber <code>.zip</code> map above — its available difficulties will be auto-detected here, or select the difficulty tiers you want to generate AI Replays for.</div>',
            )
        levels = map_levels(path)
        if not levels:
            return (
                gr.update(choices=ALL_DIFF_CHOICES, value=["ExpertPlus"], visible=True),
                '<div class="head"><span class="err">No Standard difficulties found in this .zip archive.</span></div>',
            )
        try:
            rep = analyzer.analyze_map_zip(path)
            badges = []
            for lv in rep.get("levels", []):
                st_txt = f"{lv['stars']:.1f}★" if lv.get("stars") else f"NJS {lv['njs']:g}"
                badges.append(f"<b>{html.escape(SHOWN.get(lv['difficulty'], lv['difficulty']))}</b> ({st_txt}, {lv['notes']} notes)")
            info_html = (
                f'<div style="padding:8px 12px;border-radius:10px;background:rgba(139,92,246,0.12);border:1px solid rgba(139,92,246,0.3);font-size:13px;">'
                f'✅ <b>{html.escape(rep.get("song_name") or Path(path).stem)}</b> · Found {len(levels)} difficulty tier(s): '
                + " · ".join(badges)
                + "</div>"
            )
        except Exception:
            info_html = f'<div class="sum">Detected difficulties: {", ".join(lbl for lbl, _ in levels)}</div>'
        return gr.update(choices=levels, value=[levels[-1][1]], visible=True), info_html

    def play(path, names, height, accuracy_pct, candidates_cnt, steps_str, player_name, seed):
        if not path:
            raise gr.Error("Upload a Beat Saber map (.zip) first.")
        if not names:
            raise gr.Error("Select at least one difficulty to generate an AI Replay for.")
        avail = [val for _, val in map_levels(path)]
        valid_names = [n for n in names if not avail or n in avail]
        if not valid_names and avail:
            valid_names = [avail[-1]]
        if not valid_names:
            raise gr.Error(f"Selected difficulty is not inside this map zip. Available in zip: {', '.join(avail)}")
        seed = number(seed, "Seed")
        steps_val = int(str(steps_str).split()[0]) if steps_str else 6
        acc_val = float(accuracy_pct) / 100.0
        pname = (player_name or "").strip() or "NebulaMapper Bot"
        folder = OUT / uuid.uuid4().hex[:10]
        folder.mkdir(parents=True, exist_ok=True)
        stem = Path(path).stem
        q, result = queue.Queue(), {}

        def one(name):
            import torch
            from bsgen import botplay
            stream = torch.cuda.Stream() if str(device).startswith("cuda") else None
            with torch.cuda.stream(stream) if stream else contextlib.nullcontext():
                motion = botplay.play_motion(
                    state["player"],
                    path,
                    name,
                    seed=int(seed or 0),
                    candidates=int(candidates_cnt),
                    steps=steps_val,
                    height=float(height),
                    accuracy=acc_val,
                    dev=device,
                    progress=lambda done, total: q.put(("beat", name, done, total)),
                )
            out = folder / f"{stem} {name}.bsor"
            s = botplay.save_replay(path, name, motion, out, player_name=pname)
            q.put(("log", f"{name} replay: {botplay.describe(s)}"))
            return out

        def work():
            try:
                with lock:
                    q.put(("phase", "Loading 6-DoF AI Replay model"))
                    player(device)
                    q.put(("phase", "Simulating AI saber swings"))
                    with ThreadPoolExecutor(len(valid_names)) as ex:
                        result["files"] = [str(f) for f in ex.map(one, valid_names)]
            except Exception as e:
                result["error"] = str(e)
            q.put(None)

        threading.Thread(target=work, daemon=True).start()
        view = Status(valid_names, len(valid_names), doing="Simulating AI saber swings")
        yield from follow(q, view)
        if "error" in result:
            yield gr.update(value=None, visible=False), view.html(error=result["error"])
            raise gr.Error(result["error"])
        yield gr.update(value=result["files"], visible=True), view.html(done=True)

    def from_youtube(link):
        from bsgen import youtube
        if not (link or "").strip():
            raise gr.Error("Paste a YouTube link first.")
        gr.Info("Fetching audio stream from YouTube…")
        try:
            path, note = youtube.download(link, OUT / "youtube" / uuid.uuid4().hex[:10])
            if note:
                gr.Warning(note)
            return str(path)
        except youtube.YouTubeError as e:
            raise gr.Error(str(e)) from None

    def make(song, youtube_url, mode, diff, count, *rest):
        stars = rest[:MAX_LEVELS]
        (speed_mode, style, arcs, extras, replay, njs, bpm, first_ms, seed, mapper, fmt, title, artist, cover, version, scene, rhythm_opt,
         color_scheme, cust_l, cust_r, cust_w, audio_preset, trim_start, trim_end, level_author,
         temp_val, cfg_val, loop_val, music_fit_chk, song_stars) = rest[MAX_LEVELS:]
        if not song and (youtube_url or "").strip():
            song = from_youtube(youtube_url.strip())
        if not song:
            raise gr.Error("Drop an audio track on the left or paste a YouTube URL first.")
        if mode == "Stars":
            levels = [(x, None) for x in sorted(set(stars[:count]))]
        else:
            legacy = version.split()[0] in hub.LEGACY and not ckpt
            levels = [(x if legacy else (song_stars or {}).get(name, x), name) for d, (x, name) in DIFFS.items() if d in diff]
            if not levels:
                raise gr.Error("Pick at least one difficulty.")
        folder = OUT / uuid.uuid4().hex[:10]
        folder.mkdir(parents=True, exist_ok=True)
        padded = lead_in(song, folder)
        if padded:
            song, tagged_title, tagged_artist, tagged_cover = padded
            title, artist, cover = title or tagged_title, artist or tagged_artist, cover or tagged_cover
        argv = ["--audio", str(song), "--stars", *(f"{x:g}" for x, _ in levels),
                "--format", fmt, "--device", device, "--out", str(folder)]
        if mode != "Stars":
            argv += ["--difficulty", *(name for _, name in levels)]
        version = version.split()[0]
        if ckpt:
            argv += ["--ckpt", ckpt]
        else:
            argv += ["--model", version]
        argv += ["--environment", SCENES[scene]]
        if speed_mode and "Turbo" in str(speed_mode):
            argv += ["--star-tries", "1", "--star-rounds", "1", "--rhythm-tries", "2"]
            if "Lights" in extras and not str(device).startswith("cuda"):
                argv += ["--lights", "rules"]
        elif speed_mode and "Precision" in str(speed_mode):
            argv += ["--star-tries", "2", "--star-rounds", "2", "--rhythm-tries", "12"]
        else:
            argv += ["--star-tries", "1", "--star-rounds", "1", "--rhythm-tries", "4"]
        if color_scheme and color_scheme != "Default (Game Colors)":
            argv += ["--color-scheme", color_scheme]
        if color_scheme == "Custom Hex Colors (Pick Below)":
            argv += ["--custom-colors", f"{cust_l},{cust_r},{cust_w}"]
        if audio_preset and audio_preset != "Original (No FX)":
            argv += ["--audio-preset", audio_preset]
        if (t_start := number(trim_start, "Trim Start")) is not None and t_start > 0:
            argv += ["--trim-start", f"{t_start:g}"]
        if (t_end := number(trim_end, "Trim End")) is not None and t_end > 0:
            argv += ["--trim-end", f"{t_end:g}"]
        if level_author and level_author.strip():
            argv += ["--level-author", level_author.strip()]
        for flag, value in (("--title", title), ("--artist", artist), ("--cover", cover)):
            if value:
                argv += [flag, str(value)]
        if (seed := number(seed, "Seed")) is not None:
            argv += ["--seed", str(int(seed))]
        if (njs := number(njs, "Note speed")) is not None:
            argv += ["--njs", f"{njs:g}"]
        if STYLES[style]:
            argv += ["--style", STYLES[style]]
        argv += ["--no-arcs"] if ARCS[arcs] is None else ["--arc-rate", f"{ARCS[arcs]:g}"]
        for part in ("Walls", "Bombs", "Lights"):
            if part not in extras:
                argv.append(f"--no-{part.lower()}")
        argv += ["--dry", f"{DRY:g}", "--xtc", f"{XTC:g}"]
        if float(temp_val or 0) > 0:
            argv += ["--temp", f"{float(temp_val):g}"]
        if float(cfg_val or 1.5) != 1.5:
            argv += ["--cfg", f"{float(cfg_val):g}"]
        if float(loop_val or 8.0) != 8.0:
            argv += ["--max-loop", f"{float(loop_val):g}"]
        if music_fit_chk:
            argv += ["--candidates", "2", "--music-fit"]
        if version == hub.NEWEST and not ckpt and rhythm_opt != RHYTHM[0]:
            argv.append("--rhythm" if rhythm_opt == RHYTHM[1] else "--no-rhythm")
        bpm, first_ms = number(bpm, "Song BPM"), number(first_ms, "First beat")
        if bpm:
            argv += ["--bpm", f"{bpm:g}"]
            if first_ms is not None:
                argv += ["--offset-ms", f"{first_ms + (LEAD_IN * 1000 if padded else 0):g}"]
        if mapper and mapper != "Any":
            argv += ["--mapper", mapper]
        if replay:
            argv.append("--replay")
        args, _ = generate.parser().parse_known_args(argv)

        q, result = queue.Queue(), {}
        if padded:
            q.put(("log", f"the song starts right away: {LEAD_IN:g} s of silence put in front"))

        def progress(name, done, total):
            if stopping.is_set():
                raise Stopped()
            q.put(("beat", name, done, total))

        def work():
            try:
                with lock:
                    if stopping.is_set():
                        raise Stopped()
                    if state["model"] is None or state.get("version") != version:
                        q.put(("phase", "Loading neural models from Hugging Face"))
                        state["model"] = None
                        state["model"] = generate.load_map_model(version, ckpt, device)
                        state["version"] = version
                    if replay:
                        player(device)
                    if "Lights" in extras and state.get("lighter") is None and version not in hub.LEGACY:
                        state["lighter"] = lightmodel.load(hub.light_model(), device)
                    q.put(("phase", "Analyzing audio spectrogram & beat grid"))
                    result["zip"] = generate.run(args, state["model"], log=lambda m: q.put(("log", m)),
                                                 progress=progress,
                                                 player=state.get("player"), lighter=state.get("lighter"))
            except Stopped:
                result["error"] = "Forge aborted by user"
            except Exception as e:
                result["error"] = str(e)
            q.put(None)

        stopping.clear()
        threading.Thread(target=work, daemon=True).start()
        slots = export.difficulty_names([x for x, _ in levels]) if mode == "Stars" else [name for _, name in levels]
        view = Status([name for _, name in levels] if mode != "Stars" else None, len(levels), bool(replay),
                      stars=dict(zip(slots, (x for x, _ in levels))))
        for files, shown in follow(q, view):
            yield files, shown, gr.update(visible=True), gr.update()
        if "error" in result:
            yield gr.update(value=None, visible=False), view.html(error=result["error"]), gr.update(visible=False), gr.update()
            if not stopping.is_set():
                raise gr.Error(result["error"])
            return
        z = Path(result["zip"])
        files = [str(z)] + sorted(str(p) for p in z.parent.iterdir()
                                  if p.suffix == ".bsor" and p.name.startswith(z.stem + " "))
        try:
            preview_html = analyzer.render_analysis_html(analyzer.analyze_map_zip(z))
        except Exception:
            preview_html = ""
        yield gr.update(value=files, visible=True), view.html(done=True), gr.update(visible=False), gr.update(value=preview_html, visible=bool(preview_html))

    mappers = get_mapper_choices(ckpt)

    header_svg = (
        '<svg id="nebula-emblem" viewBox="0 0 74 74" xmlns="http://www.w3.org/2000/svg">'
        '<defs>'
        '<linearGradient id="ng" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#c084fc"/><stop offset="50%" stop-color="#ec4899"/><stop offset="100%" stop-color="#22d3ee"/>'
        '</linearGradient>'
        '</defs>'
        '<rect x="2" y="2" width="70" height="70" rx="18" fill="#0d1021" stroke="url(#ng)" stroke-width="2.5"/>'
        '<path d="M18 56 L56 18" stroke="#22d3ee" stroke-width="5" stroke-linecap="round"/>'
        '<path d="M18 18 L56 56" stroke="#ec4899" stroke-width="5" stroke-linecap="round"/>'
        '<circle cx="37" cy="37" r="9" fill="#0d1021" stroke="#c084fc" stroke-width="2.5"/>'
        '</svg>'
    )

    with gr.Blocks(title="NebulaMapper — PhantomVoid Neural BeatStudio", analytics_enabled=False) as demo:
        gr.HTML(
            f'<div id="nebula-header">'
            f'  <div id="nebula-brand">'
            f'    {header_svg}'
            f'    <div>'
            f'      <h1><span class="grad">NEBULA</span>MAPPER <span style="font-size:14px;font-weight:600;color:#a78bfa;vertical-align:middle;margin-left:6px;">PHANTOMVOID NEURAL BEATSTUDIO</span></h1>'
            f'      <p>Neural Beat Saber Map Architecture · Chroma Lightshows · Autonomous AI Replays · Hugging Face Model Forge</p>'
            f'    </div>'
            f'  </div>'
            f'  <div id="nebula-badges">'
            f'    <span class="nebula-pill"><span class="dot"></span>{html.escape(hw_label)}</span>'
            f'    <a class="nebula-pill" href="https://huggingface.co/{html.escape(cfg_init["hf_repo"])}/tree/main" target="_blank" rel="noopener">'
            f'      🤗 Hub: {html.escape(cfg_init["hf_repo"])}'
            f'    </a>'
            f'    <a class="nebula-pill" href="https://github.com/GalaxyFSRP1/NebulaMapper" target="_blank" rel="noopener">'
            f'      ⚡ GitHub: GalaxyFSRP1/NebulaMapper'
            f'    </a>'
            f'  </div>'
            f'</div>'
        )
        if str(device) == "cpu":
            gr.HTML(
                f'<div id="no-gpu" style="color:#fda4af;background:rgba(244,63,94,0.1);border:1px solid rgba(244,63,94,0.3);'
                f'border-radius:10px;text-align:center;padding:8px 12px;margin-bottom:10px;font-size:13px;">'
                f'⚡ {html.escape(NO_GPU)}</div>'
            )

        with gr.Tabs():
            with gr.Tab("⚡ Map Forge"):
                with gr.Row(equal_height=False):
                    # LEFT DECK: Song & Difficulty
                    with gr.Column(scale=5, min_width=340):
                        with gr.Group(elem_classes=["studio-card"]):
                            gr.HTML('<div class="deck-heading"><span>🎵 1. Song</span></div>')
                            song = gr.Audio(label="Drop Song File (.mp3, .ogg, .wav, .flac, .egg)", type="filepath",
                                            format=None, sources=["upload"], elem_id="song")
                            with gr.Row(elem_id="youtube-row"):
                                youtube_link = gr.Textbox(show_label=False, scale=1, min_width=0, max_lines=1,
                                                          placeholder="Or paste a YouTube URL…", elem_id="youtube-link")
                                youtube_go = gr.Button("Fetch", scale=0, min_width=0, elem_id="youtube-go")
                            with gr.Row(visible=False, elem_id="song-details") as names:
                                cover = gr.Image(label="Cover", type="filepath", sources=["upload"],
                                                 height=124, width=124, min_width=124, scale=0, show_label=False,
                                                 buttons=[], placeholder="Click to upload cover art",
                                                 elem_id="cover-picker")
                                with gr.Column(scale=1, min_width=140):
                                    title = gr.Textbox(label="Song Title", show_label=False, placeholder="Song title")
                                    artist = gr.Textbox(label="Artist", show_label=False, placeholder="Artist")
                            auto_cover = gr.State(None)

                        with gr.Group(elem_classes=["studio-card"]):
                            gr.HTML('<div class="deck-heading"><span>🌌 2. Difficulty & Preset</span></div>')
                            preset_pick = gr.Dropdown(
                                list(QUICK_PRESETS.keys()),
                                value="Nebula Signature (PhantomVoid)",
                                label="Mapping Preset",
                            )
                            mode = gr.Radio(["Difficulty", "Stars"], value="Difficulty", label="Mode", elem_id="mode")
                            diff = gr.CheckboxGroup(list(DIFFS), value=["Expert+"], label="Difficulties")
                            song_stars = gr.State(None)
                            with gr.Group(visible=False) as star_box:
                                stars = [gr.Slider(1, 15, value=v, step=0.5, label=star_labels(1, v)[0]["label"] if i == 0 else "",
                                                   visible=i == 0) for i, v in enumerate(START_STARS)]
                                with gr.Row():
                                    more = gr.Button("+ Add Star Tier", size="sm")
                                    fewer = gr.Button("− Remove Tier", size="sm", visible=False)
                            count = gr.State(1)

                    # RIGHT DECK: Style, Colors & Launch
                    with gr.Column(scale=6, min_width=360):
                        with gr.Group(elem_classes=["studio-card"]):
                            gr.HTML('<div class="deck-heading"><span>🎨 3. Style, Colors & Options</span></div>')
                            with gr.Row():
                                color_scheme = gr.Dropdown(
                                    list(export.COLOR_SCHEMES.keys()),
                                    value="Nebula (Purple & Cyan)",
                                    label="Color Palette",
                                )
                                scene = gr.Dropdown(list(SCENES), value="Kaleidoscope", label="Stage Scene")
                            palette_preview = gr.HTML(export.render_palette_preview_html("Nebula (Purple & Cyan)"))
                            with gr.Row(visible=False) as custom_color_row:
                                cust_left = gr.ColorPicker(value="#a855f7", label="Left Saber & Light")
                                cust_right = gr.ColorPicker(value="#06b6d4", label="Right Saber & Light")
                                cust_wall = gr.ColorPicker(value="#4c1d95", label="Obstacles / Walls")
                            style = gr.Radio(list(STYLES), value="Auto", label="Pattern Style")
                            with gr.Row():
                                extras = gr.CheckboxGroup(
                                    ["Walls", "Bombs", "Lights"],
                                    value=["Walls", "Bombs", "Lights"],
                                    label="Include",
                                )
                                replay = gr.Checkbox(False, label="Also make AI Replays (.bsor)")

                            with gr.Accordion("⚙️ More Options (Speed Mode, Audio FX, Trim, BPM, Seed & AI Controls)", open=False):
                                speed_mode = gr.Radio(
                                    ["⚡ Turbo (Fastest — 1-Pass)", "⚖️ Balanced", "🎯 Precision Ranked"],
                                    value="⚡ Turbo (Fastest — 1-Pass)",
                                    label="Generation Speed Mode",
                                )
                                arcs = gr.Radio(list(ARCS), value="Normal", label="Arcs")
                                with gr.Row():
                                    audio_preset = gr.Dropdown(
                                        list(audio_fx.AUDIO_PRESETS.keys()),
                                        value="Original (No FX)",
                                        label="Audio Speed / Pitch FX",
                                    )
                                    level_author = gr.Textbox(
                                        value=cfg_init.get("mapper_author", "NebulaMapper (PhantomVoidTTV)"),
                                        label="Mapper Credit (_levelAuthorName)",
                                    )
                                with gr.Row():
                                    trim_start = gr.Textbox(label="Trim Start (s)", placeholder="0")
                                    trim_end = gr.Textbox(label="Trim End (s)", placeholder="full song")
                                with gr.Row():
                                    mapper = gr.Dropdown(
                                        mappers,
                                        value="phantomvoid" if "phantomvoid" in mappers else "Any",
                                        label="Mapper Persona",
                                        visible=len(mappers) > 1,
                                    )
                                    fmt = gr.Radio(["v2", "v3"], value="v2", label="Beatmap Format")
                                with gr.Row():
                                    temp_slider = gr.Slider(0.0, 1.35, value=0.0, step=0.05, label="Temperature (0 = Auto)")
                                    cfg_slider = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="Star Fit (CFG)")
                                with gr.Row():
                                    loop_slider = gr.Slider(0.0, 16.0, value=8.0, step=1.0, label="Anti-Loop Window (Beats)")
                                    music_fit_chk = gr.Checkbox(False, label="Strict Music Onset Fit")
                                with gr.Row():
                                    bpm = gr.Textbox(label="Song BPM", placeholder="detected")
                                    first_ms = gr.Textbox(label="First Beat (ms)", placeholder="detected")
                                with gr.Row():
                                    njs = gr.Textbox(label="Note Speed (NJS)", placeholder="auto")
                                    seed = gr.Textbox(label="Seed", placeholder="random")
                                with gr.Row():
                                    version = gr.Dropdown(
                                        [f"{v} · {hub.MAP_NOTES[v]}" for v in hub.VERSIONS],
                                        value=f"{hub.NEWEST} · {hub.MAP_NOTES[hub.NEWEST]}",
                                        label="Model Version",
                                        visible=not ckpt,
                                    )
                                    rhythm_opt = gr.Dropdown(RHYTHM, value=RHYTHM[0], label="Rhythm Model", visible=not ckpt)

                        go = gr.Button("⚡ Forge Beatmap Pack", elem_id="go", size="lg")
                        stop = gr.Button("⏹ Stop", size="sm", visible=False)
                        status = gr.HTML(
                            '<div class="sum">Drop a song or paste a YouTube link on the left, then click <b>Forge Beatmap Pack</b>.</div>',
                            elem_id="status",
                            elem_classes=["status"],
                            padding=True,
                        )
                        out = gr.File(label="Forged Beatmap (.zip) & AI Replays (.bsor)", file_count="multiple", height=140, visible=False)
                        forge_preview = gr.HTML(visible=False, padding=True)

            with gr.Tab("🤖 AI Replays"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=340):
                        with gr.Group(elem_classes=["studio-card"]):
                            gr.HTML('<div class="deck-heading"><span>🤖 AI Replay Generator (.bsor)</span></div>')
                            map_zip = gr.File(
                                label="Upload Beat Saber Map (.zip)",
                                file_types=[".zip"],
                                type="filepath",
                                height=140,
                                elem_id="map-zip",
                            )
                            play_map_info = gr.HTML(
                                '<div class="sum">Upload a Beat Saber <code>.zip</code> map above and pick the difficulty below.</div>'
                            )
                            play_diffs = gr.CheckboxGroup(
                                choices=ALL_DIFF_CHOICES,
                                value=["ExpertPlus"],
                                label="Difficulty",
                                visible=True,
                            )
                            with gr.Accordion("⚙️ Bot Skill & Motion Settings", open=False):
                                with gr.Row():
                                    play_acc = gr.Slider(85.0, 99.5, value=96.5, step=0.5, label="Target Accuracy (%)")
                                    height = gr.Slider(1.4, 2.1, value=1.7, step=0.05, label="Player Height (m)")
                                with gr.Row():
                                    play_candidates = gr.Slider(1, 8, value=1, step=1, label="Candidates (1 = Fastest)")
                                    play_steps = gr.Radio(["4 (Ultra-Turbo)", "6 (Fast — Recommended)", "12 (Balanced)", "24 (Precision)"], value="6 (Fast — Recommended)", label="Motion Steps")
                                with gr.Row():
                                    play_name = gr.Textbox(value="NebulaMapper Bot", label="Player Name")
                                    play_seed = gr.Textbox(label="Seed", placeholder="0")
                            play_go = gr.Button("🤖 Generate AI Replays (.bsor)", elem_id="play", size="lg")

                    with gr.Column(scale=6, min_width=360):
                        with gr.Group(elem_classes=["studio-card"]):
                            gr.HTML(
                                '<div class="deck-heading">'
                                '<span>📡 AI Replay Telemetry & Download</span>'
                                '<a href="https://allpoland.github.io/ArcViewer/" target="_blank" rel="noopener" style="color:#22d3ee;font-size:11.5px;text-decoration:none;">🌐 Open ArcViewer 3D Replay Viewer ↗</a>'
                                '</div>'
                            )
                            play_status = gr.HTML(
                                '<div class="sum">Upload a Beat Saber <code>.zip</code> map, choose your difficulty tiers on the left, and click <b>Generate AI Replays (.bsor)</b>.</div>',
                                elem_classes=["status"],
                                padding=True,
                            )
                            play_out = gr.File(label="Generated AI Replays (.bsor)", file_count="multiple", height=180, visible=False)

            with gr.Tab("📊 Map Telemetry & Heatmap"):
                with gr.Group(elem_classes=["studio-card"]):
                    gr.HTML('<div class="deck-heading"><span>📈 Beatmap Inspector, Highway Slice, NPS Curve & 4×3 Heatmap</span></div>')
                    with gr.Row():
                        analyze_zip = gr.File(
                            label="Drop Any Beat Saber Map (.zip) to Inspect Telemetry",
                            file_types=[".zip"],
                            type="filepath",
                            height=120,
                            scale=3,
                        )
                        analyze_diff_pick = gr.Dropdown(
                            choices=["Auto (Highest Difficulty)", "Easy", "Normal", "Hard", "Expert", "ExpertPlus"],
                            value="Auto (Highest Difficulty)",
                            label="Difficulty Tier to Inspect Grid & Highway Slice",
                            scale=2,
                        )
                    analyze_html = gr.HTML(
                        '<div class="sum">Drop a Beat Saber <code>.zip</code> map above to compute Star ★ / Tech / Pass ratings, '
                        'Peak Highway Slice, NPS density curves, parity reset rates, and 4×3 note placement heatmaps.</div>',
                        padding=True,
                    )

            with gr.Tab("🗂️ Session Vault"):
                with gr.Group(elem_classes=["studio-card"]):
                    gr.HTML('<div class="deck-heading"><span>🗂️ Generated Beatmaps & AI Replays Vault</span></div>')
                    with gr.Row():
                        vault_refresh_btn = gr.Button("🔄 Refresh Session Vault", size="sm")
                        vault_send_replay_btn = gr.Button("🤖 Load Latest Map into AI Replays & Telemetry", size="sm")
                    vault_html = gr.HTML(render_vault_html())
                    vault_files = gr.File(
                        value=[str(p) for p in scan_vault_files()],
                        label="Download Any Generated Map (.zip) or AI Replay (.bsor)",
                        file_count="multiple",
                        height=180,
                    )

            with gr.Tab("🧬 Neural Model Studio & Hugging Face"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        hf_status_box = gr.HTML(render_hf_status_html(cfg_init["hf_repo"], cfg_init["hf_token"]))
                        with gr.Group(elem_classes=["studio-card"]):
                            gr.HTML('<div class="deck-heading"><span>🔗 Hugging Face Repository Connection</span></div>')
                            with gr.Row():
                                hf_repo_in = gr.Textbox(
                                    value=cfg_init["hf_repo"],
                                    label="Hugging Face Model Repo ID",
                                    placeholder="PhantomVoidTTV/NebulaMapper",
                                )
                                hf_branch_in = gr.Textbox(
                                    value=cfg_init.get("hf_branch", "main"),
                                    label="Branch",
                                    scale=0,
                                    min_width=100,
                                )
                            hf_token_in = gr.Textbox(
                                value=cfg_init["hf_token"],
                                label="Hugging Face Write Token",
                                type="password",
                            )
                            with gr.Row():
                                hf_save_btn = gr.Button("💾 Save Hub Config", size="sm")
                                hf_reload_btn = gr.Button("🔄 Pull Latest Weights from Hub", size="sm")

                    with gr.Column(scale=6):
                        with gr.Group(elem_classes=["studio-card"]):
                            gr.HTML('<div class="deck-heading"><span>🧬 Custom Model Synthesizer & Fine-Tuner</span></div>')
                            gen_mode = gr.Radio(
                                [
                                    ("Nebula Evolved (Synthesize Personas + Calibrate Weights)", "nebula_evolved"),
                                    ("Fine-Tune on Custom Beat Saber Maps (.zip)", "finetune"),
                                    ("Checkpoint Blender (Interpolate v1.2 + v1.1)", "blend"),
                                    ("Fresh Architecture from Scratch", "scratch"),
                                ],
                                value="nebula_evolved",
                                label="Synthesis Mode",
                            )
                            gen_components = gr.CheckboxGroup(
                                ["map", "rhythm", "lights", "player", "labeler", "timing"],
                                value=["map", "rhythm", "lights", "player", "labeler", "timing"],
                                label="Neural Sub-Models to Build & Upload",
                            )
                            custom_mappers_in = gr.Textbox(
                                value=", ".join(modelgen.DEFAULT_CUSTOM_MAPPERS),
                                label="Custom Mapper Personas (injected into map.safetensors & map.json)",
                            )
                            with gr.Row():
                                blend_alpha_in = gr.Slider(0.0, 0.5, value=0.08, step=0.02, label="v1.1 Checkpoint Blend Ratio")
                                creativity_in = gr.Slider(0.0, 0.5, value=0.10, step=0.05, label="Persona Creativity Bias")
                            with gr.Row():
                                density_scale_in = gr.Slider(0.8, 1.3, value=1.0, step=0.05, label="Rhythm HPS Density Scale")
                                intensity_bias_in = gr.Slider(-1.0, 1.0, value=0.0, step=0.1, label="Lightshow Intensity Bias")
                            with gr.Row():
                                smoothness_in = gr.Slider(-0.5, 0.5, value=0.0, step=0.1, label="AI Replay Motion Smoothness")
                                star_cal_in = gr.Slider(-1.0, 1.0, value=0.0, step=0.1, label="Star Rating Estimator Offset")
                            with gr.Group(visible=False) as finetune_box:
                                finetune_files = gr.File(
                                    label="Upload Beat Saber Map(s) (.zip) for Fine-Tuning",
                                    file_types=[".zip"],
                                    file_count="multiple",
                                    type="filepath",
                                    height=120,
                                )
                                train_steps_in = gr.Slider(2, 30, value=8, step=2, label="AdamW Fine-Tuning Steps")
                            with gr.Group(visible=False) as scratch_box:
                                scratch_size = gr.Radio(["nano", "small", "base"], value="base", label="Scratch Architecture Size")
                            auto_push_chk = gr.Checkbox(
                                True,
                                label="Automatically commit & push generated models to Hugging Face and activate immediately",
                            )
                            gen_models_btn = gr.Button("🚀 Synthesize Models & Push to Hugging Face", elem_id="gen-models-btn", size="lg")
                            gen_models_log = gr.HTML(
                                '<div class="sum">Configure your model synthesis parameters above and click <b>Synthesize Models & Push to Hugging Face</b>.</div>',
                                elem_classes=["status"],
                                padding=True,
                            )

        def on_color_change(scheme, cl, cr, cw):
            is_cust = scheme == "Custom Hex Colors (Pick Below)"
            return gr.update(visible=is_cust), export.render_palette_preview_html(scheme, cl, cr, cw)

        def apply_preset(preset_name, cl, cr, cw):
            p = QUICK_PRESETS.get(preset_name)
            if not p:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            return (
                gr.update(value=p["style"]),
                gr.update(value=p["arcs"]),
                gr.update(value=p["mapper"]),
                gr.update(value=p["color"]),
                gr.update(value=p["scene"]),
                export.render_palette_preview_html(p["color"], cl, cr, cw),
            )

        def resize(n):
            n = max(1, min(MAX_LEVELS, n))
            return [n, *(gr.update(visible=i < n) for i in range(MAX_LEVELS)),
                    gr.update(visible=n < MAX_LEVELS), gr.update(visible=n > 1)]

        def find_cover(t, a, duration=None, picture=None):
            from bsgen import artwork
            found = None
            if not picture and t:
                picture, found = artwork.find(t, a, duration)
            if not picture:
                return None, a or found or ""
            path = OUT / "covers" / f"{uuid.uuid4().hex[:10]}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(io.BytesIO(picture)) as im:
                    im.convert("RGB").save(path)
            except Exception:
                return None, a or found or ""
            return str(path), a or found or ""

        def song_stars_of(path):
            import librosa
            from bsgen import sample, songstars
            y, sr = librosa.load(path, sr=None, mono=True)
            return songstars.stars(*sample.song_mel(y, sr))

        def song_names(path, version):
            if not path:
                return gr.update(visible=False), "", "", None, None, gr.update(choices=diff_choices(None, version)), None
            from bsgen import artwork
            with ThreadPoolExecutor(1) as ex:
                found = ex.submit(song_stars_of, path)
                t, a, duration, picture = artwork.song_info(path)
                c, a = find_cover(t, a, duration, picture)
                try:
                    by_name = found.result()
                except Exception:
                    by_name = None
            return gr.update(visible=True), t, a, c, c, gr.update(choices=diff_choices(by_name, version)), by_name

        @functools.lru_cache(64)
        def found_cover(t, a):
            return find_cover(t, a)

        def renamed(t, a, c, auto):
            if c and c != auto:
                return c, auto, a
            c, a = found_cover(t.strip(), a.strip())
            return c, c, a

        def on_analyze(zip_file, sel_diff):
            if not zip_file:
                return '<div class="sum">Drop a Beat Saber <code>.zip</code> map above to compute Star ★ / Tech / Pass ratings, NPS density curves, and 4×3 heatmaps.</div>'
            try:
                rep = analyzer.analyze_map_zip(zip_file)
                diff_arg = None if not sel_diff or sel_diff.startswith("Auto") else sel_diff
                return analyzer.render_analysis_html(rep, selected_diff=diff_arg)
            except Exception as e:
                raise gr.Error(f"Could not analyze map zip: {e}")

        def on_refresh_vault():
            items = scan_vault_files()
            return render_vault_html(), [str(p) for p in items]

        def on_load_latest_map():
            zips = [p for p in scan_vault_files() if p.suffix == ".zip"]
            if not zips:
                raise gr.Error("No generated .zip maps found in Vault yet.")
            latest = str(zips[0])
            diffs_upd, info_html = pick_levels(latest)
            rep_html = on_analyze(latest, "Auto (Highest Difficulty)")
            gr.Info(f"Loaded {zips[0].name} into AI Replays & Map Telemetry!")
            return latest, diffs_upd, info_html, latest, rep_html

        def on_save_hf(repo_id, branch, token):
            repo_id = (repo_id or "").strip() or hub.DEFAULT_REPO
            branch = (branch or "").strip() or "main"
            token = (token or "").strip() or hub.DEFAULT_TOKEN
            hub.save_config(hf_repo=repo_id, hf_branch=branch, hf_token=token)
            gr.Info(f"Saved Hugging Face settings ({repo_id}@{branch})")
            return render_hf_status_html(repo_id, token)

        def on_reload_hf(repo_id, branch, token):
            repo_id = (repo_id or "").strip() or hub.DEFAULT_REPO
            branch = (branch or "").strip() or "main"
            token = (token or "").strip() or hub.DEFAULT_TOKEN
            hub.save_config(hf_repo=repo_id, hf_branch=branch, hf_token=token)
            with lock:
                state["model"] = None
                state["lighter"] = None
                state["player"] = None
                new_mappers = get_mapper_choices(ckpt, force_download=True)
            gr.Info(f"Pulled latest models from {repo_id}@{branch}!")
            return render_hf_status_html(repo_id, token), gr.update(choices=new_mappers, value=new_mappers[1] if len(new_mappers) > 1 else "Any")

        def on_generate_models(
            repo_id, branch, token, mode_val, comps, mappers_str, blend_a, creat, dens, intens, smooth, star_cal,
            ft_files, ft_steps, sc_size, auto_push
        ):
            repo_id = (repo_id or "").strip() or hub.DEFAULT_REPO
            branch = (branch or "").strip() or "main"
            token = (token or "").strip() or hub.DEFAULT_TOKEN
            hub.save_config(hf_repo=repo_id, hf_branch=branch, hf_token=token)
            if not comps:
                raise gr.Error("Select at least one model component to generate.")
            custom_list = [m.strip().lower() for m in (mappers_str or "").split(",") if m.strip()]
            logs = []

            def log_fn(msg):
                logs.append(html.escape(str(msg)))

            yield (
                '<div class="head"><span>Synthesizing neural weights & building models…</span></div>'
                '<div class="bar busy"><div class="fill" style="width:50%"></div></div>',
                gr.update(),
                gr.update(),
            )
            try:
                res = modelgen.generate_all_models(
                    out_dir=ROOT / "models" if not auto_push else None,
                    mode=mode_val,
                    size=sc_size,
                    components=comps,
                    custom_mappers=custom_list,
                    blend_alpha=float(blend_a),
                    creativity_bias=float(creat),
                    density_scale=float(dens),
                    intensity_bias=float(intens),
                    smoothness=float(smooth),
                    star_calibration=float(star_cal),
                    map_zip_paths=ft_files or [],
                    train_steps=int(ft_steps),
                    push_to_hub=bool(auto_push),
                    repo_id=repo_id,
                    token=token,
                    branch=branch,
                    sync_legacy=False,
                    log=log_fn,
                )
                with lock:
                    state["model"] = None
                    state["lighter"] = None
                    state["player"] = None
                    new_mappers = get_mapper_choices(ckpt, force_download=bool(res.get("hub_result")))
                commit_link = ""
                if res.get("hub_result"):
                    url = res["hub_result"].get("commit_url") or f"https://huggingface.co/{repo_id}/tree/{branch}"
                    commit_link = f' · <a href="{html.escape(url)}" target="_blank" style="color:#22d3ee;">View Commit on Hugging Face</a>'
                done_html = (
                    f'<div class="head"><span style="color:#4ade80;">✓ Neural Models Synthesized & Activated!{commit_link}</span></div>'
                    f'<div class="bar"><div class="fill" style="width:100%"></div></div>'
                    f'<div style="margin-top:8px;font-family:monospace;font-size:12px;opacity:0.85;line-height:1.5;">'
                    + "<br>".join(logs[-12:]) + "</div>"
                )
                yield (
                    done_html,
                    render_hf_status_html(repo_id, token),
                    gr.update(choices=new_mappers, value=custom_list[0] if custom_list and custom_list[0] in new_mappers else "Any"),
                )
            except Exception as e:
                err_html = f'<div class="head"><span class="err">Error: {html.escape(str(e))}</span></div>'
                yield err_html, render_hf_status_html(repo_id, token), gr.update()
                raise gr.Error(str(e))

        for ev in (youtube_go.click, youtube_link.submit):
            ev(from_youtube, youtube_link, song, concurrency_limit=1)
        preset_pick.change(apply_preset, [preset_pick, cust_left, cust_right, cust_wall],
                           [style, arcs, mapper, color_scheme, scene, palette_preview])
        for ctrl in (color_scheme, cust_left, cust_right, cust_wall):
            ctrl.change(on_color_change, [color_scheme, cust_left, cust_right, cust_wall], [custom_color_row, palette_preview])
        song.change(song_names, [song, version], [names, title, artist, cover, auto_cover, diff, song_stars])
        for box in (title, artist):
            box.blur(renamed, [title, artist, cover, auto_cover], [cover, auto_cover, artist])
        mode.change(lambda m: (gr.update(visible=m == "Difficulty"), gr.update(visible=m == "Stars")), mode, [diff, star_box])
        more.click(lambda n: resize(n + 1), count, [count, *stars, more, fewer])
        fewer.click(lambda n: resize(n - 1), count, [count, *stars, more, fewer])
        for ev in [count.change] + [x.change for x in stars]:
            ev(star_labels, [count, *stars], stars, concurrency_limit=None, show_progress="hidden")
        go.click(
            make,
            [song, youtube_link, mode, diff, count, *stars, speed_mode, style, arcs, extras, replay, njs, bpm, first_ms, seed, mapper, fmt,
             title, artist, cover, version, scene, rhythm_opt, color_scheme, cust_left, cust_right, cust_wall,
             audio_preset, trim_start, trim_end, level_author, temp_slider, cfg_slider, loop_slider, music_fit_chk, song_stars],
            [out, status, stop, forge_preview],
            js=SCROLL,
        )
        stop.click(stopping.set)
        version.change(lambda v: gr.update(visible=v.split()[0] == hub.NEWEST), version, rhythm_opt)
        version.change(lambda v, b: gr.update(choices=diff_choices(b, v)), [version, song_stars], diff)
        map_zip.change(pick_levels, map_zip, [play_diffs, play_map_info])
        play_go.click(
            play,
            [map_zip, play_diffs, height, play_acc, play_candidates, play_steps, play_name, play_seed],
            [play_out, play_status],
        )
        analyze_zip.change(on_analyze, [analyze_zip, analyze_diff_pick], analyze_html)
        analyze_diff_pick.change(on_analyze, [analyze_zip, analyze_diff_pick], analyze_html)
        vault_refresh_btn.click(on_refresh_vault, None, [vault_html, vault_files])
        vault_send_replay_btn.click(
            on_load_latest_map,
            None,
            [map_zip, play_diffs, play_map_info, analyze_zip, analyze_html],
        )
        gen_mode.change(
            lambda m: (gr.update(visible=m == "finetune"), gr.update(visible=m == "scratch")),
            gen_mode,
            [finetune_box, scratch_box],
        )
        hf_save_btn.click(on_save_hf, [hf_repo_in, hf_branch_in, hf_token_in], hf_status_box)
        hf_reload_btn.click(on_reload_hf, [hf_repo_in, hf_branch_in, hf_token_in], [hf_status_box, mapper])
        gen_models_btn.click(
            on_generate_models,
            [hf_repo_in, hf_branch_in, hf_token_in, gen_mode, gen_components, custom_mappers_in,
             blend_alpha_in, creativity_in, density_scale_in, intensity_bias_in, smoothness_in, star_cal_in,
             finetune_files, train_steps_in, scratch_size, auto_push_chk],
            [gen_models_log, hf_status_box, mapper],
        )
    return demo


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="map model checkpoint (default: the released model on Hugging Face)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--share", action="store_true", help="also make a public link (on Colab always)")
    ap.add_argument("--no-prefetch", action="store_true", help="load models lazily on first request")
    args = ap.parse_args()
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
        font=[gr.themes.GoogleFont("Space Grotesk"), "system-ui", "sans-serif"],
    )
    if str(args.device) == "cpu":
        print(NO_GPU, flush=True)
    demo = build(args.ckpt, args.device)
    share = args.share or "COLAB_RELEASE_TAG" in os.environ
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        _, local, public = demo.queue(default_concurrency_limit=1).launch(
            server_name=args.host, server_port=args.port, share=share, theme=theme, css=CSS, js=DARK,
            allowed_paths=[str(OUT), str(ROOT / "out")], quiet=True, prevent_thread_lock=True)
    if share and not public:
        print(said.getvalue(), end="", flush=True)
    show_link(local, public)
    if not args.ckpt and not args.no_prefetch:
        print("loading the models in the background...", flush=True)
        threading.Thread(target=prefetch, args=(args.device,), daemon=True).start()
    demo.block_thread()


if __name__ == "__main__":
    main()
