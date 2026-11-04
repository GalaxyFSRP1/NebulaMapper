import subprocess
from pathlib import Path

AUDIO_PRESETS = {
    "Original (No FX)": {"speed": 1.0, "pitch_semitones": 0.0, "bass_db": 0.0, "treble_db": 0.0, "normalize": False},
    "Loudness Normalized (-14 LUFS)": {"speed": 1.0, "pitch_semitones": 0.0, "bass_db": 0.0, "treble_db": 0.0, "normalize": True},
    "Bass Boost (EDM / Speed)": {"speed": 1.0, "pitch_semitones": 0.0, "bass_db": 5.5, "treble_db": 0.0, "normalize": True},
    "Synthwave Warmth (Sub-Bass + Crisp Highs)": {"speed": 1.0, "pitch_semitones": 0.0, "bass_db": 4.0, "treble_db": 3.0, "normalize": True},
    "Speed Practice (1.10x Tempo)": {"speed": 1.10, "pitch_semitones": 0.0, "bass_db": 0.0, "treble_db": 0.0, "normalize": False},
    "Faster Song (1.15x Tempo)": {"speed": 1.15, "pitch_semitones": 0.0, "bass_db": 0.0, "treble_db": 0.0, "normalize": False},
    "Super Fast (1.25x Tempo)": {"speed": 1.25, "pitch_semitones": 0.0, "bass_db": 0.0, "treble_db": 0.0, "normalize": False},
    "Hyper Speed (1.35x Tempo)": {"speed": 1.35, "pitch_semitones": 0.0, "bass_db": 0.0, "treble_db": 0.0, "normalize": False},
    "Nightcore (1.20x + Pitch)": {"speed": 1.20, "pitch_semitones": 3.15, "bass_db": 2.0, "treble_db": 1.5, "normalize": True},
    "Daycore / Chill (0.85x)": {"speed": 0.85, "pitch_semitones": -2.8, "bass_db": 1.5, "treble_db": 0.0, "normalize": True},
}


def apply_audio_fx(
    song_path: str | Path,
    out_dir: str | Path,
    preset: str = "Original (No FX)",
    speed: float | None = None,
    pitch_semitones: float | None = None,
    bass_db: float | None = None,
    treble_db: float | None = None,
    normalize: bool | None = None,
    trim_start: float | None = None,
    trim_end: float | None = None,
) -> Path:
    """Apply speed, pitch, bass/treble EQ, loudness normalization, or audio trimming via ffmpeg."""
    p_cfg = dict(AUDIO_PRESETS.get(preset, AUDIO_PRESETS["Original (No FX)"]))
    spd = float(speed if speed is not None else p_cfg["speed"])
    semis = float(pitch_semitones if pitch_semitones is not None else p_cfg["pitch_semitones"])
    bass = float(bass_db if bass_db is not None else p_cfg["bass_db"])
    treble = float(treble_db if treble_db is not None else p_cfg.get("treble_db", 0.0))
    norm = bool(normalize if normalize is not None else p_cfg["normalize"])
    has_trim = (trim_start is not None and trim_start > 0) or (trim_end is not None and trim_end > 0)

    if abs(spd - 1.0) < 1e-3 and abs(semis) < 1e-3 and abs(bass) < 1e-3 and abs(treble) < 1e-3 and not norm and not has_trim:
        return Path(song_path)

    out_dir = Path(out_dir) / "audio_fx"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(song_path).stem}_fx.ogg"

    filters = []
    pitch_factor = 2.0 ** (semis / 12.0)
    if abs(pitch_factor - 1.0) >= 1e-3:
        filters.append(f"asetrate=44100*{pitch_factor:.5f},aresample=44100")
        tempo_factor = spd / pitch_factor
    else:
        tempo_factor = spd

    while tempo_factor > 2.0:
        filters.append("atempo=2.0")
        tempo_factor /= 2.0
    while tempo_factor < 0.5:
        filters.append("atempo=0.5")
        tempo_factor /= 0.5
    if abs(tempo_factor - 1.0) >= 1e-3:
        filters.append(f"atempo={tempo_factor:.5f}")

    if abs(bass) >= 0.1:
        filters.append(f"bass=g={bass:.1f}:f=100:w=0.6")
    if abs(treble) >= 0.1:
        filters.append(f"treble=g={treble:.1f}:f=4500:w=0.6")

    if norm:
        filters.append("loudnorm=I=-14:TP=-1.5:LRA=11")

    cmd = ["ffmpeg", "-y", "-v", "error"]
    if trim_start is not None and trim_start > 0:
        cmd += ["-ss", f"{trim_start:g}"]
    if trim_end is not None and trim_end > 0:
        cmd += ["-to", f"{trim_end:g}"]
    cmd += ["-i", str(song_path), "-vn"]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-c:a", "libvorbis", "-q:a", "6", "-map_metadata", "0", str(out_path)]
    subprocess.run(cmd, check=True)
    return out_path
