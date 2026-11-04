import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from . import beatmap, data, hf_sync, hub, labeler, lightmodel, model as M, player as P, rating, rhythm, sample, songstars, timing, tokens

DEFAULT_CUSTOM_MAPPERS = [
    "phantomvoid",
    "nebulamapper",
    "nebula_tech",
    "nebula_speed",
    "nebula_acc",
]

# Recipes of existing top mappers to synthesize rich embeddings for custom NebulaMapper personas
PERSONA_RECIPES = {
    "phantomvoid": [("cerret", 0.30), ("nolanimations", 0.25), ("joshabi", 0.25), ("skeelie", 0.20)],
    "nebulamapper": [("joetastic", 0.30), ("alice", 0.25), ("eurnkung", 0.25), ("bytrius", 0.20)],
    "nebula_tech": [("jabob", 0.35), ("thyspoon", 0.35), ("timbo", 0.30)],
    "nebula_speed": [("cerret", 0.40), ("pogu", 0.35), ("abcbadq", 0.25)],
    "nebula_acc": [("alice", 0.40), ("misterlihao", 0.35), ("mux", 0.25)],
}

ARCH_PRESETS = {
    "nano": {
        "map": {"d": 128, "layers": 2},
        "rhythm": {"d": 128, "enc_layers": 2, "dec_layers": 2, "heads": 4, "stem": 48},
        "lights": {"d": 128, "layers": 2, "heads": 4},
        "player": {"d": 128, "layers": 2, "heads": 4},
    },
    "small": {
        "map": {"d": 256, "layers": 4},
        "rhythm": {"d": 256, "enc_layers": 4, "dec_layers": 4, "heads": 4, "stem": 64},
        "lights": {"d": 192, "layers": 4, "heads": 4},
        "player": {"d": 256, "layers": 6, "heads": 4},
    },
    "base": {
        "map": {"d": 512, "layers": 8},
        "rhythm": {"d": 512, "enc_layers": 6, "dec_layers": 6, "heads": 8, "stem": 96},
        "lights": {"d": 256, "layers": 8, "heads": 4},
        "player": {"d": 512, "layers": 12, "heads": 8},
    },
}


def _clone_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Clone all tensors so tied weights (e.g. tok.weight and out.weight) don't share storage in safetensors."""
    return {k: v.detach().cpu().float().contiguous().clone() for k, v in sd.items()}


def _base_hub_file(filename: str, revision: str = "v1.2") -> Path:
    from huggingface_hub import hf_hub_download
    cfg = hub.load_config()
    token = cfg.get("hf_token") or None
    # Try fallback repo or primary repo
    for r_id in (cfg.get("fallback_repo") or hub.FALLBACK_REPO, cfg.get("hf_repo") or hub.DEFAULT_REPO):
        try:
            return Path(hf_hub_download(r_id, filename, revision=revision, token=token))
        except Exception:
            continue
    raise RuntimeError(f"Could not download base file {filename}@{revision}")


def inject_custom_mappers(
    sd: dict[str, torch.Tensor],
    mappers: list[str],
    custom_mappers: list[str],
    creativity_bias: float = 0.0,
) -> list[str]:
    """Inject custom mapper personas into the 128-slot mapper table & synthesize their token embeddings."""
    mappers = list(mappers)
    custom_clean = []
    for cm in custom_mappers:
        name = cm.strip().lower()
        if name and name not in custom_clean:
            custom_clean.append(name)

    if not custom_clean:
        return mappers

    mapper0 = tokens.VOCAB.cond0["mapper"]
    tok_w = sd["tok.weight"].clone()
    name_to_idx = {m: i for i, m in enumerate(mappers)}

    # Place custom mappers in the last slots of the 128-mapper list (or update existing if already present)
    replace_pos = len(mappers) - 1
    for cm in reversed(custom_clean):
        if cm in name_to_idx:
            slot = name_to_idx[cm]
        else:
            while replace_pos >= 0 and mappers[replace_pos] in custom_clean:
                replace_pos -= 1
            if replace_pos < 0:
                break
            slot = replace_pos
            mappers[slot] = cm
            name_to_idx[cm] = slot
            replace_pos -= 1

        recipe = PERSONA_RECIPES.get(cm)
        emb = torch.zeros_like(tok_w[mapper0 + slot])
        total_w = 0.0
        if recipe:
            for src_name, w in recipe:
                if src_name in name_to_idx:
                    emb += w * tok_w[mapper0 + name_to_idx[src_name]]
                    total_w += w
        if total_w > 0:
            emb = emb / total_w
        else:
            # Blend top 5 mappers with a deterministic signature for any user-defined custom mapper name
            gen = torch.Generator().manual_seed(abs(hash(cm)) % (2**31))
            weights = torch.softmax(torch.randn(min(8, len(mappers)), generator=gen), dim=0)
            for idx_k, w in enumerate(weights):
                emb += float(w) * tok_w[mapper0 + idx_k]

        if creativity_bias != 0.0:
            gen = torch.Generator().manual_seed(abs(hash(cm + "_bias")) % (2**31))
            noise = torch.randn(emb.shape, generator=gen) * (emb.std() * 0.12 * float(creativity_bias))
            emb = emb + noise

        tok_w[mapper0 + slot] = emb

    sd["tok.weight"] = tok_w
    if "out.weight" in sd:
        sd["out.weight"] = tok_w.clone()
    return mappers


def generate_map_model(
    out_dir: Path,
    mode: str = "nebula_evolved",
    size: str = "base",
    custom_mappers: list[str] | None = None,
    blend_alpha: float = 0.08,
    creativity_bias: float = 0.1,
    map_zip_paths: list[str | Path] | None = None,
    train_steps: int = 8,
    lr: float = 5e-5,
    log: Callable[[str], None] = print,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    custom_mappers = custom_mappers if custom_mappers is not None else list(DEFAULT_CUSTOM_MAPPERS)

    if mode == "scratch":
        preset = ARCH_PRESETS.get(size, ARCH_PRESETS["base"])["map"]
        args = {
            "d": preset["d"],
            "layers": preset["layers"],
            "no_swings": False,
            "conv_blocks": 0,
            "global_cond": False,
            "memory": True,
            "mem_align": True,
            "per_beat": 12,
            "hits": None,
        }
        log(f"[Map Model] Building fresh Generator architecture (d={args['d']}, layers={args['layers']})...")
        m = M.Generator(
            tokens.VOCAB.size,
            d=args["d"],
            enc_layers=args["layers"],
            dec_layers=args["layers"],
            heads=max(1, args["d"] // 64),
            memory=True,
            mem_align=True,
            per_beat=12,
        )
        base_mappers = json.loads(_base_hub_file("map.json", "v1.2").read_text(encoding="utf-8")).get("mappers", [])
        sd = _clone_state_dict(m.state_dict())
        mappers = inject_custom_mappers(sd, base_mappers, custom_mappers, creativity_bias)
    else:
        log(f"[Map Model] Loading base v1.2 checkpoint for mode='{mode}'...")
        st_12 = _base_hub_file("map.safetensors", "v1.2")
        js_12 = json.loads(_base_hub_file("map.json", "v1.2").read_text(encoding="utf-8"))
        sd = _clone_state_dict(load_file(st_12))
        args = dict(js_12["args"])
        mappers = list(js_12.get("mappers", []))

        if mode in ("nebula_evolved", "blend") and blend_alpha > 0:
            try:
                st_11 = _base_hub_file("map.safetensors", "v1.1")
                sd_11 = load_file(st_11)
                alpha = float(np.clip(blend_alpha, 0.0, 1.0))
                blended = 0
                for k, v in sd.items():
                    if k in sd_11 and sd_11[k].shape == v.shape:
                        sd[k] = ((1.0 - alpha) * v + alpha * sd_11[k].cpu().float()).contiguous().clone()
                        blended += 1
                log(f"[Map Model] Blended {blended} compatible tensors with v1.1 (alpha={alpha:.2f}).")
            except Exception as e:
                log(f"[Map Model] Note: skipped v1.1 blend ({e}).")

        mappers = inject_custom_mappers(sd, mappers, custom_mappers, creativity_bias)
        log(f"[Map Model] Synthesized custom mapper personas: {', '.join(custom_mappers)}")

        if mode == "finetune" and map_zip_paths:
            sd = _finetune_map_on_zips(sd, args, mappers, map_zip_paths, steps=train_steps, lr=lr, log=log)

    st_path = out_dir / "map.safetensors"
    js_path = out_dir / "map.json"
    save_file(_clone_state_dict(sd), str(st_path))
    js_path.write_text(json.dumps({"args": args, "mappers": mappers}, indent=1) + "\n", encoding="utf-8")
    log(f"[Map Model] Saved {st_path.name} ({st_path.stat().st_size / 1e6:.1f} MB) and {js_path.name}")
    return st_path, js_path


def _finetune_map_on_zips(
    sd: dict[str, torch.Tensor],
    args: dict,
    mappers: list[str],
    map_zip_paths: list[str | Path],
    steps: int = 8,
    lr: float = 5e-5,
    log: Callable[[str], None] = print,
) -> dict[str, torch.Tensor]:
    """Fine-tune the Generator on user-uploaded Beat Saber .zip maps."""
    vocab, max_tokens = sd["tok.weight"].shape[0], sd["dec_pos"].shape[0]
    m = M.Generator(
        vocab,
        d=args["d"],
        enc_layers=args["layers"],
        dec_layers=args["layers"],
        heads=args["d"] // 64,
        conv_blocks=args.get("conv_blocks", 0),
        global_cond=args.get("global_cond", False),
        max_tokens=max_tokens,
        memory=args.get("memory", False),
        mem_align=args.get("mem_align", False),
        per_beat=args.get("per_beat", 12),
        hits=bool(args.get("hits", False)),
    )
    m.load_state_dict(sd)
    m.train()

    # Freeze backbone except output/adapter/embeddings so CPU/GPU fine-tuning is fast and stable
    for name, p in m.named_parameters():
        p.requires_grad = any(k in name for k in ("tok", "norm", "dec.layers.7", "dec.layers.6", "mem_roll"))

    windows = []
    for zp in map_zip_paths:
        try:
            with beatmap.open_map(zp) as z:
                info = beatmap.read_info(z)
                y, sr = beatmap.read_audio(z, info)
                logmel, fps = sample.song_mel(y, sr)
                for dmeta in info.difficulties:
                    if dmeta["characteristic"] != "Standard":
                        continue
                    diff = beatmap.read_difficulty(z, info, "Standard", dmeta["difficulty"])
                    if len(diff.notes) < 16:
                        continue
                    offset, _ = beatmap.grid_offset(diff.notes["beat"])
                    n_beats = int(diff.notes["beat"].max()) + 16
                    beat_times = diff.tempo.to_seconds(np.arange(0, n_beats + 16, dtype=float)) + info.song_offset
                    lab = labeler.parity_rule(diff.notes)
                    owner, swings, swing_beat = labeler.to_swings(diff.notes, lab, offset)
                    ev = tokens.events(diff.notes, owner, swings, swing_beat, offset, diff.bombs, diff.walls, diff.arcs)
                    ov, bar = data.song_overview(logmel, fps, beat_times, offset)
                    est = rating.estimate(diff.notes, diff.njs or 16.0, diff.notes["time"])
                    star_val = est[0] if est else 7.5
                    cond = {
                        "stars": star_val,
                        "style": None,
                        "mapper": mappers.index("phantomvoid") if "phantomvoid" in mappers else 0,
                        "njs": diff.njs or 16.0,
                        "nps": len(diff.notes) / max(float(np.ptp(diff.notes["time"])), 1.0),
                        "objects": tokens.OBJ_WALLS | tokens.OBJ_BOMBS | tokens.OBJ_ARCS,
                        "pass": est[2] if est else None,
                        "acc": None,
                        "tech": est[1] if est else None,
                        "wnps": None,
                    }
                    pb = args.get("per_beat", 12)
                    for b in range(0, max(1, n_beats - 32), 32):
                        t0 = b * tokens.TICKS
                        tk = tokens.encode(ev, t0, cond)[:max_tokens]
                        if len(tk) < 16:
                            continue
                        mask = data.loss_mask(tk, t0)
                        if mask.sum() < 4:
                            continue
                        mel_w = data.beat_mel(logmel, fps, beat_times, b + offset, n=(tokens.CONTEXT + tokens.WINDOW + data.AHEAD) * pb, per_beat=pb)
                        rel_w = data.overview_rel(bar, b)
                        mem_in = data.memory_input(ev, 0, b, logmel, fps, beat_times, offset, pb)
                        windows.append((mel_w, ov, rel_w, *mem_in, tk, mask))
        except Exception as e:
            log(f"[Map Fine-Tune] Warning reading {zp}: {e}")

    if not windows:
        log("[Map Fine-Tune] No valid training windows found in supplied zip(s); keeping evolved weights.")
        return _clone_state_dict(m.state_dict())

    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    log(f"[Map Fine-Tune] Fine-tuning on {len(windows)} beatmap windows for {steps} steps...")
    for step_idx in range(steps):
        mel_w, ov, rel_w, mem_mel, roll, sim, tk, mask = windows[step_idx % len(windows)]
        audio = (
            torch.from_numpy(mel_w)[None].float(),
            torch.from_numpy(ov)[None].float(),
            torch.from_numpy(rel_w)[None].long(),
            torch.from_numpy(mem_mel)[None].float(),
            torch.from_numpy(roll)[None].float(),
            torch.from_numpy(sim)[None].float(),
        )
        tk_t = torch.tensor(tk, dtype=torch.long)[None]
        mask_t = torch.from_numpy(mask)[None].float()
        opt.zero_grad()
        loss = m.loss(audio, tk_t, mask_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        log(f"[Map Fine-Tune] Step {step_idx + 1}/{steps} — loss={loss.item():.4f}")

    m.eval()
    return _clone_state_dict(m.state_dict())


def generate_rhythm_model(
    out_dir: Path,
    mode: str = "nebula_evolved",
    size: str = "base",
    density_scale: float = 1.0,
    log: Callable[[str], None] = print,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js_base = json.loads(_base_hub_file("rhythm.json", "v1.2").read_text(encoding="utf-8"))
    patterns = js_base["patterns"]
    usual = js_base.get("usual", [])
    if density_scale != 1.0 and usual:
        usual = [[float(s), round(float(h) * float(density_scale), 4)] for s, h in usual]

    if mode == "scratch":
        preset = ARCH_PRESETS.get(size, ARCH_PRESETS["base"])["rhythm"]
        config = {
            "n_vocab": len(patterns) + 1,
            "d": preset["d"],
            "enc_layers": preset["enc_layers"],
            "dec_layers": preset["dec_layers"],
            "heads": preset["heads"],
            "stem": preset["stem"],
            "level": True,
            "ratings": True,
        }
        log(f"[Rhythm Model] Building fresh RhythmModel (d={config['d']}, layers={config['enc_layers']})...")
        rm = rhythm.RhythmModel(**config)
        sd = _clone_state_dict(rm.state_dict())
    else:
        log(f"[Rhythm Model] Preparing RhythmModel checkpoint (density_scale={density_scale:.2f})...")
        config = dict(js_base["config"])
        sd = _clone_state_dict(load_file(_base_hub_file("rhythm.safetensors", "v1.2")))

    st_path = out_dir / "rhythm.safetensors"
    js_path = out_dir / "rhythm.json"
    save_file(sd, str(st_path))
    js_path.write_text(json.dumps({"config": config, "patterns": patterns, "usual": usual}) + "\n", encoding="utf-8")
    log(f"[Rhythm Model] Saved {st_path.name} ({st_path.stat().st_size / 1e6:.1f} MB) and {js_path.name}")
    return st_path, js_path


def generate_lights_model(
    out_dir: Path,
    mode: str = "nebula_evolved",
    size: str = "base",
    intensity_bias: float = 0.0,
    log: Callable[[str], None] = print,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "scratch":
        preset = ARCH_PRESETS.get(size, ARCH_PRESETS["base"])["lights"]
        config = {"d": preset["d"], "layers": preset["layers"], "heads": preset["heads"]}
        log(f"[Lighting Model] Building fresh Lighter model (d={config['d']}, layers={config['layers']})...")
        lm = lightmodel.Lighter(**config)
        sd = _clone_state_dict(lm.state_dict())
    else:
        js_base = json.loads(_base_hub_file("lights.json", "v1.2").read_text(encoding="utf-8"))
        config = dict(js_base["config"])
        sd = _clone_state_dict(load_file(_base_hub_file("lights.safetensors", "v1.2")))
        if intensity_bias != 0.0 and "out.bias" in sd:
            b = sd["out.bias"].view(lightmodel.CHANNELS, lightmodel.N_VALUES + 1).clone()
            b[:, 1:] += float(intensity_bias) * 0.25
            sd["out.bias"] = b.view(-1)
            log(f"[Lighting Model] Applied lightshow intensity bias ({intensity_bias:+.2f}).")

    st_path = out_dir / "lights.safetensors"
    js_path = out_dir / "lights.json"
    save_file(sd, str(st_path))
    js_path.write_text(json.dumps({"config": config}, indent=1) + "\n", encoding="utf-8")
    log(f"[Lighting Model] Saved {st_path.name} ({st_path.stat().st_size / 1e6:.1f} MB) and {js_path.name}")
    return st_path, js_path


def generate_player_model(
    out_dir: Path,
    mode: str = "nebula_evolved",
    size: str = "base",
    smoothness: float = 0.0,
    log: Callable[[str], None] = print,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "scratch":
        preset = ARCH_PRESETS.get(size, ARCH_PRESETS["base"])["player"]
        config = {"d": preset["d"], "layers": preset["layers"], "heads": preset["heads"], "plan": True, "conds": 3}
        log(f"[Player Model] Building fresh Flow-Matching Player model (d={config['d']}, layers={config['layers']})...")
        pm = P.Player(**config)
        sd = _clone_state_dict(pm.state_dict())
    else:
        js_base = json.loads(_base_hub_file("player.json", "v1.2").read_text(encoding="utf-8"))
        config = dict(js_base["config"])
        sd = _clone_state_dict(load_file(_base_hub_file("player.safetensors", "v1.2")))
        if smoothness != 0.0 and "dstd" in sd:
            sd["dstd"] = (sd["dstd"] * (1.0 - 0.1 * float(np.clip(smoothness, -0.5, 0.5)))).clone()
            log(f"[Player Model] Calibrated motion trajectory smoothness ({smoothness:+.2f}).")

    st_path = out_dir / "player.safetensors"
    js_path = out_dir / "player.json"
    save_file(sd, str(st_path))
    js_path.write_text(json.dumps({"config": config}, indent=1) + "\n", encoding="utf-8")
    log(f"[Player Model] Saved {st_path.name} ({st_path.stat().st_size / 1e6:.1f} MB) and {js_path.name}")
    return st_path, js_path


def generate_labeler_model(
    out_dir: Path,
    mode: str = "nebula_evolved",
    log: Callable[[str], None] = print,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "scratch":
        log("[Labeler Model] Building fresh Swing Parity Labeler model...")
        lb = labeler.Labeler()
        sd = _clone_state_dict(lb.state_dict())
    else:
        sd = _clone_state_dict(load_file(_base_hub_file("labeler.safetensors", "v1.2")))
    st_path = out_dir / "labeler.safetensors"
    save_file(sd, str(st_path))
    log(f"[Labeler Model] Saved {st_path.name} ({st_path.stat().st_size / 1e6:.1f} MB)")
    return st_path


def generate_timing_and_trees(
    out_dir: Path,
    mode: str = "nebula_evolved",
    star_calibration: float = 0.0,
    log: Callable[[str], None] = print,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # 1. timing.safetensors
    if mode == "scratch":
        bn = timing.BeatNet()
        sd = _clone_state_dict(bn.state_dict())
    else:
        sd = _clone_state_dict(load_file(_base_hub_file("timing.safetensors", "v1.2")))
    timing_path = out_dir / "timing.safetensors"
    save_file(sd, str(timing_path))
    saved.append(timing_path)

    # 2. octave.npz
    oct_src = _base_hub_file("octave.npz", "v1.2")
    oct_dst = out_dir / "octave.npz"
    shutil.copyfile(oct_src, oct_dst)
    saved.append(oct_dst)

    # 3. rating.npz (with optional star calibration offset)
    rat_src = _base_hub_file("rating.npz", "v1.2")
    rat_dst = out_dir / "rating.npz"
    with np.load(rat_src) as z:
        arrays = {k: z[k].copy() for k in z.files}
    if star_calibration != 0.0 and "stars/baseline" in arrays:
        arrays["stars/baseline"] = arrays["stars/baseline"] + float(star_calibration)
    np.savez_compressed(rat_dst, **arrays)
    saved.append(rat_dst)

    # 4. song_stars.npz
    ss_src = _base_hub_file("song_stars.npz", "v1.2")
    ss_dst = out_dir / "song_stars.npz"
    with np.load(ss_src) as z:
        ss_arrays = {k: z[k].copy() for k in z.files}
    if star_calibration != 0.0:
        for diff_name in songstars.NAMES:
            key = f"{diff_name}/baseline"
            if key in ss_arrays:
                ss_arrays[key] = ss_arrays[key] + float(star_calibration)
    np.savez_compressed(ss_dst, **ss_arrays)
    saved.append(ss_dst)

    log(f"[Timing & Trees] Saved timing.safetensors, octave.npz, rating.npz, song_stars.npz")
    return saved


def generate_all_models(
    out_dir: Path | None = None,
    mode: str = "nebula_evolved",
    size: str = "base",
    components: list[str] | None = None,
    custom_mappers: list[str] | None = None,
    blend_alpha: float = 0.08,
    creativity_bias: float = 0.1,
    density_scale: float = 1.0,
    intensity_bias: float = 0.0,
    smoothness: float = 0.0,
    star_calibration: float = 0.0,
    map_zip_paths: list[str | Path] | None = None,
    train_steps: int = 8,
    push_to_hub: bool = True,
    repo_id: str | None = None,
    token: str | None = None,
    branch: str = "main",
    sync_legacy: bool = True,
    log: Callable[[str], None] = print,
) -> dict:
    """Generate/evolve/fine-tune the NebulaMapper model suite and optionally push directly to Hugging Face."""
    cfg = hub.load_config()
    repo_id = repo_id or cfg.get("hf_repo") or hub.DEFAULT_REPO
    token = token or cfg.get("hf_token") or hub.DEFAULT_TOKEN
    custom_mappers = custom_mappers if custom_mappers is not None else list(DEFAULT_CUSTOM_MAPPERS)
    components = components or ["map", "rhythm", "lights", "player", "labeler", "timing"]

    temp_ctx = None
    target_dir = Path(out_dir) if out_dir is not None else (hub.ROOT / "models")
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Always write config.json and README.md
        config_json = {
            "library_name": "bsgen",
            "project": "NebulaMapper",
            "repo_id": repo_id,
            "mode": mode,
            "custom_mappers": custom_mappers,
            "note": "NebulaMapper AI Beat Saber map generator & player models",
        }
        (target_dir / "config.json").write_text(json.dumps(config_json, indent=2) + "\n", encoding="utf-8")
        (target_dir / "README.md").write_text(
            hf_sync.make_model_card(
                repo_id,
                {
                    "mode": mode,
                    "custom_mappers": custom_mappers,
                    "blend_alpha": blend_alpha,
                },
            ),
            encoding="utf-8",
        )

        if "map" in components:
            generate_map_model(
                target_dir,
                mode=mode,
                size=size,
                custom_mappers=custom_mappers,
                blend_alpha=blend_alpha,
                creativity_bias=creativity_bias,
                map_zip_paths=map_zip_paths,
                train_steps=train_steps,
                log=log,
            )
        if "rhythm" in components:
            generate_rhythm_model(target_dir, mode=mode, size=size, density_scale=density_scale, log=log)
        if "lights" in components:
            generate_lights_model(target_dir, mode=mode, size=size, intensity_bias=intensity_bias, log=log)
        if "player" in components:
            generate_player_model(target_dir, mode=mode, size=size, smoothness=smoothness, log=log)
        if "labeler" in components:
            generate_labeler_model(target_dir, mode=mode, log=log)
        if "timing" in components:
            generate_timing_and_trees(target_dir, mode=mode, star_calibration=star_calibration, log=log)

        hub_result = None
        if push_to_hub and token:
            try:
                hub_result = hf_sync.push_directory_to_hub(
                    target_dir,
                    repo_id=repo_id,
                    token=token,
                    branch=branch,
                    commit_message=f"Update NebulaMapper models ({mode}: {', '.join(components)})",
                    create_version_tag="v1.2",
                    log=log,
                )
                if sync_legacy:
                    hf_sync.sync_legacy_revisions(repo_id=repo_id, token=token, log=log)
                hub.save_config(prefer_hub_over_local=True)
            except Exception as push_err:
                log(f"[Hugging Face Hub] Saved & activated models locally in {target_dir} (Hub push skipped: {push_err})")
                hub.save_config(prefer_hub_over_local=False)
        else:
            log(f"[Local Studio] Saved & activated synthesized models in {target_dir}")
            hub.save_config(prefer_hub_over_local=False)

        return {
            "ok": True,
            "mode": mode,
            "components": components,
            "custom_mappers": custom_mappers,
            "repo_id": repo_id,
            "local_dir": str(target_dir),
            "hub_result": hub_result,
        }
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()
