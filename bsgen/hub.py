import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "nebula_config.json"

DEFAULT_REPO = "PhantomVoidTTV/NebulaMapper"
FALLBACK_REPO = "cata121/bsgen-1.0"
DEFAULT_TOKEN = ""

_FETCH_CACHE: dict[tuple, list[Path]] = {}


def load_config() -> dict:
    cfg = {
        "hf_repo": os.environ.get("NEBULAMAPPER_HF_REPO") or os.environ.get("BSGEN_HF_REPO", DEFAULT_REPO),
        "hf_token": os.environ.get("NEBULAMAPPER_HF_TOKEN") or os.environ.get("BSGEN_HF_TOKEN") or os.environ.get("HF_TOKEN") or DEFAULT_TOKEN,
        "hf_branch": os.environ.get("NEBULAMAPPER_HF_BRANCH") or os.environ.get("BSGEN_HF_BRANCH", "main"),
        "fallback_repo": FALLBACK_REPO,
        "auto_push_to_hf": True,
        "prefer_hub_over_local": True,
        "mapper_author": "NebulaMapper (PhantomVoidTTV)",
    }
    if CONFIG_PATH.exists():
        try:
            disk = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            for k, v in disk.items():
                if v is not None and v != "":
                    cfg[k] = v
        except Exception:
            pass
    if os.environ.get("NEBULAMAPPER_HF_REPO") or os.environ.get("BSGEN_HF_REPO"):
        cfg["hf_repo"] = os.environ.get("NEBULAMAPPER_HF_REPO") or os.environ["BSGEN_HF_REPO"]
    if os.environ.get("NEBULAMAPPER_HF_TOKEN") or os.environ.get("BSGEN_HF_TOKEN"):
        cfg["hf_token"] = os.environ.get("NEBULAMAPPER_HF_TOKEN") or os.environ["BSGEN_HF_TOKEN"]
    return cfg


def save_config(**updates) -> dict:
    cfg = load_config()
    cfg.update({k: v for k, v in updates.items() if v is not None})
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    global REPO
    REPO = cfg["hf_repo"]
    _FETCH_CACHE.clear()
    return cfg


_cfg = load_config()
REPO = _cfg["hf_repo"]
VERSIONS = ["1.2", "1.1", "1.0"]
NEWEST = VERSIONS[0]
LEGACY = ("1.1", "1.0")
MAP_NOTES = {
    "1.2": "NebulaMapper main / v1.2 (newest, most stable)",
    "1.1": "v1.1 (more creative, sometimes weird)",
    "1.0": "v1.0 (first release)",
}


def get_token() -> str | None:
    return load_config().get("hf_token") or None


def get_repo() -> str:
    return load_config().get("hf_repo") or DEFAULT_REPO


def fetch(*names, version=NEWEST, force_download=False, repo=None):
    folder = os.environ.get("NEBULAMAPPER_MODELS") or os.environ.get("BSGEN_MODELS")
    if folder and version == NEWEST and not force_download:
        paths = [Path(folder) / n for n in names]
        if all(p.exists() for p in paths):
            return paths

    cfg = load_config()
    target_repo = repo or cfg.get("hf_repo") or REPO
    token = cfg.get("hf_token") or None
    branch = cfg.get("hf_branch") or "main"
    fallback = cfg.get("fallback_repo") or FALLBACK_REPO

    cache_key = (target_repo, branch, str(version), tuple(names))
    if not force_download and cache_key in _FETCH_CACHE:
        cached_paths = _FETCH_CACHE[cache_key]
        if all(p.exists() for p in cached_paths):
            return cached_paths

    from huggingface_hub import hf_hub_download

    tok_list = [token, None] if token else [None]
    candidates = []
    if version in (None, NEWEST, "main"):
        for t in tok_list:
            candidates += [
                (target_repo, branch, t),
                (target_repo, f"v{NEWEST}", t),
                (fallback, f"v{NEWEST}", t),
            ]
    else:
        rev = version if str(version).startswith("v") else f"v{version}"
        for t in tok_list:
            candidates += [
                (target_repo, rev, t),
                (target_repo, branch, t),
                (fallback, rev, t),
            ]

    # Try local HF disk cache first (zero network latency) when not forcing download
    if not force_download:
        for r_id, rev, tok in candidates:
            try:
                out = [
                    Path(hf_hub_download(r_id, n, revision=rev, token=tok, local_files_only=True))
                    for n in names
                ]
                _FETCH_CACHE[cache_key] = out
                return out
            except Exception:
                continue

    last_err = None
    for r_id, rev, tok in candidates:
        try:
            out = [
                Path(hf_hub_download(r_id, n, revision=rev, token=tok, force_download=force_download))
                for n in names
            ]
            _FETCH_CACHE[cache_key] = out
            return out
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not fetch {names} from {target_repo} (or fallback {fallback}): {last_err}")


LOCAL_DIR = ROOT / "models"
LOCAL_MAP = LOCAL_DIR / "map.pt"
LOCAL_MAP_ST = LOCAL_DIR / "map.safetensors"


def _prefer_local(path: Path) -> bool:
    if not path.exists():
        return False
    return not load_config().get("prefer_hub_over_local", True)


def map_model(version=None, force_download=False):
    if version in (None, NEWEST) and not force_download:
        if _prefer_local(LOCAL_MAP_ST) and (LOCAL_DIR / "map.json").exists():
            return LOCAL_MAP_ST
        if _prefer_local(LOCAL_MAP):
            return LOCAL_MAP
    return fetch("map.safetensors", "map.json", version=version or NEWEST, force_download=force_download)[0]


LOCAL_PLAYER = LOCAL_MAP.with_name("player.pt")
LOCAL_PLAYER_ST = LOCAL_DIR / "player.safetensors"


def player_model(force_download=False):
    if not force_download:
        if _prefer_local(LOCAL_PLAYER_ST) and (LOCAL_DIR / "player.json").exists():
            return LOCAL_PLAYER_ST
        if _prefer_local(LOCAL_PLAYER):
            return LOCAL_PLAYER
    return fetch("player.safetensors", "player.json", force_download=force_download)[0]


LOCAL_LABELER = LOCAL_MAP.with_name("labeler.pt")
LOCAL_LABELER_ST = LOCAL_DIR / "labeler.safetensors"


def labeler_model(force_download=False):
    if not force_download:
        if _prefer_local(LOCAL_LABELER_ST):
            return LOCAL_LABELER_ST
        if _prefer_local(LOCAL_LABELER):
            return LOCAL_LABELER
    return fetch("labeler.safetensors", force_download=force_download)[0]


LOCAL_LIGHTS = LOCAL_MAP.with_name("lights.pt")
LOCAL_LIGHTS_ST = LOCAL_DIR / "lights.safetensors"


def light_model(force_download=False):
    if not force_download:
        if _prefer_local(LOCAL_LIGHTS_ST) and (LOCAL_DIR / "lights.json").exists():
            return LOCAL_LIGHTS_ST
        if _prefer_local(LOCAL_LIGHTS):
            return LOCAL_LIGHTS
    return fetch("lights.safetensors", "lights.json", force_download=force_download)[0]


LOCAL_RHYTHM = LOCAL_MAP.with_name("rhythm.pt")
LOCAL_RHYTHM_ST = LOCAL_DIR / "rhythm.safetensors"


def rhythm_model(force_download=False):
    if not force_download:
        if _prefer_local(LOCAL_RHYTHM_ST) and (LOCAL_DIR / "rhythm.json").exists():
            return LOCAL_RHYTHM_ST
        if _prefer_local(LOCAL_RHYTHM):
            return LOCAL_RHYTHM
    return fetch("rhythm.safetensors", "rhythm.json", force_download=force_download)[0]


LOCAL_RATING = LOCAL_MAP.with_name("rating.npz")


def rating_model(force_download=False):
    if not force_download and _prefer_local(LOCAL_RATING):
        return LOCAL_RATING
    return fetch("rating.npz", force_download=force_download)[0]


LOCAL_SONG_STARS = LOCAL_MAP.with_name("song_stars.npz")


def song_stars_model(force_download=False):
    if not force_download and _prefer_local(LOCAL_SONG_STARS):
        return LOCAL_SONG_STARS
    return fetch("song_stars.npz", force_download=force_download)[0]


def timing_model(force_download=False):
    local = ROOT / "runs" / "timing"
    if not force_download and (local / "timing.safetensors").exists() and not load_config().get("prefer_hub_over_local", True):
        return local
    if not force_download and _prefer_local(LOCAL_DIR / "timing.safetensors") and (LOCAL_DIR / "octave.npz").exists():
        return LOCAL_DIR
    return fetch("timing.safetensors", "octave.npz", force_download=force_download)[0].parent
