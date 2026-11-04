import json
import time
from pathlib import Path
from typing import Callable

from . import hub

ALL_MODEL_FILES = (
    "config.json",
    "map.safetensors",
    "map.json",
    "rhythm.safetensors",
    "rhythm.json",
    "lights.safetensors",
    "lights.json",
    "player.safetensors",
    "player.json",
    "labeler.safetensors",
    "timing.safetensors",
    "octave.npz",
    "rating.npz",
    "song_stars.npz",
)


def make_model_card(repo_id: str, metadata: dict | None = None) -> str:
    meta = metadata or {}
    mode = meta.get("mode", "nebula_evolved")
    custom_mappers = meta.get("custom_mappers", ["phantomvoid", "nebulamapper", "nebula_tech", "nebula_speed", "nebula_acc"])
    blend_alpha = meta.get("blend_alpha", 0.0)
    timestamp = meta.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    mappers_md = ", ".join(f"`{m}`" for m in custom_mappers) if custom_mappers else "`any`"

    return f"""---
library_name: bsgen
license: agpl-3.0
pipeline_tag: audio-to-audio
tags:
  - beat-saber
  - rhythm-game
  - beatmap-generation
  - music-information-retrieval
  - pytorch
  - safetensors
  - nebulamapper
---

# 🌌 {repo_id.split('/')[-1]} — AI Beat Saber Map & Replay Models

Official model repository for **NebulaMapper** (`{repo_id}`), an enhanced AI Beat Saber map generator, lighting choreographer, and autonomous VR player bot.

## 📦 Included Models in `{repo_id}`

| Model Component | Weights / Artifact | Config | Description |
| :--- | :--- | :--- | :--- |
| **Map Model** | `map.safetensors` | `map.json` | 65M+ parameter Encoder-Decoder Transformer with audio overview & musical memory alignment |
| **Rhythm Model** | `rhythm.safetensors` | `rhythm.json` | 47M+ parameter RoPE Transformer predicting exact onset hit ticks per beat |
| **Lighting Model** | `lights.safetensors` | `lights.json` | 7M+ parameter Iterative Masked Diffusion / Transformer for 23 Beat Saber environments |
| **Player Bot Model** | `player.safetensors` | `player.json` | 38M+ parameter Flow-Matching 6-DoF VR headset & dual-saber trajectory generator |
| **Swing Labeler** | `labeler.safetensors` | — | Bidirectional Transformer for forehand/backhand parity & reset detection |
| **Timing BeatNet** | `timing.safetensors` | `octave.npz` | Audio log-mel beat activation network + octave random forest classifier |
| **Star Estimator** | `rating.npz` | — | Gradient boosted decision forest estimating Star ★, Tech, and Pass ratings |
| **Song Stars** | `song_stars.npz` | — | Audio-to-difficulty star rating and target HPS/Tech/Pass estimator |

## ✨ Custom Build Metadata

- **Generation Mode**: `{mode}`
- **Custom Mapper Personas**: {mappers_md}
- **Checkpoint Blend Ratio**: `{blend_alpha}`
- **Last Updated**: `{timestamp}`

## 🚀 Quick Usage

```bash
# Generate a Beat Saber map automatically using {repo_id}
python scripts/generate.py --audio song.mp3 --stars 7.5 10.0 --mapper phantomvoid --replay

# Launch the NebulaMapper Web UI (automatically pulls & pushes models to {repo_id})
python scripts/webui.py

# Generate / Fine-tune / Evolve models and auto-upload to {repo_id}
python scripts/generate_models.py --mode nebula_evolved --push
```
"""


def get_repo_status(repo_id: str | None = None, token: str | None = None) -> dict:
    from huggingface_hub import HfApi

    cfg = hub.load_config()
    repo_id = repo_id or cfg.get("hf_repo") or hub.DEFAULT_REPO
    token = (token or cfg.get("hf_token") or hub.DEFAULT_TOKEN or "").strip() or None
    for tok in ([token, None] if token else [None]):
        api = HfApi(token=tok)
        try:
            info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
            refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")
            files = {}
            for sib in info.siblings or []:
                files[sib.rfilename] = {
                    "size": sib.size,
                    "blob_id": getattr(sib, "blob_id", None),
                }
            branches = [b.name for b in refs.branches]
            tags = [t.name for t in refs.tags]
            missing = [f for f in ALL_MODEL_FILES if f not in files]
            return {
                "ok": True,
                "repo_id": repo_id,
                "url": f"https://huggingface.co/{repo_id}/tree/main",
                "branches": branches,
                "tags": tags,
                "files": files,
                "missing": missing,
                "complete": len(missing) == 0,
            }
        except Exception as e:
            last_err = e
            continue
    return {
        "ok": False,
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "error": str(last_err),
        "files": {},
        "missing": list(ALL_MODEL_FILES),
        "complete": False,
    }


REQUIRED_FILES = ALL_MODEL_FILES
check_repo_status = get_repo_status


def push_directory_to_hub(
    folder: Path,
    repo_id: str | None = None,
    token: str | None = None,
    branch: str = "main",
    commit_message: str = "Upload NebulaMapper AI models",
    create_version_tag: str | None = "v1.2",
    log: Callable[[str], None] = print,
) -> dict:
    from huggingface_hub import CommitOperationAdd, HfApi

    cfg = hub.load_config()
    repo_id = repo_id or cfg.get("hf_repo") or hub.DEFAULT_REPO
    token = token or cfg.get("hf_token") or hub.DEFAULT_TOKEN
    folder = Path(folder)

    api = HfApi(token=token)
    log(f"Ensuring Hugging Face model repository '{repo_id}' exists...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

    operations = []
    uploaded_files = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            operations.append(
                CommitOperationAdd(
                    path_in_repo=p.name,
                    path_or_fileobj=str(p),
                )
            )
            uploaded_files.append(p.name)

    if not operations:
        raise ValueError(f"No model files found in {folder} to upload.")

    log(f"Uploading {len(operations)} files ({', '.join(uploaded_files)}) to {repo_id}@{branch}...")
    commit_info = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=commit_message,
        revision=branch,
    )
    log(f"Successfully committed to {repo_id}@{branch}: {commit_info.commit_url}")

    if create_version_tag:
        tag_name = create_version_tag if create_version_tag.startswith("v") else f"v{create_version_tag}"
        try:
            api.create_tag(
                repo_id=repo_id,
                repo_type="model",
                tag=tag_name,
                revision=branch,
                tag_message=f"NebulaMapper release {tag_name}",
                exist_ok=True,
            )
            log(f"Updated tag '{tag_name}' on {repo_id}")
        except Exception as e:
            log(f"Note: could not update tag {tag_name}: {e}")

    return {
        "repo_id": repo_id,
        "branch": branch,
        "commit_url": getattr(commit_info, "commit_url", f"https://huggingface.co/{repo_id}/tree/{branch}"),
        "files": uploaded_files,
    }


def sync_legacy_revisions(
    repo_id: str | None = None,
    token: str | None = None,
    source_repo: str = hub.FALLBACK_REPO,
    log: Callable[[str], None] = print,
) -> None:
    """Ensure v1.1 and v1.0 revisions also exist on the user's Hugging Face repo so every model version works."""
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    cfg = hub.load_config()
    repo_id = repo_id or cfg.get("hf_repo") or hub.DEFAULT_REPO
    token = token or cfg.get("hf_token") or hub.DEFAULT_TOKEN
    api = HfApi(token=token)

    refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")
    existing_tags = {t.name for t in refs.tags}
    existing_branches = {b.name for b in refs.branches}

    legacy_specs = {
        "v1.1": ["map.json", "map.safetensors", "octave.npz", "player.json", "player.safetensors", "timing.safetensors"],
        "v1.0": ["map.json", "map.safetensors", "octave.npz", "timing.safetensors"],
    }
    for rev, filenames in legacy_specs.items():
        if rev in existing_tags or rev in existing_branches:
            log(f"Revision {rev} already exists on {repo_id}.")
            continue
        log(f"Syncing legacy revision {rev} to {repo_id}...")
        try:
            api.create_branch(repo_id=repo_id, repo_type="model", branch=rev, exist_ok=True)
            ops = []
            for fn in filenames:
                local_path = hf_hub_download(source_repo, fn, revision=rev, token=token)
                ops.append(CommitOperationAdd(path_in_repo=fn, path_or_fileobj=local_path))
            api.create_commit(
                repo_id=repo_id,
                repo_type="model",
                operations=ops,
                commit_message=f"Sync {rev} legacy models to {repo_id}",
                revision=rev,
            )
            api.create_tag(repo_id=repo_id, repo_type="model", tag=rev, revision=rev, exist_ok=True)
            log(f"Synced {rev} to {repo_id}!")
        except Exception as e:
            log(f"Warning syncing {rev}: {e}")
