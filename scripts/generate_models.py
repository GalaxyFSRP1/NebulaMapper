import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bsgen import hf_sync, hub, modelgen


def main():
    ap = argparse.ArgumentParser(description="Generate / evolve / fine-tune NebulaMapper models and push to Hugging Face")
    ap.add_argument("--mode", choices=["nebula_evolved", "blend", "finetune", "scratch"], default="nebula_evolved",
                    help="Model generation mode (default: nebula_evolved)")
    ap.add_argument("--size", choices=["nano", "small", "base"], default="base",
                    help="Architecture size when --mode=scratch")
    ap.add_argument("--components", nargs="+",
                    choices=["map", "rhythm", "lights", "player", "labeler", "timing"],
                    default=["map", "rhythm", "lights", "player", "labeler", "timing"],
                    help="Which model components to generate/update")
    ap.add_argument("--custom-mappers", nargs="+", default=modelgen.DEFAULT_CUSTOM_MAPPERS,
                    help="Custom mapper persona names to synthesize into map.safetensors / map.json")
    ap.add_argument("--blend-alpha", type=float, default=0.08,
                    help="Checkpoint blend ratio (0.0 to 1.0) for nebula_evolved / blend modes")
    ap.add_argument("--creativity", type=float, default=0.1,
                    help="Creativity bias for custom mapper persona embeddings")
    ap.add_argument("--density-scale", type=float, default=1.0,
                    help="Rhythm model HPS density multiplier")
    ap.add_argument("--intensity-bias", type=float, default=0.0,
                    help="Lighting model intensity bias")
    ap.add_argument("--smoothness", type=float, default=0.0,
                    help="Player bot motion trajectory smoothness bias")
    ap.add_argument("--star-calibration", type=float, default=0.0,
                    help="Star rating estimator calibration offset")
    ap.add_argument("--finetune-maps", nargs="*", help="Beat Saber .zip map files for --mode=finetune")
    ap.add_argument("--train-steps", type=int, default=8, help="Fine-tuning steps for --mode=finetune")
    ap.add_argument("--repo", default=None, help="Hugging Face repo ID (default: PhantomVoidTTV/NebulaMapper)")
    ap.add_argument("--token", default=None, help="Hugging Face write token")
    ap.add_argument("--branch", default="main", help="Hugging Face branch (default: main)")
    ap.add_argument("--out-dir", default=None, help="Optional local folder to keep generated model files")
    ap.add_argument("--no-push", action="store_true", help="Do not upload to Hugging Face")
    ap.add_argument("--no-legacy-sync", action="store_true", help="Skip syncing v1.1 and v1.0 legacy branches")
    ap.add_argument("--status", action="store_true", help="Only print current Hugging Face repo status")
    args = ap.parse_args()

    if args.repo or args.token:
        hub.save_config(hf_repo=args.repo, hf_token=args.token)

    if args.status:
        st = hf_sync.get_repo_status(args.repo, args.token)
        print(f"Repo: {st['repo_id']} ({st['url']})")
        print(f"Status OK: {st['ok']} | Complete: {st['complete']}")
        print(f"Branches: {st.get('branches', [])} | Tags: {st.get('tags', [])}")
        print(f"Files ({len(st['files'])}): {list(st['files'].keys())}")
        if st["missing"]:
            print(f"Missing: {st['missing']}")
        return

    res = modelgen.generate_all_models(
        out_dir=Path(args.out_dir) if args.out_dir else None,
        mode=args.mode,
        size=args.size,
        components=args.components,
        custom_mappers=args.custom_mappers,
        blend_alpha=args.blend_alpha,
        creativity_bias=args.creativity,
        density_scale=args.density_scale,
        intensity_bias=args.intensity_bias,
        smoothness=args.smoothness,
        star_calibration=args.star_calibration,
        map_zip_paths=args.finetune_maps,
        train_steps=args.train_steps,
        push_to_hub=not args.no_push,
        repo_id=args.repo,
        token=args.token,
        branch=args.branch,
        sync_legacy=not args.no_legacy_sync,
        log=lambda m: print(m, flush=True),
    )
    print("\n=== Done! ===")
    if res.get("hub_result"):
        print(f"Live on Hugging Face: https://huggingface.co/{res['repo_id']}/tree/{args.branch}")


if __name__ == "__main__":
    main()
