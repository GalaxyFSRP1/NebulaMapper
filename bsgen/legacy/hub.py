from pathlib import Path
from .. import hub as main_hub

REPO = main_hub.REPO


def fetch(*names, version="1.1"):
    return main_hub.fetch(*names, version=version)


def map_model(version="1.1"):
    return fetch("map.safetensors", "map.json", version=version)[0]


def timing_model():
    return fetch("timing.safetensors", "octave.npz")[0].parent
