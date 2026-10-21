# -*- coding: utf-8 -*-
"""python -m keyword_tagger <input.usd|usda|usdc|usdz|dir> ... [-o out_dir] [--model gemma4:e4b]"""
import argparse
import glob
import os

from .tagger import process_file

EXTS = (".usd", ".usda", ".usdc", ".usdz", ".glb", ".gltf")


def main():
    ap = argparse.ArgumentParser(prog="keyword_tagger")
    ap.add_argument("inputs", nargs="+", help="USD files or directories")
    ap.add_argument("-o", "--out", default="keyword_tagger_out", help="output dir")
    ap.add_argument("--model", default="gemma4:e4b",
                    help="Ollama model, or claude-* (e.g. claude-opus-5-5) for the Claude API")
    a = ap.parse_args()
    files = []
    for p in a.inputs:
        if os.path.isdir(p):
            files += sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(EXTS))
        else:
            files.append(p)
    for f in files:
        process_file(f, a.out, a.model)


if __name__ == "__main__":
    main()
