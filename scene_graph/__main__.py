# -*- coding: utf-8 -*-
"""
python -m scene_graph build <scene.usda> [-o out_dir]
python -m scene_graph add   <scene.usda> <subject_path> <predicate> <object_path> [-o out_dir]
python -m scene_graph list  <scene.usda> [-o out_dir]
"""
import argparse

from .layer import PREDICATES, add_manual, build, check_manual, describe, open_sg, sg_path


def _print(stage):
    for src, s, p, o in describe(stage):
        print(f"  [{src}] {s}  --{p}-->  {o}")


def main():
    ap = argparse.ArgumentParser(prog="scene_graph")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("build", "add", "list"):
        sp = sub.add_parser(c)
        sp.add_argument("scene")
        sp.add_argument("-o", "--out", default="scene_graph_out")
        if c == "add":
            sp.add_argument("subject")
            sp.add_argument("predicate", choices=PREDICATES)
            sp.add_argument("object")
    a = ap.parse_args()

    if a.cmd == "build":
        stage, rels, warn = build(a.scene, a.out)
        print(f"-> {sg_path(a.scene, a.out)}  ({len(rels)} auto)")
    elif a.cmd == "add":
        add_manual(a.scene, a.out, a.subject, a.predicate, a.object)
        stage = open_sg(a.scene, a.out)
        warn = check_manual(stage)
    else:
        stage = open_sg(a.scene, a.out)
        warn = check_manual(stage)
    _print(stage)
    for e, r, t in warn:
        print(f"  WARNING: {e} {r} -> {t} (missing prim)")


if __name__ == "__main__":
    main()
