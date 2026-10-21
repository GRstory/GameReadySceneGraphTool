# -*- coding: utf-8 -*-
"""
<이름>_scenegraph.usda 레이어: 원본을 sublayer로 걸고 /SceneGraph/Relations 아래에 엣지 프림을 둔다.

엣지 프림 = <subj>__<pred>__<obj>
    uniform token gameReady:sg:predicate
    rel gameReady:sg:subject / gameReady:sg:object
    uniform token gameReady:sg:source  (auto | manual)
다시 빌드하면 auto 엣지만 지우고 새로 만든다. manual 엣지는 두고, 대상이 없으면 경고만 낸다.
"""
import os

from pxr import Sdf, Usd

from keyword_tagger.glb import _ident

from .relations import auto_relations

ROOT = "/SceneGraph/Relations"
PRED = "gameReady:sg:predicate"
SUBJ = "gameReady:sg:subject"
OBJ = "gameReady:sg:object"
SRC = "gameReady:sg:source"
PREDICATES = ("on", "inside", "attachedTo", "locatedIn", "activates")


def sg_path(src, out_dir):
    name = os.path.splitext(os.path.basename(src))[0]
    return os.path.join(out_dir, name + "_scenegraph.usda")


def open_sg(src, out_dir):
    """scenegraph 레이어 스테이지를 열거나 새로 만든다 (편집 대상 = scenegraph 레이어)."""
    dst = sg_path(src, out_dir)
    if os.path.exists(dst):
        return Usd.Stage.Open(dst)
    os.makedirs(out_dir, exist_ok=True)
    layer = Sdf.Layer.CreateNew(dst)
    rel = os.path.relpath(os.path.abspath(src), os.path.dirname(os.path.abspath(dst))).replace("\\", "/")
    layer.subLayerPaths.append(rel)
    stage = Usd.Stage.Open(layer)
    base = Sdf.Layer.FindOrOpen(src)
    if base.defaultPrim:
        layer.defaultPrim = base.defaultPrim
    for k in ("upAxis", "metersPerUnit"):
        if base.pseudoRoot.HasInfo(k):
            layer.pseudoRoot.SetInfo(k, base.pseudoRoot.GetInfo(k))
    return stage


def edges(stage):
    root = stage.GetPrimAtPath(ROOT)
    return list(root.GetChildren()) if root else []


def _targets(edge, name):
    t = edge.GetRelationship(name).GetTargets()
    return t[0] if t else None


def add_edge(stage, subj, pred, obj, source):
    assert pred in PREDICATES, pred
    stage.DefinePrim("/SceneGraph", "Scope")
    stage.DefinePrim(ROOT, "Scope")
    base = f"{_ident(subj.name)}__{pred}__{_ident(obj.name)}"
    name, i = base, 1
    while stage.GetPrimAtPath(f"{ROOT}/{name}"):
        name, i = f"{base}_{i}", i + 1
    e = stage.DefinePrim(f"{ROOT}/{name}")
    e.CreateAttribute(PRED, Sdf.ValueTypeNames.Token, variability=Sdf.VariabilityUniform).Set(pred)
    e.CreateRelationship(SUBJ).SetTargets([subj])
    e.CreateRelationship(OBJ).SetTargets([obj])
    e.CreateAttribute(SRC, Sdf.ValueTypeNames.Token, variability=Sdf.VariabilityUniform).Set(source)
    return e


def build(src, out_dir):
    stage = open_sg(src, out_dir)
    for e in edges(stage):
        if e.GetAttribute(SRC).Get() == "auto":
            stage.RemovePrim(e.GetPath())
    rels = auto_relations(stage)
    for s, p, o in rels:
        add_edge(stage, s.GetPath(), p, o.GetPath(), "auto")
    warn = check_manual(stage)
    stage.GetEditTarget().GetLayer().Save()
    return stage, rels, warn


def check_manual(stage):
    """대상 프림이 없는 manual 엣지 목록."""
    bad = []
    for e in edges(stage):
        if e.GetAttribute(SRC).Get() != "manual":
            continue
        for r in (SUBJ, OBJ):
            t = _targets(e, r)
            if t is None or not stage.GetPrimAtPath(t):
                bad.append((e.GetPath(), r, t))
    return bad


def add_manual(src, out_dir, subj, pred, obj):
    stage = open_sg(src, out_dir)
    for p in (subj, obj):
        if not stage.GetPrimAtPath(p):
            raise SystemExit(f"prim not found: {p}")
    e = add_edge(stage, Sdf.Path(subj), pred, Sdf.Path(obj), "manual")
    stage.GetEditTarget().GetLayer().Save()
    return e


def describe(stage):
    for e in edges(stage):
        s, o = _targets(e, SUBJ), _targets(e, OBJ)
        yield e.GetAttribute(SRC).Get(), s, e.GetAttribute(PRED).Get(), o
