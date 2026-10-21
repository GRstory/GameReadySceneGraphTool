# -*- coding: utf-8 -*-
"""
직접 부모-자식 프림 쌍의 공간 관계 판정 (on / inside / attachedTo / locatedIn).

"물체" = 자기 지오메트리를 가진 프림. 자기 지오메트리는 자신이 Mesh이거나 바로 아래 Mesh 자식.
박스는 부모 로컬 좌표계 기준: 부모 박스 = 부모 자기 메시의 AABB, 자식 박스 = 자식 서브트리 점들을
부모 좌표계로 옮긴 AABB. 부모가 회전해 있어도 부모 기준으로 판정된다(= 월드 OBB).
"""
import numpy as np
from pxr import Gf, Usd, UsdGeom

from keyword_tagger.tagger import _meshes_under, _skip

REGION_ATTR = "gameReady:sg:region"

INSIDE_FRAC = 0.9     # 자식 박스 부피 중 부모 박스 안 비율
ON_SINK = 0.3         # on: 자식 높이 대비 허용 파묻힘
ON_GAP = 0.02         # on/attachedTo: 부모 박스 대각선 대비 허용 간격
ON_OVERLAP = 0.5      # on: 자식 바닥 면적 중 부모 윗면과 겹치는 비율
UP_DOT = 0.9          # on: 부모 로컬 up과 월드 up 정렬 최소 내적


def own_meshes(prim):
    if prim.IsA(UsdGeom.Mesh):
        return [prim]
    return [c for c in prim.GetChildren() if c.IsA(UsdGeom.Mesh) and not _skip(c)]


def is_region(prim):
    a = prim.GetAttribute(REGION_ATTR)
    return bool(a and a.Get())


def _points(meshes, xcache, to_frame):
    pts = []
    for m in meshes:
        p = UsdGeom.Mesh(m).GetPointsAttr().Get()
        if not p:
            continue
        M = xcache.GetLocalToWorldTransform(m) * to_frame
        pts.append(np.array([M.Transform(Gf.Vec3d(*v)) for v in p]))
    return np.concatenate(pts) if pts else None


def _box(pts):
    return pts.min(0), pts.max(0)


def _vol(lo, hi):
    return float(np.prod(np.maximum(hi - lo, 0)))


def classify(parent, child, xcache, up):
    """부모 기준 자식의 관계 술어 하나 또는 None. up = 축 인덱스(0/1/2)."""
    P = xcache.GetLocalToWorldTransform(parent)
    to_parent = P.GetInverse()
    cpts = _points(_meshes_under(child), xcache, to_parent)
    if cpts is None:
        return None
    clo, chi = _box(cpts)

    if is_region(parent):
        rpts = _points(_meshes_under(parent), xcache, to_parent)
        if rpts is None:
            return None
        rlo, rhi = _box(rpts)
        c = (clo + chi) / 2
        return "locatedIn" if np.all(c >= rlo) and np.all(c <= rhi) else None

    ppts = _points(own_meshes(parent), xcache, to_parent)
    if ppts is None:
        return None
    plo, phi = _box(ppts)
    tol = ON_GAP * float(np.linalg.norm(phi - plo))

    # inside
    ilo, ihi = np.maximum(clo, plo), np.minimum(chi, phi)
    cvol = _vol(clo, chi)
    if cvol > 0 and _vol(ilo, ihi) / cvol >= INSIDE_FRAC:
        return "inside"

    # on: 자식 바닥이 부모 윗면 근처, 수평으로 충분히 겹침, 부모가 똑바로 서 있음
    h = [i for i in range(3) if i != up]
    ch = chi[up] - clo[up]
    gap = clo[up] - phi[up]
    foot = np.prod(chi[h] - clo[h])
    over = np.prod(np.maximum(np.minimum(chi[h], phi[h]) - np.maximum(clo[h], plo[h]), 0))
    upw = np.zeros(3)
    upw[up] = 1
    wup = P.TransformDir(Gf.Vec3d(*upw))
    upright = Gf.Dot(wup.GetNormalized(), Gf.Vec3d(*upw)) >= UP_DOT
    if upright and -ON_SINK * ch <= gap <= tol and foot > 0 and over / foot >= ON_OVERLAP:
        return "on"

    # attachedTo: 두 박스가 닿거나 겹침
    sep = np.maximum(np.maximum(plo - chi, clo - phi), 0)
    if np.all(sep <= tol):
        return "attachedTo"
    return None


def auto_relations(stage):
    """[(subject_prim, predicate, object_prim)] — 직접 부모-자식 쌍만."""
    up = 2 if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z else 1
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = []
    it = iter(Usd.PrimRange(stage.GetPseudoRoot()))
    for prim in it:
        if _skip(prim):
            it.PruneChildren()
            continue
        if prim.IsPseudoRoot() or not prim.IsA(UsdGeom.Xformable) or prim.IsA(UsdGeom.Mesh):
            continue
        parent = prim.GetParent()
        if parent.IsPseudoRoot() or not own_meshes(prim) and not _meshes_under(prim):
            continue
        if not (is_region(parent) or own_meshes(parent)):
            continue      # 순수 그룹 Xform
        pred = classify(parent, prim, xcache, up)
        if pred:
            out.append((prim, pred, parent))
    return out
