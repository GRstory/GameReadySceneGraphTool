# -*- coding: utf-8 -*-
"""glb/gltf -> usda 변환 (태깅 입력용).

/<에셋명> Xform 아래에 glTF 노드마다 Mesh 하나. 월드 변환은 점에 굽는다.
baseColor 텍스처는 usda 옆 <에셋명>_tex/ 에 png로 저장해 UsdPreviewSurface로 연결한다.
"""
import os
import re

import numpy as np
import trimesh
from pxr import Sdf, Usd, UsdGeom, UsdShade


def _ident(s):
    s = re.sub(r"[^0-9A-Za-z_]", "_", s)
    return s if s and not s[0].isdigit() else "_" + s


def glb_to_usda(src, dst):
    name = _ident(os.path.splitext(os.path.basename(src))[0])
    scene = trimesh.load(src, force="scene")
    tex_dir = os.path.join(os.path.dirname(dst), name + "_tex")
    os.makedirs(tex_dir, exist_ok=True)

    stage = Usd.Stage.CreateNew(dst)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)    # glTF는 Y-up, 미터
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/" + name)
    stage.SetDefaultPrim(root.GetPrim())

    mats, used = {}, set()
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        pname = _ident(node)
        while pname in used:
            pname += "_"
        used.add(pname)
        pts = trimesh.transform_points(g.vertices, T)
        m = UsdGeom.Mesh.Define(stage, f"/{name}/{pname}")
        m.CreatePointsAttr([tuple(p) for p in pts])
        m.CreateFaceVertexCountsAttr([3] * len(g.faces))
        m.CreateFaceVertexIndicesAttr(g.faces.flatten().tolist())
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        uv = getattr(g.visual, "uv", None)
        if uv is not None and len(uv) == len(g.vertices):
            UsdGeom.PrimvarsAPI(m).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
            ).Set([tuple(t) for t in np.asarray(uv)])
        mat = getattr(g.visual, "material", None)
        if mat is not None:
            if id(mat) not in mats:
                mats[id(mat)] = _material(stage, name, mat, tex_dir, len(mats))
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mats[id(mat)])
    stage.GetRootLayer().Save()
    return dst


def _material(stage, root, mat, tex_dir, i):
    """trimesh 머티리얼 -> UsdPreviewSurface (baseColor 텍스처 또는 상수색)."""
    path = f"/{root}/Looks/mat_{i}"
    m = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is not None:
        f = os.path.join(tex_dir, f"mat_{i}_basecolor.png")
        img.convert("RGB").save(f)
        tx = UsdShade.Shader.Define(stage, path + "/basecolor")
        tx.CreateIdAttr("UsdUVTexture")
        rel = os.path.relpath(f, os.path.dirname(stage.GetRootLayer().realPath)).replace("\\", "/")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./" + rel)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tx.ConnectableAPI(), "rgb")
    else:
        c = getattr(mat, "baseColorFactor", None)
        c = np.asarray(c if c is not None else (180, 180, 180, 255), dtype=np.float64)[:3]
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            tuple(c / 255.0 if c.max() > 1 else c))
    return m
