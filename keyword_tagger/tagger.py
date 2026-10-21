# -*- coding: utf-8 -*-
"""
GameReady USD 프림별 검색 키워드 태거.

프림 지오메트리 오프스크린 렌더(4뷰, 텍스처 반영) -> 로컬 VLM(Ollama gemma4:e4b) -> 정규화
-> `string[] gameReady:label:keywords` 저작. 부모 프림의 키워드를 자식 프롬프트에 힌트로 준다.
색상 단어는 키워드에서 제외한다.
"""
import base64
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import time
import urllib.request
import zipfile

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from pxr import Sdf, Usd, UsdGeom, UsdShade, UsdUtils

ATTR = "gameReady:label:keywords"
N_KEYWORDS = 5

STOP = {"object", "item", "thing", "prop", "model", "mesh", "asset", "3d", "render",
        # gemma3:4b가 자주 내는 3D 잡음어 (계획 외 추가)
        "geometry", "geometric", "polygon", "low poly", "segment", "shape", "fragment", "piece"}
COLORS = {"red", "green", "blue", "yellow", "white", "black", "gray", "grey", "brown",
          "orange", "purple", "pink", "silver", "gold", "golden", "beige", "tan",
          "cyan", "magenta", "violet", "teal", "navy", "maroon", "turquoise", "ivory",
          "cream", "crimson", "scarlet", "olive", "khaki", "amber", "lavender",
          "indigo", "colored", "colorful", "multicolored"}

SCHEMA = {"type": "object",
          "properties": {"name": {"type": "string"},
                         "keywords": {"type": "array", "items": {"type": "string"}}},
          "required": ["name", "keywords"]}

PROMPT = """{images}
Hints: prim name "{name}", parent prim name "{parent}". Names may be meaningful or junk like "mesh_0".{context}

"name": the most specific English name of this object, may be several words (e.g. "rifle bolt").
"keywords": exactly 5 English keywords a person would type to find this object in a game engine asset search.
Every keyword is ONE single word, no spaces. Split compound names into separate words
(e.g. "rifle bolt" -> "rifle", "bolt"; "treasure chest" -> "treasure", "chest").
Keyword order:
1) the words of the object name
2) one or two broader categories a searcher would accept (e.g. apple -> fruit, food). Never overly abstract ones like plant, organism, structure.
3) a visible material or shape
4) a notable state only if clearly visible (broken, rusty, open)
Rules: no color words. lowercase, singular nouns/adjectives.
Never use: object, item, thing, prop, model, mesh, asset, 3d, render, geometry, polygon, low poly, segment, shape, fragment, piece.
Never copy a junk name hint (like "segment_3", "mesh_0", "MATCUTResult") as a keyword.
If the part alone is ambiguous, use the hints.
Reply as JSON: {{"name": "...", "keywords": [...]}}"""

IMAGES_ALONE = "These 4 images show one part of a 3D game asset from 4 angles."
IMAGES_CTX = ("Images 1-4 show one part of a 3D game asset from 4 angles. "
              "Images 5-6 show the whole object from front and back: the part is drawn in its real colors, "
              "everything else is white. Use images 5-6 to see where the part sits in the whole object.")

# 조상 프림에 키워드가 있으면 붙이는 문맥 힌트
PART_CONTEXT = """
This is one part cut out of a larger object: a {main}.
Look at the shape in the images and decide which specific part of the {main} it is.
"name" = the part name. Keywords = the words of the part name, the words of "{main}", then a visible material or shape."""


# ---------- 프림 선택 ----------

def _skip(prim):
    """머티리얼 서브트리 / guide·proxy / 비가시 프림이면 True."""
    if prim.GetTypeName() in ("Material", "Shader", "NodeGraph") or prim.GetName() in ("Looks", "Materials"):
        return True
    if prim.IsA(UsdGeom.Imageable):
        img = UsdGeom.Imageable(prim)
        if img.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            return True
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible:
            return True
    return False


def _meshes_under(prim):
    """prim 이하(자신 포함)의 렌더 대상 Mesh 프림."""
    out = []
    it = iter(Usd.PrimRange(prim))
    for p in it:
        if _skip(p):
            it.PruneChildren()
            continue
        if p.IsA(UsdGeom.Mesh):
            out.append(p)
    return out


def select_targets(stage):
    """메시를 가진 Xform/Mesh 프림 목록 (pseudo-root 제외)."""
    targets = []
    it = iter(stage.Traverse())
    for p in it:
        if _skip(p):
            it.PruneChildren()
            continue
        if (p.IsA(UsdGeom.Xform) or p.IsA(UsdGeom.Mesh)) and _meshes_under(p):
            targets.append(p)
    return targets


# ---------- 지오메트리 ----------

def _bound_material(prim):
    mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    return mat if mat else None


def _material_color(mat, mats):
    """머티리얼 -> (텍스처 RGB 배열 | None, 상수 diffuseColor | None). 경로별 캐시."""
    key = str(mat.GetPath())
    if key in mats:
        return mats[key]
    tex, const = None, None
    shader = mat.ComputeSurfaceSource()[0]
    inp = shader.GetInput("diffuseColor") if shader else None
    if inp:
        srcs = inp.GetConnectedSources()[0]
        if srcs:                                    # UsdUVTexture 연결
            f = UsdShade.Shader(srcs[0].source.GetPrim()).GetInput("file")
            path = f.Get().resolvedPath if f and f.Get() else ""
            if path and os.path.exists(path):
                tex = np.asarray(Image.open(path).convert("RGB"))
        elif inp.Get() is not None:
            const = np.clip(np.asarray(inp.Get(), dtype=np.float64), 0, 1)
    mats[key] = (tex, const)
    return mats[key]


def face_colors(mp, idx, fv, fid, n_faces, mats):
    """삼각형별 RGB: GeomSubset/프림 바인딩 머티리얼의 텍스처를 UV로 샘플 (없으면 상수색, 회색)."""
    m = UsdGeom.Mesh(mp)
    face_mat = [_bound_material(mp)] * n_faces
    for sub in UsdGeom.Subset.GetAllGeomSubsets(m):
        sm = _bound_material(sub.GetPrim())
        if sm:
            for i in sub.GetIndicesAttr().Get() or []:
                if i < n_faces:
                    face_mat[i] = sm
    tri_uv = None
    pv = UsdGeom.PrimvarsAPI(mp).GetPrimvar("st")
    if pv and pv.HasValue():
        vals = np.asarray(pv.ComputeFlattened(), dtype=np.float64)
        interp = pv.GetInterpolation()
        if interp == UsdGeom.Tokens.faceVarying and len(vals) > fv.max():
            tri_uv = vals[fv]
        elif interp in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying) and len(vals) > idx.max():
            tri_uv = vals[idx[fv]]
    out = np.full((len(fid), 3), 180, dtype=np.uint8)
    tri_mat = [face_mat[f] for f in fid]
    for mat in {str(x.GetPath()): x for x in tri_mat if x}.values():
        sel = np.array([x is not None and x.GetPath() == mat.GetPath() for x in tri_mat])
        tex, const = _material_color(mat, mats)
        if tex is not None and tri_uv is not None:
            # 꼭짓점 3개 + 무게중심 UV 샘플 평균
            uv = np.concatenate([tri_uv[sel], tri_uv[sel].mean(axis=1, keepdims=True)], axis=1) % 1.0
            h, w = tex.shape[:2]
            x = (uv[..., 0] * (w - 1)).astype(np.int64)
            y = ((1.0 - uv[..., 1]) * (h - 1)).astype(np.int64)
            out[sel] = tex[y, x].mean(axis=1).astype(np.uint8)
        elif const is not None:
            out[sel] = (const * 255).astype(np.uint8)
    return out


def build_mesh(mesh_prims, xcache, mats, highlight=None):
    """월드 변환 적용 + 팬 삼각화 -> 단일 trimesh (면 색 = 텍스처 샘플).
    highlight(메시 경로 집합)가 있으면 그 밖의 메시는 흰색."""
    verts, faces, colors, off = [], [], [], 0
    for mp in mesh_prims:
        m = UsdGeom.Mesh(mp)
        pts = m.GetPointsAttr().Get()
        counts = m.GetFaceVertexCountsAttr().Get()
        idx = m.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not idx:
            continue
        pts = np.asarray(pts, dtype=np.float64)
        xf = np.asarray(xcache.GetLocalToWorldTransform(mp), dtype=np.float64)
        pts = pts @ xf[:3, :3] + xf[3, :3]          # USD는 행벡터 규약
        idx = np.asarray(idx, dtype=np.int64)
        fv, fid, s = [], [], 0                      # 삼각형별 face-vertex 위치, 원래 면 번호
        for f, c in enumerate(counts):              # 쿼드/엔곤 팬 삼각화
            for k in range(1, c - 1):
                fv.append((s, s + k, s + k + 1))
                fid.append(f)
            s += c
        if not fv:
            continue
        fv = np.asarray(fv, dtype=np.int64)
        verts.append(pts)
        faces.append(idx[fv] + off)
        if highlight is not None and mp.GetPath() not in highlight:
            colors.append(np.full((len(fv), 3), 255, dtype=np.uint8))
        else:
            colors.append(face_colors(mp, idx, fv, fid, len(counts), mats))
        off += len(pts)
    if not verts:
        return None
    return trimesh.Trimesh(np.vstack(verts), np.vstack(faces),
                           face_colors=np.vstack(colors), process=False)


def geom_hash(mesh):
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(mesh.vertices, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    return h.hexdigest()


# ---------- 렌더 / VLM ----------

_PYGLET_OK = True


def _soft_render(mesh, el, az, size, up="Y"):
    """pyglet 없이 Pillow로 그리는 단순 렌더 (정사영, 페인터 알고리즘, 면 색 x 램버트)."""
    v = mesh.vertices - mesh.bounds.mean(axis=0)
    if up == "Z":                                   # Z-up -> Y-up
        v = v[:, [0, 2, 1]] * [1, 1, -1]
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    r = rx @ ry                                     # 방위각 회전 후 내려다보기
    v = v @ r.T
    n = mesh.face_normals @ r.T
    ss = size * 2                                   # 2x 슈퍼샘플링 후 축소
    scale = ss * 0.45 / max(np.linalg.norm(v, axis=1).max(), 1e-9)
    xy = np.c_[ss / 2 + v[:, 0] * scale, ss / 2 - v[:, 1] * scale]
    light = np.array([0.4, 0.6, 1.0]) / np.linalg.norm([0.4, 0.6, 1.0])
    shade = 0.15 + 0.65 * np.abs(n @ light) + 0.2 * np.abs(n[:, 2])
    order = np.argsort(v[mesh.faces, 2].mean(axis=1))   # 먼 면부터
    img = Image.new("RGB", (ss, ss), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    tri = xy[mesh.faces]
    rgb = (mesh.visual.face_colors[:, :3] * np.minimum(shade, 1.0)[:, None]).astype(np.uint8)
    for f in order:
        draw.polygon([tuple(p) for p in tri[f]], fill=tuple(int(c) for c in rgb[f]))
    img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def render_views(mesh, out_dir, base, n_views=4, size=512, up="Y"):
    """멀티뷰 오프스크린 렌더 (trimesh/pyglet, 없으면 Pillow). PNG 경로 목록 반환."""
    global _PYGLET_OK
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(n_views):
        az = 2.0 * np.pi * i / n_views
        png = None
        for attempt in range(3):        # pyglet 창 리사이즈 레이스(높이 0) 재시도
            if not _PYGLET_OK:
                png = _soft_render(mesh, 0.35, az, size, up)
                break
            try:
                scene = trimesh.Scene(mesh)
                scene.set_camera(angles=(0.35, az, 0))
                png = scene.save_image(resolution=(size, size))
                break
            except ZeroDivisionError:
                continue
            except ImportError:         # trimesh 뷰어는 pyglet<2 필요
                _PYGLET_OK = False
        if png is None:
            continue
        p = os.path.join(out_dir, f"{base}_view{i}.png")
        with open(p, "wb") as f:
            f.write(png)
        paths.append(p)
    return paths


def _ask_ollama(image_paths, prompt, model, host="http://localhost:11434", timeout=600):
    imgs = [base64.b64encode(open(p, "rb").read()).decode() for p in image_paths]
    req = urllib.request.Request(
        host + "/api/generate",
        data=json.dumps({"model": model, "prompt": prompt, "images": imgs,
                         "stream": False, "format": SCHEMA,
                         "options": {"temperature": 0}}).encode(),
        headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return _parse(r.get("response", "{}"))


SECRETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "secrets")
_claude = None


def _secret(env, fname):
    """환경변수, 없으면 secrets/<fname> 내용 (없으면 None)."""
    v = os.environ.get(env)
    p = os.path.join(SECRETS, fname)
    if not v and os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            v = f.read().strip()
    return v or None


def _ask_claude(image_paths, prompt, model):
    """Claude API 질의. 키: ANTHROPIC_API_KEY | secrets/anthropic_api_key.txt.
    워크스페이스 미지정 키면 ANTHROPIC_WORKSPACE_ID | secrets/anthropic_workspace_id.txt 도 필요."""
    global _claude
    if _claude is None:
        import anthropic
        ws = _secret("ANTHROPIC_WORKSPACE_ID", "anthropic_workspace_id.txt")
        _claude = anthropic.Anthropic(
            api_key=_secret("ANTHROPIC_API_KEY", "anthropic_api_key.txt"),
            default_headers={"anthropic-workspace-id": ws} if ws else None)
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                "data": base64.standard_b64encode(open(p, "rb").read()).decode()}}
               for p in image_paths]
    content.append({"type": "text", "text": prompt})
    r = _claude.beta.messages.create(
        model=model, max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": "low",
                       "format": {"type": "json_schema",
                                  "schema": {**SCHEMA, "additionalProperties": False}}},
        messages=[{"role": "user", "content": content}])
    if r.stop_reason == "refusal":
        print(f"    [warn] refusal: {r.stop_details}")
        return "", []
    return _parse("".join(b.text for b in r.content if b.type == "text"))


def _parse(text):
    """응답 JSON -> (이름, 키워드 목록)."""
    try:
        d = json.loads(text)
    except ValueError:
        return "", []
    name = d.get("name") if isinstance(d.get("name"), str) else ""
    return name, [k for k in d.get("keywords", []) if isinstance(k, str)]


def _ask(image_paths, prompt, model):
    """모델명이 claude- 로 시작하면 Claude API, 아니면 로컬 Ollama."""
    if model.startswith("claude-"):
        return _ask_claude(image_paths, prompt, model)
    return _ask_ollama(image_paths, prompt, model)


SINGULAR_EXC = {"lens": "lens", "lenses": "lens", "series": "series", "species": "species",
                "news": "news", "pants": "pants", "glasses": "glasses", "scissors": "scissors"}


def _singular(w):
    # 보수적 단수화: 4글자 이상, -ss/-us/-is 제외, s로 끝나는 단수/복수 고정어는 예외표
    if w in SINGULAR_EXC:
        return SINGULAR_EXC[w]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith(("xes", "ches", "shes")):
            return w[:-2]
        return w[:-1]
    return w


def _junk_names(names):
    """잡음 프림명을 베낀 키워드를 거른다.
    camelCase로 붙은 이름은 구성 단어까지: "MATCUTResult" -> {matcutresult, matcut, result}
    숫자 포함 이름은 붙인 형태와 3글자 이하 토큰만: "obj_cannon_001_hu" -> {objcannonhu, obj, hu}
    (cannon, wheel 같은 실제 단어는 살린다)"""
    out = set()
    for n in names:
        if not n:
            continue
        camel = re.search(r"[a-z][A-Z]|[A-Z]{2}[a-z]", n)
        if camel or re.search(r"\d", n):
            toks = [t.lower() for t in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+", n)]
            if camel:
                out.add(re.sub(r"[^a-z]", "", n.lower()))
                out.update(toks)
            else:                                   # 붙인 형태는 숫자 포함으로만: "cannonball"은 살린다
                out.add(re.sub(r"[^a-z0-9]", "", n.lower()))
                out.update(t for t in toks if len(t) <= 3)
    return out


def normalize(raw, names=()):
    """키워드를 한 단어씩 쪼갠 뒤 소문자/구두점 제거/단수화/중복제거/불용어·색상·잡음 프림명 제거.
    "rifle bolt" -> rifle, bolt. (kept, rejected)"""
    junk = _junk_names(names)
    kept, rejected = [], []
    for k in raw:
        t = re.sub(r"[^0-9a-z\- ]", " ", k.lower()).replace("-", " ")
        for w in map(_singular, t.split()):
            if w in STOP or w in junk or w in COLORS or w.isdigit():
                rejected.append(w)
            elif w not in kept:
                kept.append(w)
    return kept[:N_KEYWORDS], rejected


def ask_keywords(views, name, parent, parent_kw, model, ctx_views=()):
    """1회 질의 -> (키워드, 대표 이름). 5개가 안 되면 있는 만큼만 쓴다.
    대표 이름(여러 단어 가능)은 자식 프롬프트 힌트로만 쓴다."""
    # 조상 키워드 전체를 주면 그 단어(예: backrest)를 그대로 베끼므로 대표 이름만 준다
    context = PART_CONTEXT.format(main=parent_kw) if parent_kw else ""
    prompt = PROMPT.format(images=IMAGES_CTX if ctx_views else IMAGES_ALONE,
                           name=name, parent=parent, context=context)
    main, raw = _ask(list(views) + list(ctx_views), prompt, model)
    kw, _ = normalize(raw, (name, parent))
    if len(kw) < N_KEYWORDS:
        print(f"    [warn] only {len(kw)} keywords for {name}: {kw}")
    main = " ".join(re.sub(r"[^a-z ]", " ", main.lower()).split()) or (kw[0] if kw else "")
    return kw, main


def _ancestor_keywords(prim, kw_by_path):
    """가장 가까운 처리된 조상 프림과 그 대표 이름 (없으면 None, "")."""
    p = prim.GetParent()
    while p and not p.IsPseudoRoot():
        if str(p.GetPath()) in kw_by_path:
            return p, kw_by_path[str(p.GetPath())]
        p = p.GetParent()
    return None, ""


# ---------- 파일 처리 ----------

def _safe(s):
    return re.sub(r"[^0-9A-Za-z_\-]", "_", s)


def tag_stage(stage, render_dir, model="gemma4:e4b"):
    """스테이지의 대상 프림에 키워드 저작 (현재 edit target). [(path, kw, cached)] 반환.
    대상은 전위 순회 순서라 부모가 자식보다 먼저 처리되고, 부모 키워드가 자식 힌트가 된다."""
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    cache, mats, kw_by_path, results = {}, {}, {}, []
    for prim in select_targets(stage):
        mesh = build_mesh(_meshes_under(prim), xcache, mats)
        if mesh is None or len(mesh.faces) == 0:
            continue
        key = geom_hash(mesh)
        hit = key in cache
        t0 = time.time()
        if not hit:
            base = _safe(str(prim.GetPath()).strip("/").replace("/", "__")) or "root"
            views = render_views(mesh, render_dir, base, up=UsdGeom.GetStageUpAxis(stage))
            if not views:
                print(f"    [warn] render failed: {prim.GetPath()}")
                continue
            parent = prim.GetParent()
            pname = parent.GetName() if parent and not parent.IsPseudoRoot() else ""
            anc, anc_kw = _ancestor_keywords(prim, kw_by_path)
            ctx_views = []
            if anc is not None:                     # 전체 물체 렌더: 이 부위만 원래 색, 나머지는 흰색
                part = {m.GetPath() for m in _meshes_under(prim)}
                ctx = build_mesh(_meshes_under(anc), xcache, mats, highlight=part)
                ctx_views = render_views(ctx, render_dir, base + "_ctx", n_views=2,
                                         up=UsdGeom.GetStageUpAxis(stage))
            cache[key] = ask_keywords(views, prim.GetName(), pname, anc_kw, model, ctx_views)
        kw, main = cache[key]
        kw_by_path[str(prim.GetPath())] = main      # 자식 힌트용 대표 이름
        prim.CreateAttribute(ATTR, Sdf.ValueTypeNames.StringArray, custom=True).Set(kw)
        results.append((str(prim.GetPath()), kw, hit, time.time() - t0))
        print(f"  {prim.GetPath()}  ->  {kw}  ({main}){'  (cache)' if hit else ''}  [{time.time() - t0:.1f}s]")
    return results


def _usdz_root(names):
    # usdz 규칙: 첫 번째 usd 계열 파일이 루트 레이어
    for n in names:
        if n.lower().endswith((".usd", ".usda", ".usdc")):
            return n
    raise ValueError("no root layer in usdz")


def process_file(path, out_dir="keyword_tagger_out", model="gemma4:e4b"):
    name, ext = os.path.splitext(os.path.basename(path))
    ext = ext.lower()
    render_dir = os.path.join(out_dir, _safe(name))
    os.makedirs(out_dir, exist_ok=True)
    print(f"== {path}")
    if ext == ".usdz":
        tmp = tempfile.mkdtemp(prefix="kwtag_")
        try:
            with zipfile.ZipFile(path) as z:
                root = _usdz_root(z.namelist())
                z.extractall(tmp)
            root_path = os.path.join(tmp, root)
            stage = Usd.Stage.Open(root_path)
            stage.SetEditTarget(stage.GetRootLayer())
            res = tag_stage(stage, render_dir, model)
            stage.GetRootLayer().Save()
            dst = os.path.abspath(os.path.join(out_dir, f"{name}_kw.usdz"))
            if os.path.exists(dst):
                os.remove(dst)
            if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(root_path), dst):
                raise RuntimeError("usdz packaging failed: " + dst)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    elif ext in (".glb", ".gltf"):                  # usda로 변환한 뒤 그 파일에 저작
        from .glb import glb_to_usda
        dst = glb_to_usda(path, os.path.join(out_dir, f"{name}_kw.usda"))
        stage = Usd.Stage.Open(dst)
        stage.SetEditTarget(stage.GetRootLayer())
        res = tag_stage(stage, render_dir, model)
        stage.GetRootLayer().Save()
    else:
        stage = Usd.Stage.Open(path)
        stage.SetEditTarget(stage.GetRootLayer())
        res = tag_stage(stage, render_dir, model)
        dst = os.path.join(out_dir, f"{name}_kw.usda")
        stage.GetRootLayer().Export(dst)
    if Usd.Stage.Open(dst) is None:                 # 결과 파일 열리는지 확인
        raise RuntimeError("cannot reopen " + dst)
    hits = sum(1 for r in res if r[2])
    print(f"-> {dst}  ({len(res)} prims, {hits} cache hits)")
    return dst, res
