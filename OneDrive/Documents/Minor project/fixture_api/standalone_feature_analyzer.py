"""
Standalone Feature Analyzer - CadQuery port of Feature_Extractor.py

Runs OUTSIDE Fusion 360. Takes a .stp file path, returns the same kind
of report + 3-2-1 support points your Fusion script produces, plus a
.glb mesh for a web viewer.

Install:
    pip install cadquery

This is a library, not a script to double-click - see analyze_step()
at the bottom for the entry point a FastAPI endpoint would call.
"""

import os
from itertools import combinations
import cadquery as cq
from cadquery.occ_impl.shapes import Shape, Compound


# ---------------------------------------------------------------------
# MATERIAL PROPERTIES (yield strength, MPa == N/mm^2) and load/contact
# defaults used by the stress-aware support solver below. Override any
# of these per-call via analyze_step()'s keyword arguments.
# ---------------------------------------------------------------------
MATERIAL_YIELD_MPA = {
    "Aluminum 6061-T6": 276.0,
    "Aluminum 6061-O": 55.0,
    "Steel (mild, A36)": 250.0,
    "Stainless Steel 304": 215.0,
    "Cast Iron (gray)": 130.0,
    "Titanium Ti-6Al-4V": 880.0,
}

DEFAULT_MATERIAL = "Aluminum 6061-T6"
DEFAULT_SAFETY_FACTOR = 2.0     # allowable stress = yield / safety_factor
DEFAULT_APPLIED_FORCE_N = 500.0  # total clamping + machining reaction load
DEFAULT_PIN_CONTACT_RADIUS_MM = 5.0  # fixture pin/support tip contact radius
DEFAULT_PIN_HEIGHT_MM = 15.0  # how tall the visualized/exported clamp pins are
DEFAULT_NUM_SUPPORT_POINTS = 4  # supports to place (falls back to 3 if 4 isn't physically valid)


# ---------------------------------------------------------------------
# CAD (Z-up, mm) -> glTF (Y-up, meters) coordinate conversion
# ---------------------------------------------------------------------
def cad_to_gltf_point(x, y, z):
    """(x, y, z) in mm, Z-up  ->  (x, z, -y) in meters, Y-up."""
    return (x / 1000.0, z / 1000.0, -y / 1000.0)


# ---------------------------------------------------------------------
# SMART CYLINDER CLASSIFIER  (same thresholds as Fusion version)
# ---------------------------------------------------------------------
def classify_cylinder(face):
    """
    face: a cadquery Face object whose geomType() == 'CYLINDER'
    Returns ('Through Hole' | 'Corner Fillet' | 'Unknown Cylinder', info)
    """
    geom = face._geomAdaptor()  # BRepAdaptor_Surface
    cyl = geom.Cylinder()
    radius = cyl.Radius()
    diameter = radius * 2
    edge_count = len(face.Edges())

    origin = cyl.Location()  # gp_Pnt
    axis_dir = cyl.Axis().Direction()

    info = {
        "center": (origin.X(), origin.Y(), origin.Z()),
        "axis": (axis_dir.X(), axis_dir.Y(), axis_dir.Z()),
        "radius": radius,
        "diameter": diameter,
        "area": face.Area(),
        "edge_count": edge_count,
    }

    # Through hole. STEP files are in mm (vs Fusion's cm), so the
    # threshold is 10x Fusion's: diameter <= 6.0 mm covers the 5mm holes.
    # edge_count is 2 or 3: OpenCascade adds a seam edge for full-revolution
    # cylinders that Fusion's kernel doesn't count separately.
    if edge_count in (2, 3) and diameter <= 6.0:
        return "Through Hole", info

    # Corner fillet (Fusion threshold: edge_count == 4)
    if edge_count == 4:
        return "Corner Fillet", info

    return "Unknown Cylinder", info


# ---------------------------------------------------------------------
# 3-2-1 SUPPORT POINT SOLVER  (identical math to the Fusion version)
# ---------------------------------------------------------------------
def solve_reactions(p1, p2, p3, cog_x, cog_y):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3

    def det3(a):
        return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))

    A = [[1, 1, 1], [x1, x2, x3], [y1, y2, y3]]
    b = [1, cog_x, cog_y]
    detA = det3(A)
    if abs(detA) < 1e-9:
        return None
    R = []
    for i in range(3):
        Ai = [row[:] for row in A]
        for r in range(3):
            Ai[r][i] = b[r]
        R.append(det3(Ai) / detA)
    return R


def _det3(a):
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


def solve_reactions_n(points, cog_x, cog_y):
    """
    Reaction-fraction solution for N >= 3 support points satisfying
    force/moment equilibrium about the part's CoG: sum(R)=1,
    sum(R*x)=cog_x, sum(R*y)=cog_y.

    For N == 3 this is exactly determined (same result as
    solve_reactions). For N > 3 (e.g. 4 corner supports on a flat
    plate) the system is statically indeterminate, so this returns the
    minimum-norm least-squares solution R = A^T (A A^T)^-1 b - the
    standard assumption for equal-stiffness rigid supports, which
    spreads load across all points rather than picking an arbitrary
    determinate subset.
    """
    n = len(points)
    sum_x = sum(p[0] for p in points)
    sum_y = sum(p[1] for p in points)
    sum_xx = sum(p[0] ** 2 for p in points)
    sum_xy = sum(p[0] * p[1] for p in points)
    sum_yy = sum(p[1] ** 2 for p in points)

    M = [
        [n, sum_x, sum_y],
        [sum_x, sum_xx, sum_xy],
        [sum_y, sum_xy, sum_yy],
    ]
    b = [1, cog_x, cog_y]

    detM = _det3(M)
    if abs(detM) < 1e-9:
        return None
    v = []
    for i in range(3):
        Mi = [row[:] for row in M]
        for r in range(3):
            Mi[r][i] = b[r]
        v.append(_det3(Mi) / detM)

    v0, v1, v2 = v
    return [v0 + v1 * p[0] + v2 * p[1] for p in points]


def candidate_primary_points(bbox, hole_centers, hole_radius, margin):
    corners = [
        (bbox.xmax - margin, bbox.ymax - margin),
        (bbox.xmax - margin, bbox.ymin + margin),
        (bbox.xmin + margin, bbox.ymax - margin),
        (bbox.xmin + margin, bbox.ymin + margin),
    ]
    return [
        (cx, cy) for cx, cy in corners
        if all(((cx - hx) ** 2 + (cy - hy) ** 2) ** 0.5 > hole_radius * 3
               for hx, hy in hole_centers)
    ]


def choose_primary_points(corners, cog_x, cog_y):
    """Original geometry-only selector (kept for backward compatibility /
    comparison). Prefer choose_primary_points_weighted() below, which
    folds in contact-stress and CoG-centering."""
    best, best_spread = None, None
    for combo in combinations(corners, 3):
        R = solve_reactions(combo[0], combo[1], combo[2], cog_x, cog_y)
        if R is None or any(r < 0 for r in R):
            continue
        spread = max(R) - min(R)
        if best_spread is None or spread < best_spread:
            best_spread, best = spread, (combo, R)
    return best


# ---------------------------------------------------------------------
# STRESS-STRAIN + CENTER-OF-GRAVITY AWARE SUPPORT SELECTION
# ---------------------------------------------------------------------
def contact_stress_mpa(reaction_fraction, applied_force_n, pin_radius_mm):
    """
    Simplified bearing/Hertzian-style contact stress at one support point:
    the point carries `reaction_fraction` of the total applied load,
    spread over a circular pin-tip contact area.

    stress (N/mm^2 = MPa) = force (N) / area (mm^2)
    """
    force_n = reaction_fraction * applied_force_n
    area_mm2 = 3.141592653589793 * (pin_radius_mm ** 2)
    if area_mm2 <= 0:
        return float("inf")
    return force_n / area_mm2


def centroid_offset_from_cog(combo, cog_x, cog_y):
    """Distance from the support combo's centroid to the part's CoG
    (projected onto XY, i.e. the plane the part rests on). Smaller is
    better: it means the supports bracket the CoG rather than sitting
    off to one side."""
    n = len(combo)
    cx = sum(p[0] for p in combo) / n
    cy = sum(p[1] for p in combo) / n
    return ((cx - cog_x) ** 2 + (cy - cog_y) ** 2) ** 0.5


def _evaluate_combo(combo, cog_x, cog_y, allowable_mpa, applied_force_n, pin_radius_mm, bbox_diag):
    if len(combo) == 3:
        R = solve_reactions(combo[0], combo[1], combo[2], cog_x, cog_y)
    else:
        R = solve_reactions_n(combo, cog_x, cog_y)

    if R is None or any(r < -1e-6 for r in R):
        return None

    spread = max(R) - min(R)
    cog_offset = centroid_offset_from_cog(combo, cog_x, cog_y)
    stresses = [contact_stress_mpa(r, applied_force_n, pin_radius_mm) for r in R]
    max_stress = max(stresses)
    overstressed = max_stress > allowable_mpa
    stress_overshoot = max(0.0, max_stress - allowable_mpa)

    score = (
        spread
        + (cog_offset / bbox_diag)
        + 5.0 * (stress_overshoot / allowable_mpa if allowable_mpa else stress_overshoot)
    )

    return {
        "combo": combo,
        "reactions": R,
        "cog_offset_mm": cog_offset,
        "stresses_mpa": stresses,
        "max_stress_mpa": max_stress,
        "overstressed": overstressed,
        "score": score,
    }


def choose_primary_points_weighted(
    corners,
    cog_x,
    cog_y,
    material=DEFAULT_MATERIAL,
    yield_strength_mpa=None,
    safety_factor=DEFAULT_SAFETY_FACTOR,
    applied_force_n=DEFAULT_APPLIED_FORCE_N,
    pin_radius_mm=DEFAULT_PIN_CONTACT_RADIUS_MM,
    num_support_points=DEFAULT_NUM_SUPPORT_POINTS,
):
    """
    Picks the primary support points by combining three factors instead
    of geometric spread alone:

      1. Load balance   - reactions should be as even as possible
      2. CoG centering  - the supports' centroid should sit close to
                           the part's center of gravity, so the part
                           doesn't tip/rock on its supports
      3. Contact stress - the peak contact stress at any one support
                           point (reaction force / pin contact area)
                           must stay under the material's allowable
                           stress (yield / safety_factor). Candidates
                           that violate this are only used as a last
                           resort, and are flagged in the result.

    Tries `num_support_points` supports first (default 4 - e.g. all
    four corners of a rectangular plate; statically indeterminate, so
    reactions come from the minimum-norm least-squares solution). If
    that yields no physically valid (non-negative reaction) candidate
    - e.g. the CoG sits somewhere a 4-point equal-stiffness split can't
    support without a negative/pulling reaction - falls back to the
    classic 3-point statically-determinate solution, which always has
    a solution as long as 3 non-collinear candidates exist.

    Returns (combo, reactions, diagnostics) or None if nothing works.
    """
    if yield_strength_mpa is None:
        yield_strength_mpa = MATERIAL_YIELD_MPA.get(material, MATERIAL_YIELD_MPA[DEFAULT_MATERIAL])
    allowable_mpa = yield_strength_mpa / safety_factor

    # Normalize scoring terms so load balance, CoG offset (mm), and
    # stress overshoot (MPa) contribute on a comparable scale.
    bbox_diag = max(
        (max(c[0] for c in corners) - min(c[0] for c in corners)) if corners else 1.0,
        1.0,
    )

    def candidates_for(k):
        if k > len(corners):
            return []
        combos = [tuple(corners)] if len(corners) == k else list(combinations(corners, k))
        out = []
        for combo in combos:
            ev = _evaluate_combo(combo, cog_x, cog_y, allowable_mpa, applied_force_n, pin_radius_mm, bbox_diag)
            if ev is not None:
                out.append(ev)
        return out

    used_fallback_3point = False
    candidates = candidates_for(num_support_points)
    if not candidates and num_support_points != 3 and len(corners) >= 3:
        # 4-point (or N-point) equilibrium had no physically valid
        # solution - fall back to the classic determinate 3-point case.
        candidates = candidates_for(3)
        used_fallback_3point = bool(candidates)

    if not candidates:
        return None

    # Prefer candidates that don't violate the stress limit; only fall
    # back to an overstressed one if literally nothing else qualifies.
    safe = [c for c in candidates if not c["overstressed"]]
    pool = safe if safe else candidates
    best = min(pool, key=lambda c: c["score"])

    diagnostics = {
        "material": material,
        "yield_strength_mpa": yield_strength_mpa,
        "safety_factor": safety_factor,
        "allowable_stress_mpa": allowable_mpa,
        "applied_force_n": applied_force_n,
        "pin_radius_mm": pin_radius_mm,
        "cog_offset_mm": best["cog_offset_mm"],
        "max_contact_stress_mpa": best["max_stress_mpa"],
        "per_point_stress_mpa": best["stresses_mpa"],
        "overstressed": best["overstressed"],
        "used_fallback_overstressed_candidate": best["overstressed"] and bool(candidates) and not safe,
        "num_support_points": len(best["combo"]),
        "used_fallback_3point": used_fallback_3point,
    }

    return best["combo"], best["reactions"], diagnostics


# ---------------------------------------------------------------------
# CLAMP / SUPPORT PIN GEOMETRY
#
# Each 3-2-1 support point becomes a physical cylindrical pin standing
# under the part, touching it at (x, y, z_top) and extending downward
# by pin_height_mm. These are what get drawn into the .glb viewer and
# baked into the exported .stp.
# ---------------------------------------------------------------------
def build_clamp_pin_shape(x, y, z_top, radius_mm, height_mm):
    """Returns a CadQuery/OCC Shape: a cylinder whose top face sits at
    (x, y, z_top) and extends downward by height_mm - i.e. a fixture
    pin supporting the part from below at that support point."""
    pin = (
        cq.Workplane("XY")
        .circle(radius_mm)
        .extrude(-height_mm)
        .translate((x, y, z_top))
    )
    return pin.val()


def build_clamp_pin_trimesh(x, y, z_top, radius_mm, height_mm, sections=32):
    """Same pin, as a trimesh mesh in CAD mm coordinates (not yet
    converted to glTF space) - used for the web-viewer .glb."""
    import trimesh

    mesh = trimesh.creation.cylinder(radius=radius_mm, height=height_mm, sections=sections)
    # trimesh's cylinder is centered on its own origin (spans
    # [-height/2, +height/2]); shift so the TOP face sits at z=0, then
    # move to the actual support point.
    mesh.apply_translation((0, 0, -height_mm / 2.0))
    mesh.apply_translation((x, y, z_top))
    return mesh


# ---------------------------------------------------------------------
# MAIN ENTRY POINT - what your FastAPI endpoint calls
# ---------------------------------------------------------------------
def analyze_step(
    stp_path: str,
    glb_out_path: str = None,
    stp_out_path: str = None,
    material: str = DEFAULT_MATERIAL,
    yield_strength_mpa: float = None,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    applied_force_n: float = DEFAULT_APPLIED_FORCE_N,
    pin_radius_mm: float = DEFAULT_PIN_CONTACT_RADIUS_MM,
    pin_height_mm: float = DEFAULT_PIN_HEIGHT_MM,
    num_support_points: int = DEFAULT_NUM_SUPPORT_POINTS,
):
    result = cq.importers.importStep(stp_path)
    shape = result.val()

    faces = shape.Faces()

    report_lines = ["SMART FEATURE RECOGNITION REPORT", ""]
    through_holes = corner_fillets = unknown_cylinders = feature_number = 0
    hole_centers = []

    for face in faces:
        if face.geomType() != "CYLINDER":
            continue

        feature_number += 1
        feature_type, info = classify_cylinder(face)

        if feature_type == "Through Hole":
            through_holes += 1
            hole_centers.append((info["center"][0], info["center"][1]))
        elif feature_type == "Corner Fillet":
            corner_fillets += 1
        else:
            unknown_cylinders += 1

        report_lines.append(f"========== Feature {feature_number} ==========")
        report_lines.append(f"Type     : {feature_type}")
        report_lines.append(f"Center   : {tuple(round(v, 2) for v in info['center'])}")
        report_lines.append(f"Radius   : {info['radius']:.2f} mm")
        report_lines.append(f"Diameter : {info['diameter']:.2f} mm")
        report_lines.append(f"Axis     : {tuple(round(v, 2) for v in info['axis'])}")
        report_lines.append(f"Area     : {info['area']:.2f} mm^2")
        report_lines.append(f"Edges    : {info['edge_count']}")
        report_lines.append("")

    # --- mass properties + bounding box (CadQuery/OCC equivalents of
    #     Fusion's body.physicalProperties / body.boundingBox) ---
    cog = shape.Center()        # cq.Vector - center of mass
    bbox = shape.BoundingBox()

    corners = candidate_primary_points(bbox, hole_centers, hole_radius=2.5, margin=10.0)
    support_result = choose_primary_points_weighted(
        corners, cog.x, cog.y,
        material=material,
        yield_strength_mpa=yield_strength_mpa,
        safety_factor=safety_factor,
        applied_force_n=applied_force_n,
        pin_radius_mm=pin_radius_mm,
        num_support_points=num_support_points,
    )

    support_points = []
    clamp_analysis = None
    if support_result:
        pts, reactions, diag = support_result
        clamp_analysis = diag
        report_lines.append(f"{diag['num_support_points']}-POINT PRIMARY SUPPORT LAYOUT (stress + CoG aware)")
        if diag["used_fallback_3point"]:
            report_lines.append(
                f"  NOTE: requested {num_support_points} support points, but no physically valid"
                f" (non-negative reaction) {num_support_points}-point solution existed for this"
                f" CoG - fell back to the classic 3-point determinate solution."
            )
        report_lines.append(
            f"  Material: {diag['material']}  |  Yield: {diag['yield_strength_mpa']:.1f} MPa"
            f"  |  Safety factor: {diag['safety_factor']:.1f}x"
            f"  |  Allowable stress: {diag['allowable_stress_mpa']:.1f} MPa"
        )
        report_lines.append(
            f"  Applied load: {diag['applied_force_n']:.1f} N total"
            f"  |  Pin contact radius: {diag['pin_radius_mm']:.1f} mm"
        )
        report_lines.append(f"  Support-triangle centroid to CoG offset: {diag['cog_offset_mm']:.2f} mm")
        for (px, py), r, s in zip(pts, reactions, diag["per_point_stress_mpa"]):
            support_points.append({
                "x": px, "y": py,
                "reaction_pct": r * 100,
                "contact_stress_mpa": s,
            })
            flag = "  ** OVER ALLOWABLE STRESS **" if s > diag["allowable_stress_mpa"] else ""
            report_lines.append(
                f"  ({px:.2f}, {py:.2f}) mm  -> reaction share: {r*100:.1f}%"
                f"  |  contact stress: {s:.1f} MPa{flag}"
            )
        if diag["used_fallback_overstressed_candidate"]:
            report_lines.append(
                "  WARNING: every candidate triangle exceeded the allowable stress; "
                "showing the least-bad option. Consider a larger pin contact radius, "
                "more support points, or a stronger material."
            )
    else:
        report_lines.append("No valid 3-point support triangle found")

    report_lines.append("")
    report_lines.append("==============================")
    report_lines.append("FEATURE SUMMARY")
    report_lines.append("==============================")
    report_lines.append(f"Through Holes     : {through_holes}")
    report_lines.append(f"Corner Fillets    : {corner_fillets}")
    report_lines.append(f"Unknown Cylinders : {unknown_cylinders}")

    report_text = "\n".join(report_lines)

    # --- clamp/support pin geometry (CAD space, mm) --------------------
    # Pins stand under the part's primary locating plane (bottom face,
    # z = bbox.zmin), touching it at each support point's (x, y).
    z_top = bbox.zmin
    pin_shapes = [
        build_clamp_pin_shape(pt["x"], pt["y"], z_top, pin_radius_mm, pin_height_mm)
        for pt in support_points
    ]

    # --- export a mesh for the web viewer, WITH the clamp pins drawn on it ---
    # CadQuery doesn't export GLB directly, so we go STEP -> STL -> GLB
    # (via trimesh), and rotate Z-up (CAD) into Y-up (glTF) on the way.
    # Pins are built directly as trimesh cylinders (cheaper than round-
    # tripping each one through STEP/STL) and merged into one Scene so
    # they show up as red pins on the part in the viewer.
    # pip install trimesh
    if glb_out_path:
        import trimesh
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp_stl:
            stl_path = tmp_stl.name
        cq.exporters.export(shape, stl_path, exportType="STL")

        def to_gltf_space(mesh):
            rotation = trimesh.transformations.rotation_matrix(
                angle=-1.5707963267948966,  # -90 degrees, in radians
                direction=[1, 0, 0],
            )
            mesh.apply_scale(0.001)      # mm -> m
            mesh.apply_transform(rotation)  # Z-up -> Y-up
            return mesh

        part_mesh = to_gltf_space(trimesh.load(stl_path))
        part_mesh.visual.face_colors = [180, 180, 190, 255]  # neutral gray

        scene = trimesh.Scene()
        scene.add_geometry(part_mesh, geom_name="part")

        for i, pt in enumerate(support_points):
            pin_mesh = build_clamp_pin_trimesh(pt["x"], pt["y"], z_top, pin_radius_mm, pin_height_mm)
            pin_mesh = to_gltf_space(pin_mesh)
            over = pt.get("contact_stress_mpa", 0) > (clamp_analysis["allowable_stress_mpa"] if clamp_analysis else float("inf"))
            pin_mesh.visual.face_colors = [220, 40, 40, 255] if over else [220, 30, 30, 255]
            scene.add_geometry(pin_mesh, geom_name=f"clamp_{i + 1}")

        scene.export(glb_out_path)
        os.unlink(stl_path)

    # --- export the part + clamp pins as one downloadable .stp ---------
    # Uses a colored Assembly (gray part, red pins) so the STEP file
    # itself shows the clamp positions highlighted when opened in
    # Fusion 360 or another CAD tool that reads STEP color/style data.
    # Falls back to an uncolored compound if colored Assembly export
    # isn't supported by the installed CadQuery/OCC version.
    if stp_out_path:
        try:
            assy = cq.Assembly()
            assy.add(shape, name="part", color=cq.Color(0.75, 0.75, 0.78, 1.0))
            for i, pin_shape in enumerate(pin_shapes):
                assy.add(pin_shape, name=f"clamp_{i + 1}", color=cq.Color(0.9, 0.1, 0.1, 1.0))
            assy.save(stp_out_path, exportType="STEP")
        except Exception:
            assembly_shape = Compound.makeCompound([shape] + pin_shapes)
            cq.exporters.export(assembly_shape, stp_out_path, exportType="STEP")

    # Support points + COG converted into the same space as the exported
    # mesh, so the frontend can place hotspots directly on the model.
    support_points_gltf = [
        {**pt, **dict(zip(("x", "y", "z"), cad_to_gltf_point(pt["x"], pt["y"], 0)))}
        for pt in support_points
    ]
    cog_gltf = dict(zip(("x", "y", "z"), cad_to_gltf_point(cog.x, cog.y, cog.z)))

    return {
        "report_text": report_text,
        "support_points": support_points,          # original CAD mm coordinates
        "support_points_gltf": support_points_gltf,  # meters, matches the .glb
        "through_holes": through_holes,
        "corner_fillets": corner_fillets,
        "unknown_cylinders": unknown_cylinders,
        "center_of_mass": {"x": cog.x, "y": cog.y, "z": cog.z},
        "center_of_mass_gltf": cog_gltf,
        "clamp_analysis": clamp_analysis,  # material/force/stress diagnostics, or None
        "pin_height_mm": pin_height_mm,
        "glb_path": glb_out_path,   # part + clamp pins, for the web viewer
        "stp_path": stp_out_path,   # part + clamp pins, downloadable CAD file
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python standalone_feature_analyzer.py path/to/part.stp [out_dir]")
        sys.exit(1)

    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(sys.argv[1])) or "."
    glb_out = os.path.join(out_dir, "fixture_layout.glb")
    stp_out = os.path.join(out_dir, "fixture_layout.stp")

    result = analyze_step(sys.argv[1], glb_out_path=glb_out, stp_out_path=stp_out)
    print(result["report_text"])
    print()
    print(f"Viewer mesh (part + clamp pins) : {glb_out}")
    print(f"Downloadable CAD file (part + clamp pins) : {stp_out}")