"""
Inspect simple.usdc and ERCP.usdc inside Isaac Sim runtime (pxr available after app starts).
"""
import sys
sys.stdout = sys.stderr

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom

for usd_path in [
    "/data/ERCP/ercp_access/simple.usdc",
    "/data/ERCP/ercp_access/ERCP.usdc",
]:
    print()
    print("=" * 60)
    print(" ", usd_path)
    print("=" * 60)
    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    print(f"  upAxis       : {UsdGeom.GetStageUpAxis(stage)}")
    print(f"  metersPerUnit: {UsdGeom.GetStageMetersPerUnit(stage)}")
    dp = stage.GetDefaultPrim()
    print(f"  defaultPrim  : {dp.GetName() if dp else None}")
    print()

    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        typ  = prim.GetTypeName()
        xf   = UsdGeom.Xformable(prim)
        trans = None
        if xf:
            try:
                mat   = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                trans = tuple(round(v, 6) for v in mat.ExtractTranslation())
            except Exception:
                pass
        bbox_str = ""
        if typ == "Mesh":
            try:
                rng = bc.ComputeWorldBound(prim).GetRange()
                mn  = tuple(round(v, 4) for v in rng.GetMin())
                mx  = tuple(round(v, 4) for v in rng.GetMax())
                ctr = tuple(round((a+b)/2, 4) for a, b in zip(rng.GetMin(), rng.GetMax()))
                bbox_str = f"\n      bbox_min={mn}\n      bbox_max={mx}\n      bbox_ctr={ctr}"
            except Exception:
                bbox_str = "\n      bbox=ERROR"
        print(f"  {path}  type={typ}  translate={trans}{bbox_str}")

app.close()
print("\nDONE")
