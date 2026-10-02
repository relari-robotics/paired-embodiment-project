# Hand scan

**Hand by Artec 3D / Artec Group Inc.**

- Source: https://www.artec3d.com/3d-models/hand
- Download: https://cdn.artec3d.com/content-hub-3dmodels/hand-ply.7z?VersionId=dpiGMk34fv5g.2GjC3Ug__Rwflh3B12W
- Retrieved: 2026-09-14
- The downloaded archive supplies CC BY 3.0 Unported; its license is preserved
  verbatim in `LICENSE.txt`. The current source page also offers CC BY 4.0.
- `hand.ply` is the unmodified scan: millimetres, 400,002 vertices and 800,000
  triangles. It contains geometry, including nails and palm creases, but no
  scanned color texture.
- Rendering adaptations: unit conversion, landmark-based pose fitting and
  skinning to the recorded hand, and procedural skin shading. These adaptations
  are not endorsed by Artec 3D. Preserve attribution when distributing renders
  or derived models.

License links: https://creativecommons.org/licenses/by/3.0/ and
https://creativecommons.org/licenses/by/4.0/

`preview.glb` is a reduced-detail skinned derivative with an authored arm and
procedural base skin color. `preview_bind.json` records its actor bind matrices
for replay retargeting. Rebuild both together with `tools/build_human_preview.py`
inside a saved Blender review scene.
