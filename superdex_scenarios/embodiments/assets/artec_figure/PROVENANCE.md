# Standing human scan

- **Creator:** Artec 3D, “Full body scan”.
- **Source:** https://www.artec3d.com/3d-models/full-body-scan
- **Download:** https://cdn.artec3d.com/content-hub-3dmodels/full_body_obj.zip?VersionId=S9mYd33C2zSWqiJkOErbUJAkC6uiJzWS
- Retrieved 2026-09-14.
- The source archive includes the Creative Commons Attribution **3.0** license,
  reproduced in LICENSE.txt. The webpage displays CC BY 4.0; this derivative
  retains the archive's attribution and license notice.

Adaptations: converted to metres and glTF, removed the scanned wooden platform,
removed the working forearm, reconstructed the clothing beneath its hand,
baked a forward bend at the hips, and compressed the photographic texture.
This person is a stationary visual prop. The working arm is a separate skinned
surface and the working hand derives from Artec's separate hand scan.

Rebuild with `tools/build_human_figure_assets.py --source PATH/full_body.obj`.
The source OBJ is not bundled. The derived mesh and embedded texture are in
`standing.glb`; placement metadata uses +X right, +Y forward, +Z up.
