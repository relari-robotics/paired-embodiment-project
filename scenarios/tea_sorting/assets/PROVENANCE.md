# Tea-sorting assets

## Wrapper artwork

`tea_wrappers.png`: original generated print atlas, 1942 × 809 pixels; approximately 647 × 809 per face. Generated with the built-in image-generation tool on 2026-09-14, then copied into this project. Not a photograph of a commercial brand. Both packet faces currently reuse this print; the back is not a separately authored ingredients panel. No image upscaling or invented higher-resolution source is claimed.

Final prompt:

Use case: product-mockup. Asset type: production UV base-color texture atlas for three individually wrapped tea sachets in a robotics simulation, NOT a rendered scene.
Create one flat rectangular print artwork atlas, three equally wide vertical panels edge-to-edge from left to right: deep red rooibos tea wrapper, warm yellow chamomile tea wrapper, deep blue Earl Grey tea wrapper. Each panel is the same portrait rectangle, width to height 0.8, total canvas aspect 2.4:1. Entire panel background carries its color right to the boundary, no gaps, no exterior background, no perspective, no shadows, no 3D bags.
Premium believable supermarket tea packaging: subtle ivory engraved botanical illustration appropriate to each tea, refined serif type, narrow fine ornamental border inset 8%, realistic tiny ink texture on lightly fibrous matte coated paper, small print detail, generous margins for heat sealing. Exact main text on red: "ROOIBOS", yellow: "CHAMOMILE", blue: "EARL GREY". Small secondary text on each: "PURE TEA" at top and "1 TEA BAG • 2 g" at bottom. No brand names, no watermark. The three panels must be precisely equal width, perfectly front-on printable diffuse artwork, consistent type hierarchy, muted sophisticated red/yellow/blue easily distinguished by robot vision. Highest possible resolution and crisp print detail. No baked lighting, no drop shadows, no rounded panel corners.

## Photographed wood

Source: [Wood Table 001, Poly Haven](https://polyhaven.com/a/wood_table_001).
License: CC0. Authors: Dimitrios Savva (photography), Rico Cilliers (processing).
Downloaded 2026-09-14 from Poly Haven's asset service. All three maps are 8K JPG; base color is sRGB, roughness and OpenGL tangent-space normal are non-color data. The renderer maps the scan at 1.5 m physical scale, reduces saturation, and remaps roughness to a satin finish. No proprietary textures are fetched at render time.

- `wood_table_001_diff_8k.jpg`: https://dl.polyhaven.org/file/ph-assets/Textures/jpg/8k/wood_table_001/wood_table_001_diff_8k.jpg
- `wood_table_001_rough_8k.jpg`: https://dl.polyhaven.org/file/ph-assets/Textures/jpg/8k/wood_table_001/wood_table_001_rough_8k.jpg
- `wood_table_001_nor_gl_8k.jpg`: https://dl.polyhaven.org/file/ph-assets/Textures/jpg/8k/wood_table_001/wood_table_001_nor_gl_8k.jpg

## Geometry and lighting

Packet pillow geometry, perimeter heat seals, crimp ridges, paper micro-bump, beveled wooden boards, enamel labels and brass pins are generated from project code. Wrinkles are static render microgeometry, not simulated deformation. Robot render meshes come from the project's existing OpenArm assets. Lighting uses Blender's bundled studio environment plus area lights. Training cameras disable photographic depth of field by default, preserving readable grasp targets.
