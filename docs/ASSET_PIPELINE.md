# Asset Pipeline

## Textures (JPG/PNG → KTX2)
- Standard textures (JPG/PNG) are converted to KTX2 for runtime use.

## HDRI / Sky (EXR → Cubemap DDS → KTX2)
- Source HDRIs are provided as EXR.
- EXR is baked into a cubemap DDS using IBLBaker.
- The cubemap DDS is converted to KTX2 using NVIDIA Texture Tools Exporter (2024.1.1).

## Notes
- Source URLs and licenses are documented in `docs/ASSET_CREDITS.md`.
