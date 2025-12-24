# Third-Party Notices

This repository uses and/or references third-party software.
Each component is licensed by its respective copyright holder(s).
Please refer to the upstream repositories for the full license texts.

> Notes
> - “Vendored” means the source is included in this repository (often via git submodule / copied).
> - “External” means fetched via a package manager (e.g., vcpkg) and not stored in this repository.

---

## Vendored / Included in this repository

### Dear ImGui
- Upstream: https://github.com/ocornut/imgui
- License: MIT
- Local path: `external/imgui` (via `add_subdirectory`)
- License file: `external/imgui/LICENSE.txt`

### vulkan_radix_sort (vk_radix_sort)
- Upstream: https://github.com/jaesung-cs/vulkan_radix_sort
- License: MIT
- Local path: `external/vulkan_radix_sort` (or `external/vk_radix_sort`) (via `add_subdirectory`)
- License file: `external/vulkan_radix_sort/LICENSE` (or `external/vk_radix_sort/LICENSE`)

---

## External dependencies (installed via package manager, not vendored)

> Recommended install method: vcpkg.
> When distributing binaries, ensure you comply with each dependency’s license.

### GLFW
- Upstream: https://github.com/glfw/glfw
- License: zlib/libpng

### KTX-Software (libktx)
- Upstream: https://github.com/KhronosGroup/KTX-Software
- License: Apache-2.0

### fmt
- Upstream: https://github.com/fmtlib/fmt
- License: MIT

---

## Code referenced / inspired by (no code vendored unless stated)

### Velvet (README structure + some implementation reference)
- Upstream: https://github.com/vitalight/Velvet
- License: MIT
- Note: Some code patterns and UI/timestep handling were referenced from Velvet.

### elasty (README reference formatting inspiration)
- Upstream: https://github.com/yuki-koyama/elasty
- License: MIT

---

## Additional notes
- If any third-party code/assets are later copied into this repository, add them to the “Vendored / Included” section
  and keep their LICENSE files alongside the copied code.
