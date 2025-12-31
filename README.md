# PowerEngine

> GPU-accelerated particle simulation engine based on **Extended Position Based Dynamics (XPBD)**.  
> Real-time oriented particle systems (cloth / softbody / fluid, etc.) built on a unified XPBD solver.

<p align="center">
  <img src="docs/media/preview.png" alt="PowerEngine preview" width="600" />
</p>

---

## Why another XPBD particle engine?

Most real-time particle simulators either:
- focus on a single material model (cloth-only / fluid-only), or
- run on CPU and struggle with performance at higher particle counts, or
- hide critical implementation details behind closed source.

PowerEngine aims to:
- provide a **unified XPBD solver** that can cover multiple particle systems
- keep the architecture **GPU-friendly** for real-time iteration

---

## Demo

<p align="center">
  <img src="docs/media/horizontal_drop.gif" alt="horizontal_drop" width="400" />
  <img src="docs/media/vertical_drop.gif" alt="vertical_drop" width="400" />
</p>
<p align="center">
  <img src="docs/media/pinned_corner.gif" alt="pinned_corner" width="400" />
  <img src="docs/media/top_pinned_corner.gif" alt="top_pinned_corner" width="400" />
</p>
<p align="center">
  <img src="docs/media/wind2.gif" alt="wind" width="800" />
</p>

---

## Features

### Core
- [x] Extended position-based dynamics (XPBD) [Macklin+16]
- [x] Small-steps XPBD [Macklin+19]

### Constraints Solve Scheme
- **G**: Gauss–Seidel (sequential, in-place projection)
- **A**: Parallel accumulation (atomic add + apply; Jacobi-style)

> Note:
> - The accumulation path is **Jacobi-style** (constraints write to accumulators, then positions are updated in a separate pass).
> - Due to atomic update order, results are **not bitwise deterministic** and may converge differently from classic GS/Jacobi.

### Cloth
- [x] Distance/Stretch (G) [Müller+07]
- [x] Shear (A) [Müller+14]
- [x] Bending (A) [Müller+07]
- [x] Area (A) [Müller+14]
- [x] Self-Collision (A)
- [x] Inter-Collision (A)
- [x] LRA(Long Range Attachments) (per-particle sequential projection; GS-like) [Kim+12]

### Softbody
- [x] Stretch (A) [Müller+14]
- [x] Volume (A) [Müller+14]

### External Force
- [x] Wind effects [Wilson+14] (used in Disney’s Frozen)

---

## Dependencies

### Third-party libraries
- **GLFW**: window / input
- **Dear ImGui**: debug UI (vendored; built via `add_subdirectory`)
- **KTX-Software** (`KTX::ktx`): KTX/KTX2 texture container support
- **vk_radix_sort**: GPU radix sort (vendored; built via `add_subdirectory`)
- **fmt** (`fmt::fmt-header-only`): logging / formatting

### How dependencies are included
- Vendored in this repository (git submodule):
  - `external/imgui`
  - `external/vk_radix_sort`
- Fetched via package manager (recommended: **vcpkg**):
  - GLFW, KTX-Software, fmt
  
> Note : Check out [THIRD_PARTY_NOTICES.md](./docs/THIRD_PARTY_NOTICES.md)

---

## Quick Start

### Requirements
- Windows
- CMake >= 3.29
- C++ 20
- GPU backend: Vulkan 1.4+ (LunarG SDK)
- Visual Studio 2022 (MSVC) + C++ Desktop workload
- vcpkg (set environment variable `VCPKG_ROOT` to your vcpkg directory, e.g. `C:\vcpkg`)

### Build (CMake)
> Run from **x64 Native Tools Command Prompt for VS 2022**.

```bat
git clone https://github.com/steampower33/PowerEngine.git
cd PowerEngine
git submodule update --init --recursive

:: Release
cmake --preset windows-release
cmake --build --preset release
release.bat

:: Debug
cmake --preset windows-debug
cmake --build --preset debug
debug.bat
```

---

## Controls

* **W/A/S/D**: Move the camera
* **X**: Pause/Resume simulation
* **Z + X**: After pressing **Z**, pressing **X** steps the simulation **one frame at a time** (frame stepping)
* **F**: Focus the camera on the world origin
* **R**: Toggle mouse-driven camera control (enable/disable view rotation by mouse)
* **Right Mouse Button (RMB)**: Move objects *(particle-based objects are not supported yet)*
* **Left Mouse Button (LMB)**: Rotate objects
* **LMB (Particle interaction)**: Drag particles with the mouse
* **ESC**: Quit the application

---

## Docs
- [docs/README.md](./docs/README.md)


---

## References
- [docs/REFERENCES.md](./docs/REFERENCES.md)

---

## Acknowledgements
- README structure inspired by: [Velvet](https://github.com/vitalight/Velvet/tree/master) (vitalight/Velvet), [elasty](https://github.com/yuki-koyama/elasty?tab=readme-ov-file) (yuki-koyama/elasty)
- The timestamp gui code is referenced from the [Velvet](https://github.com/vitalight/Velvet/tree/master) project.
- Implementation notes referenced: Ten Minute Physics / PBD tutorial notes (Matthias Müller)
- Vulkan learning was done with [Khronos Vulkan](https://docs.vulkan.org/tutorial/latest/00_Introduction.html).