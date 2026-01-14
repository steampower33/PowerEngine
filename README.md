# PhysixStudio

> GPU-accelerated particle simulation engine based on **Extended Position Based Dynamics (XPBD)**.  
> Real-time oriented cloth particle system built on a unified XPBD solver.

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
- [x] Bend (A) [Müller+07]
- [x] Area (A) [Müller+14]
- [x] Self-Collision (A)
- [x] LRA(Long Range Attachments) (per-particle sequential projection; GS-like) [Kim+12]

### External Force
- [x] Wind effects [Wilson+14] (used in Disney’s Frozen)

---
## Technical Deep Dive

### 1. Stability Analysis: The "Jittering Cloth" Problem
During the development of the physics solver, a severe instability issue was observed where the cloth would vibrate uncontrollably even in a resting state. This occurred specifically when only Distance (Stretch) and Self-Collision constraints were active.

* **Problem:** Without diagonal constraints, a grid-based cloth mesh suffers from structural instability (**Zero-Energy Modes**). A quad element can easily distort into a rhombus without violating the edge length constraints. This lack of rigidity caused a conflict between the Self-Collision (pushing particles apart) and Stretch (pulling them together), leading to an infinite feedback loop of jitter.
* **Solution:** I implemented **Shear Constraints** by adding distance constraints across the diagonals of each quad.
* **Result:** This effectively triangulates the mesh, locking the internal angles of the quads. The system gained significant structural rigidity, and the jittering was completely eliminated, proving that shear constraints are essential not just for material properties but for the numerical stability of the solver.

### 2. Parallel Constraint Solving (Jacobi-style)
To maximize GPU parallelism using Compute Shaders, I adopted a **Jacobi-style accumulation** scheme for constraints like Shear, Bend, and Self-Collision.

Instead of sequentially projecting positions (Gauss-Seidel), which is hard to parallelize, each constraint calculates a position correction vector and adds it to a global accumulator using `atomicAdd`.

* **Trade-off:** While this allows massive parallelism, it introduces non-determinism due to the order of atomic operations.
* **Mitigation:** To mitigate stability issues often associated with Jacobi solvers, I utilized **Small-steps XPBD**, which performs multiple sub-steps per frame to ensure convergence.

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
git clone https://github.com/steampower33/PhysixStudio.git
cd PhysixStudio
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

## References

### Papers / Talks
- [Müller+07] Müller et al., *Position Based Dynamics*, 2007. DOI: https://doi.org/10.1016/j.jvcir.2007.01.005
- [Müller+14] Müller et al., *Strain Based Dynamics*, 2014. DOI: https://doi.org/10.2312/sca.20141133
- [Macklin+16] Macklin et al., *XPBD: Position-Based Simulation of Compliant Constrained Dynamics*, 2016. DOI: https://doi.org/10.1145/2994258.2994272
- [Macklin+19] Macklin et al., *Small Steps in Physics Simulation*, 2019. DOI: https://dl.acm.org/doi/10.1145/3309486.3340247
- [Wilson+14] Wilson et al., *Simulating Wind Effects on Cloth and Hair in Disney's Frozen*, SIGGRAPH 2014 Talks. DOI: https://doi.org/10.1145/2614106.2614120
- [Kim+12] Kim et al. *Long range attachments - a method to simulate inextensible clothing in computer games*, 2012. DOI: https://dl.acm.org/doi/10.5555/2422356.2422399

### Slides / Notes
- UCSD CSE169: https://cseweb.ucsd.edu/classes/wi17/cse169-a/slides/CSE169_11.pdf
- Ten Minute Physics (PBD notes): https://matthias-research.github.io/pages/tenMinutePhysics/index.html

### Rendering / API
- OpenPBR: https://github.com/AcademySoftwareFoundation/OpenPBR
- Khronos Vulkan Tutorial: https://docs.vulkan.org/tutorial/latest/00_Introduction.html


---

## Acknowledgements
- README structure inspired by: [Velvet](https://github.com/vitalight/Velvet/tree/master) (vitalight/Velvet), [elasty](https://github.com/yuki-koyama/elasty?tab=readme-ov-file) (yuki-koyama/elasty)
- The timestamp gui code is referenced from the [Velvet](https://github.com/vitalight/Velvet/tree/master) project.
- Implementation notes referenced: Ten Minute Physics / PBD tutorial notes (Matthias Müller)
- Vulkan learning was done with [Khronos Vulkan](https://docs.vulkan.org/tutorial/latest/00_Introduction.html).