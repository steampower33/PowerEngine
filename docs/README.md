
## Simulation

PowerEngine runs the XPBD simulation entirely on GPU compute shaders.  
The simulation is structured as **Substeps × Iterations**, with additional passes for self-collision and SDF-based collisions.

### High-level loop
```txt
for each frame:
  Reset scene / Copy colliders
  for substep in [0..substeps):
    Wind
    Integrate (predict positions)
    Clear lambdas / accumulators
    if substep % broadphase_interval == 0:
      Broadphase: build_hash → radix sort(vk_radix_sort) → build cell ranges(start/end) → build neighbor list
    for iter in [0..iterations):
      Solve constraints (stretch uses coloring-GS, others use atomic accumulation)
      Apply accumulated position deltas
      Clear accumulators (per-iteration)
    Collide with SDF
    Update velocity
  Recompute normals (tri → vertex)
```

### Implementation Details

- Wind

  - Implements a lift/drag force model inspired by the SIGGRAPH 2014 talk used in Disney's *Frozen* ([Winson+14], see `docs/references.md`).
  - Samples a wind **velocity field** `u_air(x,t)` and computes relative velocity `rel = u_air - u_tri` per triangle.
  - Accumulates per-vertex **delta velocity (Δv)** using atomic operations.

- Integrate (Prediction)

  - Applies user interaction (mouse pick / drag) as an external input (force/impulse).
  - Applies accumulated Δv from the wind pass.
  - Computes predicted positions `xp = x + dt * v`.

- Clear lambdas / accumulators

  - Resets per-constraint XPBD lambdas (λ) for the current substep.
  - Clears per-vertex accumulators used for parallel constraint solving (e.g., `delta_x/y/z` and `delta_count`).

- Broadphase (Self-collision neighbor search)

  - Builds a spatial hash to find nearby particles.
  - Uses radix sort from `vk_radix_sort` to sort keys and build cell ranges.
  - Generates a neighbor list for narrowphase / constraint evaluation.

- Solve constraints
  - Stretch (Edge)
    - Enforces distance constraints on mesh edges.

  - Shear
    - Enforces shear behavior on cloth quads (implementation-dependent: angle/diagonal-based).

  - Bend
    - Enforces dihedral bending angle constraints between adjacent triangles.

  - Area
    - Enforces triangle area constraints.

  - Self Collision
    - Resolves cloth self-collision constraints.
    - Multiple cloth instances may share the same SSBO; cross-object collisions are currently handled in the same pass for more natural interaction.
    - Approximated Coulomb friction is applied.

- Apply accumulated position deltas
  - Applies the accumulated corrections to predicted positions.
  - The accumulated delta is **averaged** (e.g., divided by `delta_count`) and then scaled by a relaxation factor (0.0 ~ 1.0).

- Collide with SDF
  - Applies SDF-based collisions for primitive colliders (sphere / plane / capsule).
  - Approximated Coulomb friction is applied.

- Update velocity
  - Updates `v` from the corrected predicted positions.
  - Applies max-velocity clamping.
  - Applies global velocity damping (document the exact formula and parameter range used in this project).

- Recompute normals
  - Recomputes per-vertex normals from triangle geometry after simulation.
  - Used for lighting in the rendering pass and for debugging visualization.

---

## Rendering

The renderer is primarily for **simulation visualization & debugging** (not a full game renderer).

### Features

- [x] Vulkan rendering backend
- [x] KTX/KTX2 texture loading (KTX::ktx)
- [x] ImGui debug UI
- [ ] GPU skinning / animation (if any)

### Lighting & Materials

- [x] Deferred Shading
- [x] OpenPBR material model (WIP / partial)

  - [x] baseColor / metalness / roughness
  - [x] normal / emissive
  - [x] coat / fuzz
- [x] IBL (environment lighting)
- [x] Tone mapping / exposure control

### Shadows

- [x] Shadow mapping (spot light)

  - [x] PCF filtering
  - [ ] Cascaded Shadow Maps (CSM)
