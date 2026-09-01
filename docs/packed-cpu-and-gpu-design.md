# Packed-Y texture rotation search

This document explains the experimental packed CPU search, why it is fast, how its loop
organization differs from a natural GPU implementation, and how to move the idea onto the
GPU without introducing shared writes.

The implementation discussed here is intentionally a narrow experiment. It currently targets
the common Vanilla-3 case with origin Y in `-60..0`, observations at `dy = 0` or `dy = 1`, no
allowed errors, and observations that can be expressed as complete visible-rotation groups.
Unsupported inputs continue through the general CPU scanner.

Reference implementation and discussion:
[Globexix/texture-rotation-finder](https://github.com/Globexix/texture-rotation-finder).

### Prototype status

The first packed GPU prototype is now implemented in `src/packed_search.wgsl` and selected from
`src/gpu.rs` when the narrow fast-path contract is satisfied. It has two compute entry points:

1. `generate_source_signatures` writes rotation-major masks for the selected source Z rows.
2. `filter_candidates` gives one invocation ownership of one candidate `(x,z)`, intersects its
   private `live` mask, and exactly verifies surviving Y bits.

The prototype currently supports only an automatically selected dense source-Z modulus. Inputs
that require the general six-of-32 sparse cover continue through the original GPU shader. Source
data is processed in 1024-candidate-Z bands to keep the storage buffer bounded while avoiding an
excessive number of dispatches.

On an NVIDIA GeForce RTX 5080 Laptop GPU, the retained optimized shader scans the 2.44-trillion-
position `examples/packed-gpu-benchmark.conf` in approximately **4.03 seconds** (about **605
billion positions/second**). Forcing the original shader takes approximately **14.36 seconds**
(about **170 billion positions/second**), making the packed path roughly **3.6x faster** for this
pattern. The shorter 6.1-billion-position benchmark is too brief for a stable comparison because
initialization, dispatch, and readback are a substantial fraction of its reported time. These
numbers are an initial result, not a cross-device performance claim.

Two candidate-kernel details were important to the improvement:

- observations are partitioned at cold start by candidate-Z residue, so a stride-four candidate
  loops over only its 5--8 applicable masks instead of testing all 28 observations; and
- a single-visible-rotation observation directly indexes its source plane instead of executing a
  four-plane union loop.

## 1. Vocabulary

It is easiest to understand the data flow when the three kinds of coordinates have distinct
names:

- A **candidate origin** is the position being tested: `(cx, cy, cz)`.
- An **observation** is a constraint relative to that origin:
  `(dx, dy, dz, accepted rotations)`.
- The **source position** sampled by that observation is:
  `(cx + dx, cy + dy, cz + dz)`.

A **candidate column** fixes `(cx, cz)` and varies `cy`. A **source column** fixes
`(source_x, source_z)` and varies `source_y`.

These columns have different bitfields:

- `live[candidate_z][candidate_x]` says which origin Y values are still possible.
- `source_masks[source_x][rotation]` says which source Y values produce one rotation at a
  particular source Z.

Keeping those meanings separate avoids the most common misunderstanding of the algorithm.

## 2. Packing 61 candidate Y values into one `u64`

For a candidate `(x,z)`, bits `0..60` represent origin Y values `-60..0`:

```text
bit 0  -> origin Y -60
bit 1  -> origin Y -59
...
bit 60 -> origin Y   0
```

At the start, every in-range bit is set:

```text
live = 0b000...0011111111111111111111111111111111111111111111111111111111111
```

For one source `(x,z)`, the signature generator produces four `u64` masks, one for each
visible rotation. A bit in `source_masks[rotation]` is set exactly when that source Y produces
the requested rotation.

An observation that requires rotation 2 is applied with one operation per candidate X:

```rust,ignore
live[candidate_x] &= source_masks[source_x][2] >> observation.dy;
```

The shift aligns source Y with origin Y. If `dy = 1`, source Y `-59` must constrain origin Y
`-60`, so the source mask is shifted down by one bit.

If an observation accepts more than one visible rotation, its rotation masks are ORed first,
then ANDed into `live`.

After a few independent observations, most candidate columns become zero. One zero rejects
all 61 origin Y values in that `(x,z)` column.

## 3. Current CPU pipeline

The current CPU path is **source-row oriented**. It processes the search in 64-candidate-Z
bands:

```text
for each 64-row candidate Z band
    allocate one live u64 for every candidate (x,z) column in the band
    make a cold plan grouping useful observation applications by source Z

    for each selected source Z
        generate all source-X signatures for this one Z row
        immediately push those masks into every affected candidate live row
        discard/overwrite the source row

    enumerate surviving Y bits
    verify every survivor with the exact rotation function
```

The planning step is outside the arithmetic-heavy X loops. It creates a
`BTreeMap<source_z, Vec<MaskTask>>`. A `MaskTask` contains only:

- the candidate row in the current band to update; and
- the observation whose `dx`, `dy`, and accepted rotations should be applied.

It does **not** store work for every candidate X. One task applies an observation across the
entire candidate-X row in a tight loop.

This is why `live` is allocated as:

```rust,ignore
let mut live = vec![layout.initial_live; band_rows * layout.candidate_width];
```

There is one `u64`, not one vector, for every candidate `(x,z)` column. A generated source row
is temporary. Several tasks may read it and update different rows of this persistent `live`
array before the source-row buffer is reused.

### What a Z band means

A Z band is only a cache-sized chunk of candidate rows. With `Z_BAND = 64`, a work tile is
processed as candidate Z rows `[start, start + 64)`, followed by the next 64 rows.

It is not a modulus and does not remove candidates. Every candidate Z in the requested range
is processed. A source Z used by an observation may lie outside the current candidate band
because `source_z = candidate_z + dz`.

The band bounds the live working set. For a candidate width of 1024, its live array is
`64 * 1024 * 8 = 512 KiB`.

### `TileLayout`

`TileLayout` stores values that are constant while one worker scans a work item:

- `initial_live`: the valid origin-Y bits for the configured Y range;
- `candidate_width`: the number of candidate X values;
- `source_x_start`: the left edge including the most negative observation `dx`;
- `source_width`: candidate width plus the X halo needed by all `dx` offsets; and
- `dense_z_stride`: the automatically selected single-pass source-Z modulus, or zero for the
  general sparse cover.

The X halo is necessary because a candidate at either edge may observe a source block beyond
the candidate rectangle.

## 4. The constants

### `SIGNATURE_BATCH_SIZE = 8`

The recurrence for one source column is a dependency chain: the next value depends on the
previous value and difference. Generating eight independent X columns together exposes eight
chains to the optimizer and CPU execution units. It does not change the result or pack eight X
positions into one mask; every lane still produces its own four Y masks.

Eight is an empirical implementation choice, not an algorithmic requirement. It should be
benchmarked if the generator or target CPU changes.

### `MIN_MASKS_PER_CANDIDATE_ROW = 5`

When selecting a simple source-Z modulus, every possible candidate-Z residue must receive at
least this many observations. Five is a rejection-quality heuristic: too few masks leave many
survivors for expensive exact verification.

It is not a correctness threshold. Applied masks are exact and omitted observations are checked
later, so choosing four or six would change performance rather than valid results.

### `SPARSE_RESIDUES = [2, 30, 8, 24, 13, 19]`

If no dense modulus gives every candidate residue enough observations, the implementation uses:

```text
source_z mod 32 in {+2, -2, +8, -8, +13, -13}
```

The array uses non-negative Euclidean residues, so `30`, `24`, and `19` mean `-2`, `-8`, and
`-13` modulo 32. It generates 6 of every 32 source rows, a `32 / 6 = 5.33x` reduction.

This cover is deliberately symmetric so reflected patterns have similar coverage.

### `Z_BAND = 64`

As described above, this is the number of candidate Z rows whose live masks are held and
updated together. It is independent of the selected source-Z modulus and the modulo-32
fallback.

## 5. How the source-Z cover reaches every candidate Z

Only source rows are sparse. Candidate rows are not.

Suppose the selected cover is:

```text
source_z mod 4 == 2
```

For an observation with offset `dz`, a candidate row can use it when:

```text
(candidate_z + dz) mod 4 == 2
```

Equivalently, each candidate-Z residue uses the observations whose `dz` residue complements it:

```text
candidate_z mod 4 == (2 - dz) mod 4
```

The planner computes this relationship for the current band. It groups the resulting tasks by
the selected `source_z`. The hot phase then generates that source row once and applies all of
its tasks.

Every candidate Z still has a live row. It simply receives a different subset of observations
according to its residue. `select_dense_stride` accepts a modulus only if every `dz` residue
class has at least five observations; that guarantees every candidate-Z residue gets at least
five useful masks.

If no such modulus exists, the six-residue modulo-32 cover is used. Some candidate rows may
receive fewer masks, but exact verification still preserves correctness.

## 6. Removing repeated quadratic mixer multiplications

Minecraft first forms the coordinate seed:

```text
base = (x * 3,129,871) XOR (z * 116,129,781)
seed = base XOR y
```

For a fixed source `(x,z)`, `base` is constant. The expensive quadratic mixer is:

```text
f(seed) = seed * (seed * M + 11)
M = 42,317,861
```

Naively, every Y performs the two multiplications inside `f`. But this is a quadratic
polynomial, so consecutive inputs have a finite-difference recurrence:

```text
D(seed)       = f(seed + 1) - f(seed)
              = M * (2 * seed + 1) + 11

D(seed + 1)   = D(seed) + 2*M
f(seed + 1)   = f(seed) + D(seed)
```

For each source column, the implementation computes `f(first_seed)` and its initial difference
once. The next 63 mixed values use wrapping additions only. The recurrence is exact modulo
`2^64`; it is not an approximation.

The apparent problem is that `base XOR y` is not numerically consecutive as Y changes. For the
negative range `-64..-1`, however, all seeds share the same high 58 bits and contain every
possible six-bit low value exactly once. XOR merely permutes those 64 values. The generator:

1. generates the 64 consecutive low-bit values with the recurrence;
2. places each result at bit `low XOR (base & 63)`; and
3. shifts away Y `-64..-61`, leaving mask bit 0 aligned to Y `-60`.

Y `0` and `1` do not belong to that negative-Y prefix. They are evaluated directly and inserted
as bits 60 and 61.

The quadratic mixer therefore costs only its initialization multiplications per source column,
not two mixer multiplications for every Y. The later Java LCG step still has one 64-bit
multiplication per generated Y value. So the precise statement is:

> The recurrence removes the repeated quadratic-mixer multiplications; it does not reduce the
> entire rotation calculation to one multiplication per source location.

## 7. CPU loop organizations and the measured choice

There are three useful organizations of the same mask math.

### A. Source-row streaming (current CPU code)

```text
generate one source row
    -> apply it to several candidate live rows
```

Advantages:

- each expensive source signature is generated once;
- the current source row is reused immediately and stays hot in cache;
- only one source row is stored; and
- applying one task is a simple contiguous X loop.

Costs:

- candidate live rows are written multiple times;
- the live band must stay resident; and
- task planning and indexing add some overhead.

### B. Stored source grid, then candidate-row/observation bulk loops

```text
generate all selected source rows into source[rotation][z][x]
for each candidate Z row
    for each applicable observation
        live[x] &= source[rotation][source_z][source_x]
```

This is close to the organization in the reference implementation. It is conceptually clean
and maps naturally to GPU memory access. It also gives each candidate live row more temporal
locality.

Its CPU cost is a larger source-grid working set. A local A/B implementation in this repository
was correct but took about **2.44 seconds** on the 6.1-billion-position benchmark. After restoring
the source-row-streaming form, the same release build completed in **1.94 seconds**. These are
single-machine, load-sensitive measurements, not universal constants, but the regression was
large enough that the rewrite was not retained.

The likely reason is that source streaming repeatedly reads one roughly 32-KiB signature row,
whereas the stored-grid version pulls masks from a substantially larger sparse grid. The source
reuse benefit outweighed the candidate-row write locality on this CPU and workload.

### C. Strict candidate ownership

```text
one worker owns one candidate or a small SIMD batch
    -> pulls all required source masks
    -> keeps live in registers
```

This avoids repeated shared writes to `live`, but it still needs a stored source grid and may
reload source masks across candidate owners. It remains worth testing on CPU as an 8- or
16-candidate SIMD kernel, especially if vectorization can offset the larger source working set.

### Current decision

Do not change the current packed CPU loop solely to resemble the planned GPU kernel. The source-
oriented CPU implementation already expresses the packed-Y and sparse-source ideas and is faster
in the measured A/B test. CPU and GPU should share mathematical invariants, not necessarily loop
ownership.

## 8. GPU translation without write races

The GPU should use two compute phases and candidate ownership.

### Phase A: generate source signatures

Generate only the source Z rows selected by the cover and store them in a rotation-major
structure-of-arrays layout:

```text
source[rotation][selected_z_index][source_x]
```

Keep X as the fastest-changing index. Adjacent invocations generating adjacent X positions then
write adjacent addresses.

### Phase B: filter candidate columns

One invocation owns one candidate `(x,z)` column:

```wgsl,ignore
var live: u64 = initial_live;
for each observation applicable to this candidate-z residue {
    let mask = source[rotation][source_z_index][source_x];
    live &= mask >> dy;
    if (live == 0u) { break; }
}
```

`live` stays private to that invocation, normally in registers. No other invocation writes it,
so there are no parallel overwrite races and no atomics in the rejection loop.

The dispatch should group adjacent X candidates together and avoid having a warp/wave cross a
candidate-Z-row boundary. Threads in the same warp then:

- have the same candidate Z residue;
- iterate the same observation bucket in the same order;
- request the same rotation plane and selected source Z; and
- read adjacent source X addresses.

Under those conditions the source loads are coalesced. “Perfectly 100% coalesced” is too strong
as a blanket promise: `dx` changes the starting alignment, edge lanes may cross memory segments,
and early exit can diverge. The layout is nevertheless the right one for coalescing.

A storage barrier is required between the signature-generation dispatch and the candidate-filter
dispatch. Survivors are rare, so they can be appended to an output buffer with an atomic counter
or compacted in a separate pass.

## 9. Why the GPU speedup will not automatically equal the CPU speedup

The packed algorithm reduces work on both processors, but the baseline bottlenecks differ.

The old CPU scanner pays heavily for scalar per-Y coordinate mixing, branches, and loop overhead.
Packing 61 Y candidates into one word attacks exactly those costs. A GPU already hides latency
with many invocations and may be limited by different resources: 64-bit integer throughput,
register pressure, source-grid bandwidth, occupancy, or divergence.

For a rough arithmetic comparison, a source column in a modulo-4 cover needs about 16 Java LCG
steps for the negative-Y recurrence on average, plus initialization and direct Y `0/1` work.
Each candidate column then performs roughly 5--8 source loads and ANDs before most candidates die.
That is dramatically less arithmetic than evaluating several observations independently for up
to 61 Y candidates, but it does not by itself prove a 40x end-to-end GPU gain.

The right question is therefore not “does the CPU multiplier transfer?” but:

- how many source columns are generated per candidate column;
- how many mask loads occur before `live == 0`;
- how often exact verification runs;
- whether source loads are coalesced and cacheable;
- whether `u64` arithmetic lowers occupancy or throughput; and
- how much time each compute phase takes on the target GPU.

## 10. Correctness invariants

Any CPU or GPU rewrite should preserve these rules:

1. Every applied mask must exactly represent the real rotation function for its source
   coordinates and source Y bits.
2. `dy` alignment must map source Y back to the correct origin-Y bit.
3. Omitting an observation from the mask pass may create false positives, but must never clear a
   genuine match.
4. Every surviving candidate must be checked against all compiled observations with the exact
   sampler before it is reported.
5. Coordinate addition and multiplication must match Java/Minecraft wrapping and sign-extension
   behavior.
6. Unsupported algorithms, Y ranges, error tolerance, forced errors, partial rotation groups,
   or observation heights must not silently use the narrow packed path.

The existing generated-mask unit test compares every packed source-Y bit with the authoritative
Vanilla-3 sampler at positive, negative, and large coordinates. The higher-level differential
test compares packed and brute-force searches across both dense and sparse covers.

## 11. Recommended roadmap

### Step 1: instrument the existing CPU experiment

Collect counters, outside the tight loops when possible:

- generated source rows and columns;
- mask applications;
- masks applied before each column dies;
- columns surviving the mask pass;
- total surviving Y bits; and
- exact sampler evaluations.

These numbers establish how much useful work a GPU kernel would perform.

### Step 2: build a deliberately narrow GPU prototype (completed for the dense cover)

The first implementation uses the same narrow fast-path contract as the CPU code. It includes:

1. CPU preparation of the source-Z cover and packed observations;
2. a source-signature compute shader;
3. a candidate-owned mask shader; and
4. exact verification of survivors, either at the end of the candidate shader or in a third
   dispatch.

The candidate shader currently loops over the small observation array and selects observations
for its Z residue. Pre-bucketing those observations is a possible measured optimization rather
than a correctness requirement.

### Step 3: differential correctness tests

On small ranges, compare:

- the ordinary CPU scanner;
- the packed CPU scanner; and
- the packed GPU prototype.

Include negative coordinates, tile edges, source halos, partial final Z bands, every candidate-Z
residue, `dy = 0/1`, and cases with real matches.

### Step 4: measure phases separately

Use GPU timestamp queries for signature generation, candidate filtering, survivor verification,
and readback. Report positions per second together with source columns per second and candidate
columns per second; the latter two make comparisons across Y-range sizes meaningful.

### Step 5: tune only measured bottlenecks

Likely experiments include:

- workgroup width and the mapping of workgroups to candidate Z rows;
- rotation-major versus packed-four-mask source storage;
- source Z band size;
- observation order, placing selective masks first;
- caching sparse source rows shared by adjacent bands;
- mirrored `(x,z)` signature reuse where the exact mixer symmetry permits it; and
- an explicitly vectorized candidate-owned CPU kernel for a fair third comparison.

### Step 6: decide whether to generalize

Only after measuring the prototype should the packed path gain wider Y ranges, more algorithms,
error tolerance, or complicated observation masks. Those features can substantially change the
best representation and should not obscure the initial performance result.

## 12. Reproducing the CPU experiment

Run correctness checks:

```powershell
cargo test --lib
cargo clippy --all-targets -- -D warnings
```

Build and run the packed benchmark with one CPU thread:

```powershell
cargo build --release
target/release/coordsfinder.exe --backend cpu --threads 1 examples/packed-cpu-benchmark.conf
```

Run the same configuration through the general scanner for an A/B comparison:

```powershell
$env:COORDSFINDER_DISABLE_PACKED_CPU = "1"
target/release/coordsfinder.exe --backend cpu --threads 1 examples/packed-cpu-benchmark.conf
Remove-Item Env:COORDSFINDER_DISABLE_PACKED_CPU
```

The equivalent GPU control is:

```powershell
$env:COORDSFINDER_DISABLE_PACKED_GPU = "1"
target/release/coordsfinder.exe --backend gpu examples/packed-cpu-benchmark.conf
Remove-Item Env:COORDSFINDER_DISABLE_PACKED_GPU
```

Benchmark results are sensitive to CPU frequency, background load, compiler version, and thermal
state. Compare variants in alternating runs on the same machine rather than treating one timing
as a permanent constant.

For GPU work, prefer the longer benchmark:

```powershell
target/release/coordsfinder.exe --backend gpu examples/packed-gpu-benchmark.conf
$env:COORDSFINDER_DISABLE_PACKED_GPU = "1"
target/release/coordsfinder.exe --backend gpu examples/packed-gpu-benchmark.conf
Remove-Item Env:COORDSFINDER_DISABLE_PACKED_GPU
```
