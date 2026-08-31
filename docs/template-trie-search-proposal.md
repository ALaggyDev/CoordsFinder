# Compact-template trie search proposal

Status: the first exact-match implementation described below is complete on
`dev-optimizer`. Later sections retain the original design rationale and
possible follow-up work.

This proposal adapts the square-template algorithm from [Two and Higher
Dimensional Pattern Matching in Optimal Expected Time](../314464.314680.pdf)
to CoordsFinder's exact texture-rotation search. The paper presents the
algorithm in two dimensions and states the generalization to arbitrary fixed
dimensions; this document specializes that generalization to CoordsFinder's
three-dimensional coordinates and GPU workload.

## Decision summary

The idea is worth implementing as an experimental exact-match path, with the
current candidate-at-a-time scanner retained as a fallback.

The recommended first experiment is:

1. Find a small, compact group of ordinary four-way constraints that occurs at
   every position in a rectangular box inside the sparse filter.
2. Build a flat trie of the sample words at those positions.
3. Evaluate sample groups only on a coarse three-dimensional world lattice.
4. When a trie leaf matches, derive the candidate coordinate and immediately
   run the remaining filter checks in the same shader invocation.
5. Use the existing scanner when no profitable sample plan exists, or when
   `errorTolerance` is nonzero.

This requires no generated world-text buffer and no survivor/candidate buffer.
The only new GPU data is static preprocessing data: the sample shape, flat
trie, and trie-leaf addresses. Results continue to use the existing result
buffer.

The important qualification is that a trie is a good first experiment, not a
guaranteed GPU winner. For the small sample sizes available here, a packed
base-4 lookup table may beat a trie by avoiding dependent memory reads. The
trie can compensate by rejecting a missing prefix before calculating the rest
of the sample. Both encodings can use the same planner and fused search kernel,
so a later microbenchmark can settle that choice without redesigning the
algorithm.

## Implemented first experiment

The implementation now includes:

- [`src/sample_plan.rs`](../src/sample_plan.rs), which enumerates compact
  shapes of one to four offsets, finds a solid placement box, optimizes the
  trie traversal order, builds the flat trie, and estimates expected texture
  work;
- an exact CPU reference/optimized path that directly traverses lattice
  anchors;
- the fused `search_sample_trie` GPU entry point and static trie/placement
  buffers;
- per-tile fallback at signed-integer halo boundaries;
- `searchMode = auto|naive|sample-trie` and the matching `--search-mode`
  override;
- startup and validation output showing `q`, sample offsets, placement
  dimensions, `V`, trie nodes/leaves/outputs, and expected texture checks for
  every direction (including plans rejected by `auto`);
- equivalence tests across all five texture algorithms, all four directions,
  negative coordinates, multiple tile halos, duplicate trie leaves, and GPU
  execution.

`auto` requires a hash-work estimate of at least 5.0. This intentionally wide
margin reflects GPU invocation, control, and dependent-read overhead that the
estimate does not count. A nonzero error tolerance uses the naive loop. The
trie samples canonical four-way masks; other exact
constraints remain in the fused survivor check. A forced `sample-trie` mode
reports an error when it cannot build a plan.

### Initial measurements

These are single-run implementation checks of the original, Y-batched GPU
prototype on the README's NVIDIA RTX 5080 Laptop system, not a stable
benchmark suite:

| Workload | Naive | Sample trie | Observed gain |
| --- | ---: | ---: | ---: |
| Current irregular 28-filter benchmark geometry, 600 billion candidates | 3.52 s | 3.12 s | 1.13x |
| Dense 42-filter planar patch, 600 billion candidates | 3.55 s | 0.77 s | 4.61x |
| `Dark Cave.conf`, one CPU thread, 960 million candidates | 6.60 s | 2.31 s | 2.86x |

Both modes found the same known coordinates. The irregular GPU result is below
the proposal's hoped-for 20% threshold but is still a measurable improvement;
the dense result confirms that placement volume is the controlling factor.
The static model deliberately remains a plan-ranking heuristic: it predicted
3.08x for the irregular plan (`q=2`, `V=5`) and 12.31x for the dense plan
(`q=4`, `V=30`), so it does not model GPU trie/control cost accurately enough
to be presented as a throughput forecast.

### Why the irregular filter cannot gain 10x

For a canonical four-way exact constraint, the naive loop needs about
`1 + 1/4 + 1/16 + ... = 4/3` texture hashes per candidate because it normally
rejects early. The irregular benchmark's best compact template has `q=2` and
only `V=5` reusable placements. Even if trie traversal and survivor checks
were free, sampling therefore costs at least `q/V = 2/5` hashes per candidate.
Its absolute hash-work ceiling is only `(4/3) / (2/5) = 3.33x`; the more
complete planner estimate is 3.08x. A 10x result is not available from this
template geometry.

The dense patch is qualitatively different: `q=4`, `V=30` gives a full-sample
floor of `4/30` hashes per candidate, which is where a roughly 10x theoretical
gain becomes plausible. This is why the paper's dense rectangular-pattern
result does not transfer directly to a sparse filter with 28 scattered
constraints.

### GPU follow-up

The sample kernel no longer serially batches 32 Y anchors in one invocation.
It now uses `8 x 4 x 8` workgroups and one independent anchor per invocation.
This change makes the mapping genuinely three-dimensional, but it is a
performance loss on the irregular benchmark: the large X/Z range already
provided ample occupancy, while removing batching increased the number of
shader invocations by roughly 32x. Diagnostic runs over 600 billion candidates
put the independent-anchor sparse kernel around 5.3-7.0 seconds, versus
roughly 3.6-4.9 seconds for naive on the same machine. The dense sample still
benefited, running around 1.2-1.9 seconds in the same investigation.

Moving the small trie from read-only storage to uniform memory was also tested
and rejected: divergent dependent indices serialized badly and made both
workloads slower. Packing nodes from 32 to 16 bytes showed no reliable gain.
Those experiments were reverted.

Consequently, `auto` now declines the irregular 3.08x hash-work plan and uses
the naive GPU kernel. Forced `sample-trie` remains useful for experiments, and
dense plans above the 5.0 cutoff still select the trie. A future attempt at a
large gain on sparse filters must first increase placement reuse—for example,
with a proven non-rectangular or multi-template cover—rather than micro-tuning
the current five-placement trie.

## The paper in CoordsFinder terms

The paper uses a negative search strategy:

- Preprocess many short sample words from the pattern.
- Read the corresponding short words at a static set of locations in the
  larger text.
- Discard every candidate whose short word is absent from the pattern
  dictionary.
- Fully check only the candidates that survive.

For a dense pattern of size `|P|` over a uniform alphabet of size `c`, a sample
length near `log_c(|P|)` makes a random sample selective enough that very few
candidates reach the full check. A compact ("square" in two dimensions)
sample shape maximizes the number of positions where that shape fits inside
the pattern. The resulting expected text-processing bound is
`O(|T| / |P| * log_c(|P|))` under the paper's independent uniform-text model.

The terminology maps as follows:

| Paper | Term used here |
| --- | --- |
| pattern `P` | compiled texture filter |
| text `T` | implicit world texture field |
| template `Q` | compact sample shape |
| pattern sample origin | sample placement in the filter |
| text sampling scheme | world sample lattice |
| test string | sample word |
| test dictionary | flat trie |
| elimination phase | sample rejection |
| checking phase | remaining filter check |
| positive witness | a sample word that reaches a trie leaf |

Two observations make the approach especially relevant here:

- World values are procedural, so reading a text entry means running an
  expensive texture hash rather than loading an already stored byte.
- The paper's static sampling schedule is GPU-friendly: one sample does not
  depend on the result of another sample.

The paper's expected-time theorem does **not** transfer directly. Minecraft's
coordinate functions are deterministic and may have local correlations, the
filter is sparse, constraints are acceptance masks rather than literal
characters, and GPU control/memory costs are not unit-cost character
comparisons. The paper supplies the structure; this project still needs an
empirical cost model and benchmarks.

## Current search and the opportunity

Both [`src/cpu.rs`](../src/cpu.rs) and
[`src/search.wgsl`](../src/search.wgsl) currently start from a candidate
coordinate and test compiled constraints until the first failure. For an exact
ordinary four-way filter and approximately uniform values, a constraint passes
with probability `1/4`, so a rejected candidate needs about

```text
1 + 1/4 + 1/16 + ... = 4/3
```

texture calculations on average. This is already a strong early-exit loop.
The proposed method must therefore reduce the work below roughly 1.33 texture
calculations per candidate after including trie and survivor overhead.

The reuse comes from reversing the loop. One world sample word represents many
candidate alignments at once. A dictionary lookup returns only alignments whose
sample constraints all pass.

## Adapting rectangular patterns to sparse filters

Filling the filter's bounding box with wildcard cells is correct but usually
not useful. Wildcards contribute no rejection information, and a screenshot
can leave a large, mostly empty bounding box. The planner should operate on the
real constraint geometry instead.

Let:

- `F` be the set of usable compiled filter offsets;
- `Q` be a small ordered set of relative offsets, with `(0, 0, 0)` included;
- `P_Q` be all positions `p` for which every `p + delta`, `delta in Q`, exists
  in `F` and can be represented by the trie's alphabet.

In set notation, `P_Q` is the erosion of `F` by `Q`:

```text
P_Q = { p | p + Q is contained in F }
```

The key adaptation is to find an axis-aligned solid box `O` contained in
`P_Q`. The whole filter need not be rectangular. Only this box of valid sample
placements must be rectangular. All constraints outside the selected sample
placements remain useful in the full check.

Suppose `O` has consecutive side lengths `(Lx, Ly, Lz)` and volume
`V = Lx * Ly * Lz`. Sample world words only at anchors whose coordinates have
one fixed residue modulo `(Lx, Ly, Lz)`. For every candidate origin `r`, there
is exactly one `p` in `O` such that `r + p` lies on that lattice:

```text
anchor = r + p
```

Why coverage is exact: a consecutive interval of length `Lx` contains each
residue modulo `Lx` exactly once. Therefore one and only one `p.x` makes
`r.x + p.x` land on the X lattice. The same is true independently for Y and Z,
and their Cartesian product gives one unique three-dimensional `p`.
Because `O` is contained in `P_Q`, the corresponding sample is made entirely
of real filter constraints.

This is the paper's square-template lattice argument with the dense pattern
rectangle replaced by a rectangle of valid sample origins.

### Practical planner

For the first experiment, the planner can stay deliberately small:

1. Treat a constraint as sample-eligible when its 16-bit acceptance mask is
   exactly one of the four canonical ordinary four-way masks for the selected
   texture algorithm. Other constraints remain available for the full check.
2. Try sample lengths `q = 1..4`. Four is enough for an ordinary four-way
   alphabet and the current maximum of 256 compiled constraints because
   `log_4(256) = 4`.
3. Enumerate compact shapes inside a `2 x 2 x 2` cell, including planar and
   linear shapes. A shape is a set; its traversal order can be optimized
   separately for early trie rejection.
4. Compute `P_Q` with hash-set lookups, find solid boxes inside it, and score
   the resulting plans.
5. Select the lowest-cost plan only if it beats the current loop by a safety
   margin. Otherwise use the current scanner.

This search space contains only 64 shapes for `q = 1..4` when the origin is
fixed: `1 + 7 + 21 + 35`. With at most 256 filter constraints, preprocessing is
small compared with a coordinate scan.

A later planner can consider larger-diameter shapes or a modular residue cover
whose placements are not a solid box. Those generalizations may help filters
made of separated patches, but they are not needed to test the central idea.

### Planar filters are still three-dimensional searches

Many screenshot filters contain a dense X/Z patch at one Y level. Such a plan
will normally have `Ly = 1`, so it shares work across X and Z while sampling
every searched Y. A two-dimensional square embedded in 3D is the correct
compact shape for that data. Requiring an actual cube would reject the most
common useful case for no benefit.

### Tile boundaries

For one existing X/Z tile, enumerate lattice anchors over `tile + O`, including
the small halo introduced by the placement box. A trie output gives
`candidate = anchor - p`; discard it unless it lies inside the tile's half-open
candidate bounds. This preserves exactly-once candidate ownership between
tiles. Adjacent tiles may recompute a few halo anchors, but the overhead is a
surface term and should be negligible for the existing large GPU tiles.

Coordinate additions must retain the current wrapping Java-int semantics.
Boundary tests near `i32::MIN` and `i32::MAX` are required even if normal
Minecraft searches are far from those limits.

## Sample words and acceptance masks

The GPU currently calculates a model index from 0 to 15 and checks it against a
16-bit mask. For ordinary four-way constraints, those masks form four disjoint
classes:

- Vanilla-3 uses `index >> 2` as the four-way symbol.
- The other algorithms use `index & 3`.

Therefore an ordinary four-way sample slot has an effective alphabet of four
and exactly one deterministic trie edge for every model index. The trie should
store these visible symbols, not expand a four-bit acceptance mask into four
16-way branches.

Side constraints have an effective alphabet of two. They can remain in the
full check in the first experiment. Later they can be sample slots if every
placement at a given trie depth uses the same projection; a level can then use
two of the four child entries.

Arbitrary netherrack masks are different. Overlapping mask-labelled edges can
require following more than one trie branch, which is a poor fit for the
simple deterministic kernel. The first version should leave netherrack on the
current path. A future version can evaluate mask expansion, a mask decision
diagram, or using netherrack only in the full check.

## Trie construction and fused GPU traversal

For every `p` in the selected placement box `O`, preprocessing reads the
ordinary four-way symbols at `p + Q` and inserts that word into the trie. The
leaf output is `p`, corresponding to the paper's pattern-sample address.
Duplicate words share a leaf and produce multiple addresses.

The GPU representation must not use pointers or recursion. A suitable node is
a fixed, aligned record containing four child indices and a leaf-output range:

```text
TrieNode
    children[4]       u32 node indices; a sentinel means missing
    output_start      first entry in the placement array
    output_count      number of placements for a complete word
```

With `V <= 256` and `q <= 4`, even the loose upper bound of `1 + V*q` nodes is
small enough to stay cache-friendly. A 32-byte node would put that upper bound
near 32 KiB per direction, before prefix sharing reduces it.

The fused kernel is conceptually:

```text
anchor = this invocation's world lattice point
node = root

for sample offset in Q:
    model_index = texture_variant(anchor + sample_offset)
    symbol = visible_four_way(model_index)
    node = trie[node].children[symbol]
    if node is missing:
        return

for placement p in trie[node].outputs:
    candidate = anchor - p
    if candidate is outside this tile:
        continue
    if all remaining compiled constraints pass at candidate:
        emit candidate
```

No sample word is written to memory. No candidate list is written and read
back. A trie leaf is consumed immediately by the same invocation.

Each leaf placement should also identify the `q` filter constraints already
proved by the sample. Since the first experiment caps `q` at four and filter
indices fit in a byte, four witness indices can be packed into one `u32`.
The remaining check can skip those texture calculations without duplicating a
full filter order per placement.

The existing `16 x 1 x 16` X/Z workgroup shape can remain an initial choice.
Each invocation can process one lattice X/Z anchor and a small batch of lattice
Y anchors, analogous to the current batch of 32 candidate Y coordinates.

### Expected GPU effects

Positive effects:

- Far fewer 64-bit texture-hash calculations when `V` is large.
- Fixed sample depth and no data-dependent search scheduling.
- Only four possible child choices per level, so nearby threads access a small
  set of cache-resident nodes.
- Choosing `q` near `log_4(V)` keeps the expected leaf output count per anchor,
  `V / 4^q`, near or below one.
- Fewer invocations than candidate-at-a-time search, while real scans still
  contain ample parallel work.

Costs and risks:

- Each trie level introduces a dependent buffer read before the next level.
- Threads reaching different leaf-list lengths diverge during the remaining
  checks.
- Sparse filters can have small `V`, leaving too little shared work to repay
  trie control and memory overhead.
- If almost every possible prefix exists, a trie performs all `q` texture
  calculations and all dependent node reads; a packed base-4 terminal table
  may then be faster.
- A smaller number of longer-lived invocations changes occupancy and timeout
  behavior. Tile-size defaults must be remeasured rather than assumed.

The sample offsets are a set geometrically but an ordered word in the trie.
The planner should try their few permutations and place the most selective
prefix first. Under a uniform four-way model, the expected number of sample
symbols calculated for one anchor is

```text
1 + distinct_prefixes(1)/4
  + distinct_prefixes(2)/4^2
  + ...
```

up to depth `q`. This is a better trie score than always charging exactly `q`.

## Performance model and expectation

Let:

- `V` be the selected placement-box volume;
- `q` be the ordinary four-way sample length;
- `C_verify` be the expected number of remaining texture checks after a
  positive sample.

Ignoring early prefix failure and non-hash overhead gives the conservative
texture-calculation estimate

```text
current:       C_old ~= 4/3
sample + trie: C_new ~= q/V + 4^(-q) * C_verify
```

For ordinary independent constraints, `C_verify` is again about `4/3` when the
sampled constraints are skipped. This model chooses based on `V`, not the total
filter count. That is the main change needed for sparse filters.

The following are static geometry estimates from the repository's current
example filters. They are not measured throughput results.

| Filter | Example compact plan | Estimated texture-work reduction |
| --- | ---: | ---: |
| `example.conf` / `Bright Cavern.conf` dense 42-block patch | `q=3`, `V=30` | about 11.0x |
| `Doughnut SMP.conf` useful dense subpatch | `q=2`, `V=12` | about 5.3x |
| `benchmark.conf` irregular 28-block filter | `q=2`, `V=5` | about 2.8x |

For example, `q=3`, `V=30` gives
`3/30 + (1/64)*(4/3) = 0.1208` texture calculations per candidate, compared
with about `1.333` today. The irregular benchmark gives
`2/5 + (1/16)*(4/3) = 0.4833`.

These factors are ceilings on hash-work improvement, not promises of equal
end-to-end speedup. Trie reads, output-loop control, bounds checks, workgroup
occupancy, and the existing shader's efficiency reduce the realized gain.

My initial GPU expectation is:

- dense regular filters: roughly 3x to 8x end-to-end;
- moderately patchy filters: roughly 2x to 4x;
- the current irregular `benchmark.conf`: roughly 1.2x to 2x, with a real
  possibility of no gain if dependent trie reads dominate;
- tiny or highly scattered filters: use the current scanner.

On the README's RTX 5080 Laptop baseline of 155 billion candidates/second, the
last range would correspond very roughly to 185-310 billion candidates/second
on the same workload and hardware. This should be treated as a hypothesis for
the experiment, not a target derived from the paper's theorem.

Texture outputs should also be sampled empirically for every supported
algorithm. If adjacent sample positions are correlated, the observed survivor
rate can differ from `4^(-q)` and materially change the result.

## First implementation scope

Include:

- exact matching only (`errorTolerance = 0`);
- ordinary four-way constraints as trie sample slots;
- arbitrary remaining standard constraints in the full check;
- one direction-specific trie per prepared direction;
- current CPU/GPU scanner as the correctness oracle and fallback;
- a forced `naive` versus `sample-trie` mode for A/B benchmarking, plus `auto`
  only after the cost model is calibrated.

Defer:

- mismatch tolerance;
- arbitrary netherrack masks in the trie;
- nonrectangular modular placement covers;
- large-diameter template search;
- dynamic runtime autotuning;
- replacing the trie with a packed table unless profiling justifies it.

The paper's mismatch extension uses multiple disjoint witnesses and per-
candidate counters. On a GPU that likely reintroduces the intermediate state
this exact-match design avoids, so it should be a separate design exercise
rather than an incremental flag in the first kernel.

## Suggested code shape

Keep preprocessing separate from filter semantics:

```text
src/filter.rs
    existing semantic compilation

src/sample_plan.rs                 new, backend-independent
    SampleShape
    PlacementBox
    FlatTrie
    SamplePlan
    plan_exact_filter(...)

src/cpu.rs
    current path
    optional reference implementation of the sample plan

src/gpu.rs
    upload per-direction sample plans
    select naive or sample-trie pipeline per experiment mode

src/search.wgsl
    current naive entry point
    fused sample-trie entry point
```

The CPU reference implementation is valuable even if it is not faster. It can
exercise the same lattice, trie outputs, and tile-boundary rules without GPU
debugging in the loop.

## Implementation sequence

### 1. Planner and proof-oriented tests

- Enumerate compact shapes and valid placements.
- Find candidate placement boxes and score them.
- Build the flat trie and packed leaf addresses.
- Prove by exhaustive tests on small ranges that every candidate maps to
  exactly one `(anchor, placement)` pair.
- Verify duplicate sample words and negative coordinates.

### 2. CPU reference path

- Execute the fused anchor-to-candidate algorithm on the CPU.
- Compare complete match sets with the current CPU scanner for randomized
  small filters, all texture algorithms, all directions, awkward tile edges,
  and integer-boundary cases.
- Keep the current path when the planner declines a filter.

### 3. GPU experimental kernel

- Add read-only flat-trie, placement, and sample-shape buffers.
- Dispatch lattice anchors with tile halos.
- Consume trie leaves immediately and reuse the existing result/counter
  buffers.
- Compare GPU results with both CPU paths.

### 4. Measure before adding features

Benchmark at least:

- the repository's `benchmark.conf`;
- a dense planar filter such as `Bright Cavern.conf`;
- the separated patches in `Doughnut SMP.conf`;
- synthetic tiny, dense 3D, planar, and highly scattered filters;
- Vanilla-1 through Vanilla-3 and both Sodium algorithms;
- multiple Y spans, tile sizes, and directions.

Record more than candidates/second:

- selected `q`, placement dimensions, and `V`;
- trie node count and expected prefix depth;
- anchors processed;
- leaf outputs and full checks;
- observed sample survival rate;
- per-tile GPU time and timeout behavior.

Use a debug/instrumented kernel for counters rather than keeping profiling
atomics in the release hot path.

### 5. Decide trie versus packed lookup

Once the fused design is correct, compare the flat trie with a direct base-4
terminal table using the same plans. Retain the trie if prefix rejection wins;
otherwise keep the planner and change only the dictionary encoding.

## Success criteria

The experiment is successful if:

- it produces exactly the same matches and directions as the current exact
  scanner;
- every candidate is represented once, including at tile and integer
  boundaries;
- it allocates no runtime world-text or candidate buffer;
- the automatic mode never selects a known-slower plan;
- it gives a material gain on the repository benchmark and at least a clear
  multi-fold gain on dense filters.

A reasonable go/no-go threshold after tuning is at least 20% on
`benchmark.conf` and at least 2x on a dense representative filter. If the flat
trie misses those thresholds while the measured texture-work reduction is
present, test the packed terminal table before rejecting the sampling design.

## Conclusion

The paper's main idea fits CoordsFinder, but `filter_count` is not the pattern
size that controls the speedup. The useful quantity is the number `V` of
compact sample placements that form a complete residue box. Dense screenshot
patches can make `V` large and should benefit substantially; scattered filters
may not.

The rectangular-pattern problem has a simple, correctness-preserving first
answer: find a rectangle of valid sample **origins** inside the sparse filter,
then keep every other observation for final verification. Combined with a
flat, fixed-fanout trie and a fused GPU kernel, this realizes the user's desired
buffer-free elimination path while retaining a safe fallback for unsupported
or unprofitable filters.
