# CoordsFinder codebase guide

CoordsFinder searches for Minecraft coordinates using texture rotations. This
guide explains the main parts of the program and how data moves through them.

## How a scan works

CoordsFinder does five main things:

1. It reads the search ranges, texture algorithm, directions, and filter rows
   from a config file.
2. It prepares one filter for each direction. Rows for different faces of the
   same block are combined here.
3. It divides the search area into tiles in linear or spiral order.
4. The CPU or GPU scans every coordinate in those tiles and checks the prepared
   block filters.
5. It prints coordinates whose block error count is within `errorTolerance`.

```text
config file
    |
    v
config parsing and validation
    |
    v
filter preparation for each direction
    |
    v
linear or spiral tile plan
    |
    +------------------+
    |                  |
    v                  v
CPU scanner        GPU scanner
    |                  |
    +--------+---------+
             v
          matches
```

## Files

| File | What it does |
| --- | --- |
| `src/main.rs` | Command-line interface, backend selection, progress, cancellation, and output |
| `src/config.rs` | Config parsing and validation |
| `src/types.rs` | Types shared by the parser, filter code, and scanners |
| `src/filter.rs` | Direction rotation, 16-way masks, row combining, and forced errors |
| `src/sample_plan.rs` | Compact sample-shape planning, placement boxes, and flat tries |
| `src/texture.rs` | Minecraft and Sodium texture random functions |
| `src/scan.rs` | Linear and spiral tile planning |
| `src/cpu.rs` | Multithreaded CPU scanner |
| `src/gpu.rs` | wgpu setup, GPU dispatch, and result downloads |
| `src/search.wgsl` | GPU search code |

`src/lib.rs` exports the library modules. `src/main.rs` builds the command-line
program from those modules.

## Config parsing

`config::load` reads the config file and returns a `ScanConfig`. It parses the
main settings and the rows under `[filter]`.

Filter rows have these forms:

```text
x y z | rotation
x y z | rotation side
x y z | rotation netherrack-<face>
```

Each row becomes a `RotationInfo` containing:

- an X/Y/Z offset from the coordinate being tested;
- the texture rotation; and
- `StandardFourWay`, `StandardSide`, or `Netherrack(Face)`.

The parser checks the ranges, directions, tile sizes, row syntax, rotation
values, and rotated offset limits. It also runs filter preparation once so bad
row combinations fail before the scan starts.

## Why filters use a 16-bit mask

Netherrack has 16 model choices. Existing blocks have four choices. Both kinds
are converted to the same format before scanning:

```text
CompiledRotation {
    x,
    y,
    z,
    accepted_indices: u16,
}
```

Bit `i` in `accepted_indices` tells the scanner whether 16-way model index `i`
matches that block.

The scan loop always calculates a value from 0 to 15:

```rust
let index = A::sample(x, y, z, 16);
```

It checks the block with:

```rust
let mismatch = accepted_indices & (1 << index) == 0;
```

This keeps the CPU and GPU loops the same for ordinary blocks and netherrack.

## Converting a 16-way index to an ordinary four-way variant

Ordinary filter rows still need the result Minecraft would produce with a
bound of 4. `visible_four_way` converts the 16-way result back to that value.

Vanilla-1, Vanilla-2, Sodium-1, and Sodium-2 use absolute modulo:

```text
sample(4) == sample(16) & 3

16-way: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
4-way:  0 1 2 3 0 1 2 3 0 1  2  3  0  1  2  3
```

Vanilla-3 uses Java's power-of-two `Random.nextInt`:

```text
sample(4) == sample(16) >> 2

16-way: 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15
4-way:  0 0 0 0 | 1 1 1 1 | 2 2  2  2 |  3  3  3  3
```

This is why `visible_four_way` depends on `TextureAlgorithm`. Using `& 3` for
Vanilla-3, or `>> 2` for the other algorithms, gives the wrong rotations.

For a four-way row, the compiler sets the four bits that produce the requested
variant. For a `side` row, it sets the eight bits whose four-way value has the
requested low bit.

## Netherrack

Minecraft's netherrack blockstate contains 16 equally weighted entries. It
chooses one with `nextInt(16)`, then reads the entry as two whole-cube
rotations:

```text
model_index = nextInt(16)
x_rotation = 90 * (model_index % 4)
y_rotation = 90 * (model_index / 4)
```

Minecraft applies the X rotation first and the Y rotation second. The model is
a normal cube with the same netherrack texture on all six faces. `uvlock` is
off, so each face's texture moves and turns with the cube.

The 16 model choices do not give one face 16 possible texture rotations. A
world face still shows rotation 0, 1, 2, or 3. The model index decides which
source cube face moves into each world direction and how far its texture turns.
All visible faces of one netherrack block are therefore tied to the same model
index.

The table in `src/filter.rs` was made by applying those X and Y rotations to
the textured cube. For each model index, we follow every source face to its new
world face and measure the texture's clockwise quarter-turn while looking at
that face from outside the block. The result maps every model index to the
visible rotation on all six world faces:

```text
NETHERRACK_FACE_ROTATIONS[model_index][world_face]
```

For this row:

```text
4 1 -2 | 3 netherrack-north
```

the compiler sets every bit whose model index shows rotation 3 on the north
face.

The rotation is measured clockwise from the raw netherrack texture while
looking at the block from outside the named face.

## Directions

CoordsFinder prepares a separate block filter for every value in `directions`.
The X/Z offset of every block rotates around the Y axis. The texture value then
changes according to the row type:

- Ordinary top or bottom face (`x y z | rotation`): increase `rotation` by the
  direction.
- Ordinary side face (`x y z | rotation side`): keep the same two-state side
  value.
- Netherrack top face (`netherrack-up`): increase the texture rotation by the
  direction.
- Netherrack bottom face (`netherrack-down`): decrease the texture rotation by
  the direction. It turns the other way because the face is viewed from below.
- Netherrack side face (`netherrack-north`, `-east`, `-south`, or `-west`):
  rotate the face name `north -> east -> south -> west`, but keep its in-plane
  texture rotation unchanged.

## Combining rows at the same block

`prepare_filters` groups rows with the same rotated X/Y/Z offset. Their masks
are combined with bitwise AND:

```text
combined_mask = first_mask & second_mask & third_mask
```

For example:

```text
4 1 -2 | 1 netherrack-up
4 1 -2 | 3 netherrack-north
4 1 -2 | 2 netherrack-east
```

becomes one `CompiledRotation`. The scanner calculates the random model index
once and checks it against the combined mask.

Ordinary four-way and `side` rows at the same offset are combined in the same
way. Ordinary and netherrack rows cannot be mixed at one offset because one
Minecraft block cannot use both model selectors.

After combining, filters are sorted by the number of set bits in the mask.
Blocks that accept fewer model indices are checked first because they reject
wrong coordinates more quickly.

## Error tolerance and forced errors

`errorTolerance` counts blocks, not filter rows. If three faces of one block do
not match, that block adds one error.

Sometimes rows at the same offset produce a combined mask of zero. No model
index can match that block, so it always adds one error. Filter preparation:

1. removes the zero mask from the scan loop;
2. adds one to the direction's `forced_errors` count; and
3. prints a warning showing the direction and block offset.

The CPU and GPU start each coordinate with this forced error count. This is the
same as reducing the remaining tolerance, while keeping the correct total in
the printed match.

If forced errors exceed the tolerance for one direction, that direction is
skipped. If every direction is skipped, config validation fails.

The program also rejects a direction that would have no usable filters after
zero masks are removed. Allowing it would make every coordinate match.

## Texture random functions

`texture.rs` contains these implementations:

```text
Vanilla-1
Vanilla-2
Vanilla-3
Sodium-1
Sodium-2
```

Each one implements `TextureSampler`. Before the CPU scan starts, the program
selects one concrete sampler:

```rust
run_mode::<Vanilla3>(...)
```

The filter loop can then call the sampler without checking the algorithm every
time.

Minecraft relies on Java integer overflow. The Rust code uses wrapping
arithmetic to reproduce it. Changing signed conversions, shifts, constants, or
overflow behavior can change scan results.

`get_texture` selects the sampler at runtime. It is useful in tests and other
code outside the scan loop.

## Scan planning

`scan::make_plan` splits the X/Z range into tiles. Each tile contains the full
Y range and is repeated for every direction.

The scan order can be:

- `linear`, starting from the minimum X/Z corner; or
- `spiral`, starting near the middle of the search range.

`ScanPlan` does not store every tile. `work_item(n)` calculates a tile when a
scanner asks for it. This keeps plan memory usage small even for very large
search ranges.

All ranges are half-open:

```text
[start, end)
```

## CPU scanner

`CpuScanner` creates worker threads and gives them tiles through an atomic work
index. Every worker:

1. claims the next tile;
2. loops through X, Z, and Y;
3. checks the prepared block filters at each coordinate;
4. stores matches in a small local batch; and
5. sends completed batches to the output callback.

`count_mismatches` starts from `forced_errors`. For each block, it calculates a
16-way model index and checks the corresponding bit in `accepted_indices`. It
stops when the error count exceeds the tolerance.

The worker checks for Ctrl+C between X/Z columns. Output and progress callbacks
use mutexes so messages from different workers do not overlap.

## Compact-template trie search

Exact scans can use the compact-template trie implemented in
`src/sample_plan.rs`. The planner:

1. finds canonical four-way constraints that can be represented by one trie
   symbol;
2. enumerates sample shapes of up to four offsets in a `2 x 2 x 2` cell;
3. finds a solid rectangular box where the shape fits in the sparse filter;
4. builds a four-child flat trie of the sample words in that box; and
5. estimates whether sampling should beat the ordinary early-exit loop.

The placement box's side lengths become a three-dimensional world lattice.
Every candidate maps to exactly one placement in the box and therefore one
lattice sample. A missing trie prefix rejects all of the candidates represented
by that sample. A matching leaf gives their candidate origins, which are
verified immediately against the remaining compiled filters.

The CPU and GPU implementations do not store sampled world values or a list of
surviving candidates. Only static trie/placement data and the ordinary result
buffer are needed. `searchMode = auto` applies the planner's cost threshold,
`naive` bypasses it, and `sample-trie` requires a plan for A/B tests. Nonzero
error tolerance currently uses the naive scanner.

Normal startup and `--validate` both print the planner variables for every
direction: sample size and offsets, placement-box dimensions and volume, trie
node/leaf/output counts, and modeled texture checks per candidate. Auto mode
also prints a candidate plan when it rejects that plan in favor of naive.

## GPU scanner

`gpu.rs` sets up wgpu and runs `search.wgsl`. It requires the
`SHADER_INT64` feature because the texture random functions need 64-bit integer
math.

The host code:

1. creates the GPU buffers and compute pipeline;
2. uploads one direction's filters and one tile's settings;
3. dispatches the shader;
4. reads the result counters; and
5. downloads match data only when the tile found matches.

For a selected sample plan, the host instead dispatches lattice anchors over a
small tile halo and uploads the sample offsets, flat trie, and leaf placements.
`search_sample_trie` in `search.wgsl` traverses the trie and checks leaf
survivors in the same invocation. Candidate bounds keep ownership with the
original half-open tile even when adjacent tiles recompute halo anchors.

One GPU filter record is 16 bytes:

```text
i32 x offset
i32 y offset
i32 z offset
i32 containing the u16 mask
```

Filter count and forced errors are sent with each tile because different
directions can produce different prepared filters.

The naive shader uses `16 x 1 x 16` workgroups and handles up to 32 Y
coordinates per invocation. The sample-trie shader uses genuinely
three-dimensional `8 x 4 x 8` workgroups, with each invocation handling one
independent lattice anchor rather than a serial Y batch.

The result buffer holds 262,144 matches. If one tile finds more than that, the
program asks the user to lower `gpuTileSize`.

The texture random code exists in both `texture.rs` and `search.wgsl`. Changes
to one copy must also be made in the other copy.

## Command-line program

`main.rs`:

1. parses command-line options;
2. loads the config;
3. prints filter warnings;
4. selects the CPU or GPU backend;
5. creates the tile plan;
6. installs the Ctrl+C handler;
7. runs the scan; and
8. prints the final time and match count.

Matches are written to standard output. Device information, warnings,
progress, and completion messages are written to standard error. This lets a
user redirect match coordinates without also capturing status messages.

`--validate` checks the config and creates CPU and GPU plans without running a
scan or initializing a GPU.

## Adding another block type

The current mask format works for blocks that use one position-based model
selector with no more than 16 results.

To add one:

1. Add a `RotationKind` and config marker.
2. Add its direction-rotation rules.
3. Convert each filter row into a 16-bit mask.
4. Define which row kinds can be combined at the same block offset.
5. Add CPU, GPU, direction, and reference-data tests.

Some blocks do not fit this design. Fire and chorus plant can consume several
random values at one block. They need code that tracks those calls instead of
one 16-bit mask.

## Things that are easy to break

- Netherrack faces share one 16-way model index. Do not randomize each face
  separately.
- Vanilla-3 uses `index >> 2` for ordinary four-way blocks. The other
  algorithms use `index & 3`.
- Rows at the same offset count as one block for `errorTolerance`.
- Up and down netherrack rotations change in opposite directions.
- Rust and WGSL must use the same random functions.
- The random functions need wrapping integer arithmetic.
- Match output must stay on stdout; status output must stay on stderr.
