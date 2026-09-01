// Experimental packed-Y Vanilla-3 GPU search.
//
// `generate_source_signatures` writes four rotation-major Y bitmasks for the
// sparse source rows. `filter_candidates` gives one invocation ownership of
// one candidate (x,z), keeps its 61 candidate-Y bits private, and pulls the
// masks required by that candidate. The two dispatches are separated by a
// compute-pass boundary in Rust, so the second pass observes all source writes.

struct Filter {
    // xyz are source offsets. The low 16 bits of w are the exact model-index
    // mask; bits 16..19 are the four visible rotations used by the cheap pass.
    values: vec4<i32>,
}

struct SearchResult {
    x: i32,
    y: i32,
    z: i32,
    mismatches: i32,
    direction: i32,
}

struct PackedParams {
    source_x_start: u32,
    source_z_start: u32,
    source_width: u32,
    source_rows: u32,
    candidate_x_start: u32,
    candidate_z_start: u32,
    candidate_width: u32,
    candidate_rows: u32,
    direction: u32,
    filter_count: u32,
    dense_stride: u32,
    source_z_residue: u32,
    initial_live_low: u32,
    initial_live_high: u32,
    minimum_dx: u32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> params: PackedParams;
@group(0) @binding(1) var<uniform> filters: array<Filter, 256>;
@group(0) @binding(2) var<storage, read_write> source_masks: array<u64>;
@group(0) @binding(3) var<storage, read_write> results: array<SearchResult>;
// counters[0] is the number of matches; counters[1] is the overflow flag.
@group(0) @binding(4) var<storage, read_write> counters: array<atomic<u32>>;
// buckets[candidate_z mod stride] stores mask (start,count) in xy and exact
// verifier (start,count) in zw. A vec4 matches the uniform array's 16-byte stride.
@group(0) @binding(5) var<uniform> buckets: array<vec4<u32>, 256>;

const JAVA_MULTIPLIER: u64 = 0x5deece66dlu;
const JAVA_MASK: u64 = 0xfffffffffffflu;
const MIX_MULTIPLIER: u64 = 42317861lu;
const SECOND_DIFFERENCE: u64 = 84635722lu;
const RESULT_CAPACITY: u32 = 262144u;
const PACKED_Y_MIN: i32 = -60i;
// The generic pipeline leaves this at zero and reads the stride from the
// uniform. The benchmark pipeline sets it to four, allowing the compiler to
// replace the per-observation divide and remainder with shifts and masks.
override DENSE_STRIDE: u32 = 0u;

fn packed_dense_stride() -> u32 {
    if DENSE_STRIDE == 4u {
        return 4u;
    }
    return params.dense_stride;
}

fn packed_source_row(source_z: i32) -> u32 {
    let offset = u32(source_z - bitcast<i32>(params.source_z_start));
    if DENSE_STRIDE == 4u {
        return offset >> 2u;
    }
    return offset / params.dense_stride;
}

fn packed_candidate_residue(z: i32) -> u32 {
    // A two's-complement low-bit mask is the Euclidean remainder for a
    // power-of-two divisor, including negative coordinates.
    if DENSE_STRIDE == 4u {
        return bitcast<u32>(z) & 3u;
    }
    return u32(euclidean_remainder(z, i32(params.dense_stride)));
}

fn mixed_value(seed: u64) -> u64 {
    return seed * (seed * MIX_MULTIPLIER + 11lu);
}

fn visible_rotation_from_mixed(mixed: u64) -> u32 {
    var state = ((mixed >> 16u) ^ JAVA_MULTIPLIER) & JAVA_MASK;
    state = (state * JAVA_MULTIPLIER + 11lu) & JAVA_MASK;
    return u32(state >> 46u);
}

fn exact_variant(x: i32, y: i32, z: i32) -> u32 {
    let coordinate_seed = i64(x * 3129871i) ^ i64(z) * 116129781li ^ i64(y);
    let mixed = coordinate_seed * (coordinate_seed * 42317861li + 11li);
    var state = (u64(mixed >> 16u) ^ JAVA_MULTIPLIER) & JAVA_MASK;
    state = (state * JAVA_MULTIPLIER + 11lu) & JAVA_MASK;
    let next = u32(state >> 17u);
    return u32((16lu * u64(next)) >> 31u);
}

fn euclidean_remainder(value: i32, modulus: i32) -> i32 {
    let remainder = value % modulus;
    return select(remainder, remainder + modulus, remainder < 0i);
}

fn append_result(x: i32, y: i32, z: i32) {
    let result_index = atomicAdd(&counters[0], 1u);
    if result_index < RESULT_CAPACITY {
        results[result_index] = SearchResult(
            x,
            y,
            z,
            0i,
            bitcast<i32>(params.direction),
        );
    } else {
        atomicStore(&counters[1], 1u);
    }
}

// One invocation generates all source-Y rotations for one source (x,z).
// X is the fastest-changing dispatch dimension and storage dimension.
@compute @workgroup_size(128, 1, 1)
fn generate_source_signatures(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.source_width || id.y >= params.source_rows {
        return;
    }

    let x = bitcast<i32>(params.source_x_start) + i32(id.x);
    let z = bitcast<i32>(params.source_z_start) + i32(id.y * packed_dense_stride());
    let base = u64(i64(x * 3129871i)) ^ u64(i64(z) * 116129781li);

    // For source Y -64..-1, base XOR y contains each six-bit low value once.
    // Generate those consecutive seeds with the exact quadratic recurrence,
    // then permute each result into its actual Y bit with low XOR base_low.
    let first_seed = (base ^ 0xfffffffffffffffflu) & 0xffffffffffffffc0lu;
    var value = mixed_value(first_seed);
    var difference = MIX_MULTIPLIER * (first_seed * 2lu + 1lu) + 11lu;
    let base_low = u32(base & 63lu);
    var rotation_low_bits = 0lu;
    var rotation_high_bits = 0lu;

    for (var low = 0u; low < 64u; low++) {
        let rotation = visible_rotation_from_mixed(value);
        let bit = 1lu << (low ^ base_low);
        rotation_low_bits |= select(0lu, bit, (rotation & 1u) != 0u);
        rotation_high_bits |= select(0lu, bit, (rotation & 2u) != 0u);
        value += difference;
        difference += SECOND_DIFFERENCE;
    }

    var masks = array<u64, 4>(
        (~(rotation_low_bits | rotation_high_bits)) >> 4u,
        (rotation_low_bits & ~rotation_high_bits) >> 4u,
        (~rotation_low_bits & rotation_high_bits) >> 4u,
        (rotation_low_bits & rotation_high_bits) >> 4u,
    );

    // Y=0 and Y=1 do not share the negative-Y prefix, so calculate them
    // directly and place them in source bits 60 and 61.
    let y0 = visible_rotation_from_mixed(mixed_value(base));
    let y1 = visible_rotation_from_mixed(mixed_value(base ^ 1lu));
    masks[y0] |= 1lu << 60u;
    masks[y1] |= 1lu << 61u;

    let plane_size = params.source_rows * params.source_width;
    let row_offset = id.y * params.source_width + id.x;
    for (var rotation = 0u; rotation < 4u; rotation++) {
        source_masks[rotation * plane_size + row_offset] = masks[rotation];
    }
}

// One invocation owns one candidate (x,z) column. No other invocation reads
// or writes its `live` value, so mask intersection needs no atomics.
@compute @workgroup_size(128, 1, 1)
fn filter_candidates(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.candidate_width || id.y >= params.candidate_rows {
        return;
    }

    let x = bitcast<i32>(params.candidate_x_start) + i32(id.x);
    let z = bitcast<i32>(params.candidate_z_start) + i32(id.y);
    let plane_size = params.source_rows * params.source_width;
    var live = u64(params.initial_live_low) | (u64(params.initial_live_high) << 32u);

    // Rust partitions observations by the candidate-Z residue that makes
    // candidate_z + dz land on a selected source row. All invocations in this
    // candidate row therefore follow the same short, branch-free mask loop.
    let bucket = buckets[packed_candidate_residue(z)];
    let bucket_end = bucket.x + bucket.y;
    for (var filter_index = bucket.x; filter_index < bucket_end; filter_index++) {
        let observation = filters[filter_index].values;
        let source_z = z + observation.z;
        let source_row = packed_source_row(source_z);
        let source_x = id.x + u32(observation.x - bitcast<i32>(params.minimum_dx));
        let visible_rotations = u32(observation.w) >> 16u;
        let row_offset = source_row * params.source_width + source_x;
        var accepted_y = 0lu;

        // Four-way observations select exactly one plane. Avoid a four-step
        // loop in that overwhelmingly common case; side observations retain
        // the generic union path.
        if (visible_rotations & (visible_rotations - 1u)) == 0u {
            let rotation = firstTrailingBit(visible_rotations);
            accepted_y = source_masks[rotation * plane_size + row_offset];
        } else {
            for (var rotation = 0u; rotation < 4u; rotation++) {
                if (visible_rotations & (1u << rotation)) != 0u {
                    accepted_y |= source_masks[rotation * plane_size + row_offset];
                }
            }
        }
        live &= accepted_y >> u32(observation.y);
        if live == 0lu {
            return;
        }
    }

    // Exact verification is deliberately authoritative. Sparse masks may omit
    // constraints and leave false positives, but they can never report one.
    // Enumerate only set bits. Most columns return above with live==0; rare
    // survivors now cost one iteration per surviving Y instead of 61 tests.
    var surviving_y = live;
    while surviving_y != 0lu {
        let bit = firstTrailingBit(surviving_y);
        surviving_y &= surviving_y - 1lu;
        let y = PACKED_Y_MIN + i32(bit);
        var exact = true;
        let verifier_end = bucket.z + bucket.w;
        for (var filter_index = bucket.z; filter_index < verifier_end; filter_index++) {
            let observation = filters[filter_index].values;
            let accepted_indices = u32(observation.w) & 0xffffu;
            let variant = exact_variant(
                x + observation.x,
                y + observation.y,
                z + observation.z,
            );
            if (accepted_indices & (1u << variant)) == 0u {
                exact = false;
                break;
            }
        }
        if exact {
            append_result(x, y, z);
        }
    }
}
