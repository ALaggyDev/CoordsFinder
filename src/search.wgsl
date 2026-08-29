// GPU search kernel. Rust specializes the override constants once per config
// and dispatches one X/Z plan tile at a time. Each invocation scans up to 32 Y
// coordinates for one X/Z coordinate.

struct Filter {
    values: vec4<i32>,
}

struct SearchResult {
    x: i32,
    y: i32,
    z: i32,
    mismatches: i32,
    direction: i32,
}

struct SearchParams {
    start_x: u32,
    start_z: u32,
    x_span: u32,
    z_span: u32,
    direction: u32,
    forced_errors: u32,
    filter_count: u32,
}

@group(0) @binding(0) var<uniform> params: SearchParams;
@group(0) @binding(1) var<uniform> filters: array<Filter, 256>;
@group(0) @binding(2) var<storage, read_write> results: array<SearchResult>;
// counters[0] is the number of matches; counters[1] is the overflow flag.
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;

const JAVA_MULTIPLIER: u64 = 0x5deece66dlu;
const JAVA_MASK: u64 = 0xfffffffffffflu;
const SODIUM_PHI: u64 = 0x9e3779b97f4a7c15lu;
const RESULT_CAPACITY: u32 = 262144u;
override TEXTURE_ALGORITHM: u32 = 0u;
override ERROR_TOLERANCE: u32 = 0u;
override Y_START: i32 = 0i;
override Y_SPAN: u32 = 1u;

fn coordinate_random_raw(x: i32, y: i32, z: i32) -> i64 {
    let seed = i64(x * 3129871i) ^ i64(z) * 116129781li ^ i64(y);
    return seed * (seed * 42317861li + 11li);
}

fn coordinate_random_legacy(x: i32, y: i32, z: i32) -> i32 {
    let seed = x * 3129871i ^ z * 116129781i ^ y;
    return seed * (seed * 42317861i + 11i);
}

fn absolute_modulo_16(value: i32) -> u32 {
    let magnitude = select(value, 0i - value, value < 0i);
    return u32(magnitude) & 15u;
}

fn stafford_mix13(input: u64) -> u64 {
    var value = (input ^ (input >> 30u)) * 0xbf58476d1ce4e5b9lu;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111eblu;
    return value ^ (value >> 31u);
}

fn random_vanilla2(seed_input: i64) -> i32 {
    let seed = (u64(seed_input) ^ JAVA_MULTIPLIER) & JAVA_MASK;
    return i32((seed * 0xbb20b4600a69lu + 0x40942de6balu) >> 16u);
}

fn random_vanilla3_16(seed_input: i64) -> u32 {
    var seed = (u64(seed_input) ^ JAVA_MULTIPLIER) & JAVA_MASK;
    seed = (seed * JAVA_MULTIPLIER + 11lu) & JAVA_MASK;
    let next = u32(seed >> 17u);
    return u32((16lu * u64(next)) >> 31u);
}

fn random_sodium1(seed_input: u64) -> i32 {
    var seed = seed_input;
    seed = seed ^ (seed >> 33u);
    seed = seed * 0xff51afd7ed558ccdlu;
    seed = seed ^ (seed >> 33u);
    seed = seed * 0xc4ceb9fe1a85ec53lu;
    seed = seed ^ (seed >> 33u);
    let first = stafford_mix13(seed + SODIUM_PHI);
    let second = stafford_mix13(seed + SODIUM_PHI + SODIUM_PHI);
    return i32(first + second);
}

fn rotate_left_17(value: u64) -> u64 {
    return (value << 17u) | (value >> 47u);
}

fn random_sodium2(seed_input: u64) -> i32 {
    var low = seed_input ^ 7640891576956012809lu;
    var high = low + u64(-7046029254386353131li);
    low = stafford_mix13(low);
    high = stafford_mix13(high);
    return i32(rotate_left_17(low + high) + low);
}

fn texture_variant(x: i32, y: i32, z: i32) -> u32 {
    // Keep this mapping synchronized with TextureAlgorithm's repr in Rust.
    if TEXTURE_ALGORITHM == 0u {
        return absolute_modulo_16(coordinate_random_legacy(x, y, z) >> 16u);
    }
    let seed = coordinate_random_raw(x, y, z) >> 16u;
    if TEXTURE_ALGORITHM == 1u {
        return absolute_modulo_16(random_vanilla2(seed));
    }
    if TEXTURE_ALGORITHM == 2u {
        return random_vanilla3_16(seed);
    }
    if TEXTURE_ALGORITHM == 3u {
        return absolute_modulo_16(random_sodium1(u64(seed)));
    }
    return absolute_modulo_16(random_sodium2(u64(seed)));
}

@compute @workgroup_size(16, 1, 16)
fn search(@builtin(global_invocation_id) id: vec3<u32>) {
    let x_span = params.x_span;
    let z_span = params.z_span;
    let y_base = id.y * 32u;
    if id.x >= x_span || id.z >= z_span || y_base >= Y_SPAN {
        return;
    }

    let x = bitcast<i32>(params.start_x) + i32(id.x);
    let z = bitcast<i32>(params.start_z) + i32(id.z);
    var y = Y_START + i32(y_base);
    let y_end = Y_START + i32(min(y_base + 32u, Y_SPAN));
    var filter_index = 0u;
    var mismatches = params.forced_errors;
    while y < y_end {
        let sample = filters[filter_index];
        let accepted_indices = u32(sample.values.w);
        let variant = texture_variant(
            x + sample.values.x,
            y + sample.values.y,
            z + sample.values.z,
        );
        let mismatch = (accepted_indices & (1u << variant)) == 0u;

        if ERROR_TOLERANCE == 0u {
            // The exact-match path restarts immediately at the next Y candidate
            // after the first mismatch, avoiding the rest of the filter.
            if mismatch {
                y++;
                filter_index = 0u;
            } else {
                filter_index++;
                if filter_index == params.filter_count {
                    let result_index = atomicAdd(&counters[0], 1u);
                    if result_index < RESULT_CAPACITY {
                        results[result_index] = SearchResult(x, y, z, i32(mismatches), bitcast<i32>(params.direction));
                    } else {
                        atomicStore(&counters[1], 1u);
                    }
                    y++;
                    filter_index = 0u;
                }
            }
        } else {
            if mismatch {
                mismatches++;
            }
            filter_index++;

            if mismatches > ERROR_TOLERANCE {
                y++;
                filter_index = 0u;
                mismatches = params.forced_errors;
            } else if filter_index == params.filter_count {
                let result_index = atomicAdd(&counters[0], 1u);
                if result_index < RESULT_CAPACITY {
                    results[result_index] = SearchResult(x, y, z, i32(mismatches), bitcast<i32>(params.direction));
                } else {
                    atomicStore(&counters[1], 1u);
                }
                y++;
                filter_index = 0u;
                mismatches = params.forced_errors;
            }
        }
    }
}
