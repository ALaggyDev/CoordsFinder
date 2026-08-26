struct Filter {
    x: u32,
    y: u32,
    z: u32,
    rotation: u32,
    visible_mask: u32,
}

struct SearchResult {
    x: i32,
    y: i32,
    z: i32,
    mismatches: i32,
    direction: i32,
}

@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> filters: array<Filter>;
@group(0) @binding(2) var<storage, read_write> results: array<SearchResult>;
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;

const JAVA_MULTIPLIER: u64 = 0x5deece66dlu;
const JAVA_MASK: u64 = 0xfffffffffffflu;
const SODIUM_PHI: u64 = 0x9e3779b97f4a7c15lu;

fn coordinate_random_raw(x: i32, y: i32, z: i32) -> i64 {
    let seed = i64(x * 3129871i) ^ i64(z) * 116129781li ^ i64(y);
    return seed * seed * 42317861li + seed * 11li;
}

fn absolute_modulo(value: i32) -> u32 {
    let magnitude = select(value, 0i - value, value < 0i);
    return u32(magnitude) % 4u;
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

fn random_vanilla3(seed_input: i64) -> u32 {
    var seed = (u64(seed_input) ^ JAVA_MULTIPLIER) & JAVA_MASK;
    seed = (seed * JAVA_MULTIPLIER + 11lu) & JAVA_MASK;
    let next = u32(seed >> 17u);
    return u32((4lu * u64(next)) >> 31u);
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

fn texture_variant(algorithm: u32, x: i32, y: i32, z: i32) -> u32 {
    let raw = coordinate_random_raw(x, y, z);
    if algorithm == 0u {
        return absolute_modulo(i32(raw) >> 16u);
    }
    let seed = raw >> 16u;
    if algorithm == 1u {
        return absolute_modulo(random_vanilla2(seed));
    }
    if algorithm == 2u {
        return random_vanilla3(seed);
    }
    if algorithm == 3u {
        return absolute_modulo(random_sodium1(u64(seed)));
    }
    return absolute_modulo(random_sodium2(u64(seed)));
}

@compute @workgroup_size(8, 8, 1)
fn search(@builtin(global_invocation_id) id: vec3<u32>) {
    let x_span = params[3];
    let y_span = params[4];
    let z_span = params[5];
    if id.x >= x_span || id.y >= z_span || id.z >= y_span {
        return;
    }

    let x = bitcast<i32>(params[0]) + i32(id.x);
    let y = bitcast<i32>(params[1]) + i32(id.z);
    let z = bitcast<i32>(params[2]) + i32(id.y);
    var mismatches = 0u;
    for (var index = 0u; index < params[8]; index++) {
        let sample = filters[index];
        let variant = texture_variant(
            params[6],
            x + bitcast<i32>(sample.x),
            y + bitcast<i32>(sample.y),
            z + bitcast<i32>(sample.z),
        );
        if (variant & sample.visible_mask) != sample.rotation {
            mismatches++;
            if mismatches > params[7] {
                return;
            }
        }
    }

    let result_index = atomicAdd(&counters[0], 1u);
    if result_index >= params[10] {
        atomicStore(&counters[1], 1u);
        return;
    }
    results[result_index] = SearchResult(x, y, z, i32(mismatches), bitcast<i32>(params[9]));
}
