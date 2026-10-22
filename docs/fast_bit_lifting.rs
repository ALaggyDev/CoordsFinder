//! A reference implementation of the block offset fast bit lifting algorithm by fanda857 and epic10l2.
//!
//! This algorithm is conceptually similar to a mathematical method called [Hensel Lifting](https://en.wikipedia.org/wiki/Hensel%27s_lemma).
//! The algorithm starts from the least significant bits of the XZ coordinates,
//! and recursively lifts them to higher bits while checking the block offset constraints.
//!
//! While the theory of Hensel Lifting is not new to the Minecraft technical community,
//! fanda857 and epic10l2 were probably the first ones to apply it to the block offset coordinate cracking problem.
//!
//! All credit goes to fanda857 and epic10l2 for the algorithm :)
//!
//! - by Laggy
//!
//! # Speed
//!
//! On a 10-thread CPU, this algorithm takes about 8-10 seconds to find all matching coordinates within the full 60 million block world.
//!
//! The speed is only weakly dependent on the size of the world.
//!
//! # Note
//!
//! This implementation does not handle the dripstone XZ offset clipping properly.
//!
//! fanda857's implementation: <https://gitlab.com/fanda8571/fast_offsets>

use std::time::Instant;

use rayon::prelude::*;

pub struct Constraint {
    pub dx: i32,
    pub dz: i32,
    pub target: u32,
    pub known_mask: u32,
}

impl Constraint {
    pub const fn new_partial(dx: i32, dz: i32, offx: u32, offz: u32) -> Self {
        Self {
            dx,
            dz,
            target: ((offz & 0xF) << 24) | ((offx & 0xF) << 16),
            known_mask: 0xF0F0000,
        }
    }

    pub const fn new_full(dx: i32, dz: i32, offx: u32, offy: u32, offz: u32) -> Self {
        Self {
            dx,
            dz,
            target: ((offz & 0xF) << 24) | ((offy & 0xF) << 20) | ((offx & 0xF) << 16),
            known_mask: 0xFFF0000,
        }
    }
}

#[lower::apply(wrapping)]
fn get_coordinate_seed(x: u32, z: u32) -> u32 {
    let seed = (x * 3129871) ^ (z * 116129781);
    seed * seed * 42317861 + seed * 11
}

fn check_bound(num: u32, bits: u32) -> bool {
    const BOUND: i32 = 30_000_000;

    let period = 1i32 << bits;
    let num = num as i32;

    num < BOUND || num - period > -BOUND
}

fn lift(x: u32, z: u32, bits: u32, constraints: &[Constraint], results: &mut Vec<(i32, i32)>) {
    // If we have lifted all 28 bits, we have found a valid solution.
    if bits == 28 {
        let solution = ((x as i32) << 4 >> 4, (z as i32) << 4 >> 4);
        results.push(solution);

        println!("Found solution: {:?}", solution);

        return;
    }

    // Each option represents a combination of the next bit for x and z,
    // therefore there are 4 options (00, 01, 10, 11).
    'outer: for option in 0..4 {
        let new_x = x | ((option & 1) << bits);
        let new_z = z | ((option >> 1) << bits);

        // Early prune if the coordinates are not within bounds.
        if (option & 1 != 0 && !check_bound(new_x, bits + 1))
            || (option >> 1 != 0 && !check_bound(new_z, bits + 1))
        {
            continue;
        }

        // Check all constraints.
        for constraint in constraints {
            let dx = new_x.wrapping_add_signed(constraint.dx);
            let dz = new_z.wrapping_add_signed(constraint.dz);

            let seed = get_coordinate_seed(dx, dz);
            let mask = ((1 << (bits + 1)) - 1) & constraint.known_mask;

            if seed & mask != constraint.target & mask {
                continue 'outer;
            }
        }

        // If all constraints are satisfied, lift the next bit.
        lift(new_x, new_z, bits + 1, constraints, results);
    }
}

fn solve(constraints: &[Constraint]) -> Vec<(i32, i32)> {
    (0..65536)
        .into_par_iter()
        .flat_map(|x| {
            let mut results = Vec::new();
            for z in 0..65536 {
                lift(x, z, 16, constraints, &mut results);
            }
            results
        })
        .collect()
}

fn main() {
    // 12541022, 10457150
    let constraints = [
        Constraint::new_full(2, -1, 11, 11, 5),
        Constraint::new_full(1, 0, 3, 2, 15),
        Constraint::new_full(3, -2, 10, 11, 12),
        Constraint::new_full(3, -1, 7, 8, 0),
        Constraint::new_full(-4, 1, 6, 15, 11),
    ];

    let start = Instant::now();

    let results = solve(&constraints);
    println!("Found {} solutions", results.len());

    println!("Elapsed time: {:.2?}", start.elapsed());
}
