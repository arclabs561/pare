//! Rosetta correctness fixtures: pare hypervolume asserted against pymoo.
//!
//! Reference values in `fixtures/rosetta/pare_hypervolume.json` come from
//! `gen_pare.py` (their provenance). Both pare and pymoo compute the exact
//! Lebesgue measure of the region dominated by the front and bounded by the
//! reference point, under the minimize convention, so they agree exactly (EXACT
//! class, both f64). pare orients each Minimize objective internally as
//! ref - value, matching pymoo HV(ref_point) on the raw point set (both filter
//! dominated points).
//!
//! Regenerate the fixture: `uv run tests/fixtures/rosetta/gen_pare.py`.

use pare::{Direction, ParetoFrontier};
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/rosetta/pare_hypervolume.json");

#[derive(Deserialize)]
struct Fixture {
    points_2d: Vec<Vec<f64>>,
    ref_2d: Vec<f64>,
    points_3d: Vec<Vec<f64>>,
    ref_3d: Vec<f64>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    hv_2d: f64,
    hv_3d: f64,
}

fn close(got: f64, want: f64, label: &str) {
    let tol = 1e-9 * (1.0 + want.abs());
    let diff = (got - want).abs();
    assert!(
        diff <= tol,
        "{label}: pare={got} pymoo={want} diff={diff} tol={tol}"
    );
}

fn hypervolume(points: &[Vec<f64>], ref_point: &[f64]) -> f64 {
    let dirs = vec![Direction::Minimize; ref_point.len()];
    let mut front: ParetoFrontier<()> = ParetoFrontier::new(dirs);
    for p in points {
        front.push(p.clone(), ());
    }
    front.hypervolume(ref_point)
}

#[test]
fn rosetta_hypervolume_matches_pymoo() {
    let fx: Fixture = serde_json::from_str(FIXTURE).expect("parse rosetta fixture");

    close(
        hypervolume(&fx.points_2d, &fx.ref_2d),
        fx.expected.hv_2d,
        "hv_2d",
    );
    close(
        hypervolume(&fx.points_3d, &fx.ref_3d),
        fx.expected.hv_3d,
        "hv_3d",
    );
}
