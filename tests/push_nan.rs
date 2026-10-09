//! Non-finite objective values: `ParetoFrontier::try_new` returns
//! `FrontierError::NonFinite`, and `try_push` reports the same error for a
//! single point instead of panicking like `push`.

use pare::{Direction, FrontierError, ParetoFrontier};

#[test]
fn try_push_rejects_nan_with_the_try_new_error() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Minimize]);
    assert_eq!(f.try_push(vec![0.5, 0.2], "a"), Ok(true));
    assert_eq!(
        f.try_push(vec![0.4, f64::NAN], "b"),
        Err(FrontierError::NonFinite {
            point_idx: 0,
            dim_idx: 1
        })
    );
    assert_eq!(
        f.try_push(vec![f64::INFINITY, 0.1], "c"),
        Err(FrontierError::NonFinite {
            point_idx: 0,
            dim_idx: 0
        })
    );
    assert_eq!(f.len(), 1);

    assert_eq!(
        ParetoFrontier::try_new(&[vec![0.4, f64::NAN]]).unwrap_err(),
        FrontierError::NonFinite {
            point_idx: 0,
            dim_idx: 1
        }
    );
}

#[test]
fn try_push_rejects_wrong_dimension() {
    let mut f = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    assert_eq!(
        f.try_push(vec![0.5], ()),
        Err(FrontierError::InconsistentDimensions)
    );
}

#[test]
fn try_push_matches_push_on_finite_input() {
    let mut a = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    let mut b = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
    for (i, p) in [[0.9, 0.1], [0.5, 0.5], [0.4, 0.4], [0.1, 0.9]]
        .iter()
        .enumerate()
    {
        assert_eq!(a.try_push(p.to_vec(), i), Ok(b.push(p.to_vec(), i)));
    }
    assert_eq!(a.len(), b.len());
}
