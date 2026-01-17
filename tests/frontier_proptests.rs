use pare::{Direction, ParetoFrontier};
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_pareto_invariant(values in prop::collection::vec(prop::collection::vec(0.0..1.0, 2), 1..50)) {
        let mut frontier = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
        for (i, v) in values.into_iter().enumerate() {
            frontier.push(v, i);
        }

        let points = frontier.points();
        for i in 0..points.len() {
            for j in 0..points.len() {
                if i == j { continue; }
                // No point on the frontier should dominate another.
                assert!(!frontier.point_dominates(&points[i].values, &points[j].values),
                    "Point {:?} dominates {:?}", points[i].values, points[j].values);
            }
        }
    }

    #[test]
    fn test_crowding_distance_ordering(values in prop::collection::vec(prop::collection::vec(0.0..1.0, 2), 3..50)) {
        let mut frontier = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
        for (i, v) in values.into_iter().enumerate() {
            frontier.push(v, i);
        }

        let dists = frontier.crowding_distances();
        assert_eq!(dists.len(), frontier.len());
        
        // At least two points should have infinite distance (the boundaries) 
        // if we have at least 2 points.
        if frontier.len() >= 2 {
            let infinite_count = dists.iter().filter(|&&d| d.is_infinite()).count();
            assert!(infinite_count >= 2, "Should have at least 2 boundary points with infinite distance");
        }
    }

    #[test]
    fn test_hypervolume_monotone_2d(values in prop::collection::vec(prop::collection::vec(0.0..1.0, 2), 1..80)) {
        // With maximize objectives and a reference point at the origin, hypervolume is:
        // - always >= 0
        // - monotone non-decreasing as we add points (dominated points are ignored by the frontier)
        let mut frontier = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);
        let ref_point = [0.0, 0.0];
        let mut prev = 0.0;

        for (i, v) in values.into_iter().enumerate() {
            frontier.push(v, i);
            let hv = frontier.hypervolume(&ref_point);
            assert!(hv >= -1e-9, "hypervolume must be non-negative");
            assert!(hv + 1e-9 >= prev, "hypervolume must be monotone (prev={prev}, hv={hv})");
            // Upper bound: max rectangle is 1x1 in the unit square.
            assert!(hv <= 1.0 + 1e-9, "hypervolume must be <= 1 in [0,1]^2 (hv={hv})");
            prev = hv;
        }
    }
}
