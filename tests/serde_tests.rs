//! Serde round-trip tests for public types behind the `serde` feature gate.

#![cfg(feature = "serde")]

use pare::{Direction, NormalizationStats, Point};

#[test]
fn direction_roundtrip() {
    for dir in [Direction::Maximize, Direction::Minimize] {
        let json = serde_json::to_string(&dir).unwrap();
        let back: Direction = serde_json::from_str(&json).unwrap();
        assert_eq!(dir, back);
    }
}

#[test]
fn point_roundtrip() {
    let p = Point {
        values: vec![0.1, 0.9, 0.5],
        data: "hello".to_string(),
    };
    let json = serde_json::to_string(&p).unwrap();
    let back: Point<String> = serde_json::from_str(&json).unwrap();
    assert_eq!(back.values, p.values);
    assert_eq!(back.data, p.data);
}

#[test]
fn normalization_stats_roundtrip() {
    let stats = NormalizationStats::from_values(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let json = serde_json::to_string(&stats).unwrap();
    let back: NormalizationStats = serde_json::from_str(&json).unwrap();
    assert!((back.mean - stats.mean).abs() < 1e-12);
    assert!((back.std - stats.std).abs() < 1e-12);
    assert_eq!(back.min, stats.min);
    assert_eq!(back.max, stats.max);
}
