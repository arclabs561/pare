//! Sensitivity analysis example: is your multi-objective problem real or illusory?
//!
//! This example builds the sensitivity matrix for a concrete routing/bandit scenario
//! (3 arms, 9 covariate cells, 6 objectives) and reports:
//!
//! - Eigenvalue spectrum of the Gram matrix (how many independent tradeoff axes).
//! - Pairwise cosine similarity (which objectives are redundant).
//! - Effective Pareto dimension at several thresholds.
//! - Redundant pairs (objectives that always improve/worsen together).
//!
//! Run: `cargo run --example sensitivity_analysis`

use pare::sensitivity::{analyze_redundancy, SensitivityRow};

fn main() {
    println!("=== Objective Redundancy Analysis ===\n");

    // ---------------------------------------------------------------
    // Setup: 3 arms, 3x3 covariate grid, 6 objectives
    // ---------------------------------------------------------------
    let k = 3_usize;
    let cells_1d = [0.2_f64, 0.5, 0.8];
    let mut cells = Vec::new();
    for &x1 in &cells_1d {
        for &x2 in &cells_1d {
            cells.push((x1, x2));
        }
    }
    let m_cells = cells.len(); // 9
    let n_design = k * m_cells; // 27

    // Arm response functions (linear).
    let f = |arm: usize, x1: f64, x2: f64| -> f64 {
        match arm {
            0 => 1.0 + 0.5 * x1 + 0.3 * x2,
            1 => 0.8 + 0.1 * x1 + 0.8 * x2,
            2 => 1.2 - 0.3 * x1 + 0.2 * x2,
            _ => 0.0,
        }
    };

    // Optimal arm per cell.
    let optimal: Vec<usize> = cells
        .iter()
        .map(|&(x1, x2)| {
            (0..k)
                .max_by(|&a, &b| f(a, x1, x2).partial_cmp(&f(b, x1, x2)).unwrap())
                .unwrap()
        })
        .collect();

    let p = 1.0 / k as f64; // uniform allocation
    let idx = |a: usize, j: usize| a * m_cells + j;

    // ---------------------------------------------------------------
    // Build sensitivity vectors
    // ---------------------------------------------------------------
    let objective_names = [
        "Cumulative regret",
        "MSE arm 0",
        "MSE arm 1",
        "MSE arm 2",
        "Avg detection delay",
        "Worst-case detection",
    ];

    // 0: Cumulative regret.
    let s_regret: Vec<f64> = (0..n_design)
        .map(|d| {
            let (a, j) = (d / m_cells, d % m_cells);
            let (x1, x2) = cells[j];
            f(optimal[j], x1, x2) - f(a, x1, x2)
        })
        .collect();

    // 1-3: Per-arm MSE (D-optimal leverage).
    let mut s_mse = vec![vec![0.0; n_design]; 3];
    for a in 0..k {
        for j in 0..m_cells {
            let (x1, x2) = cells[j];
            let leverage = x1 * x1 + x2 * x2 + 1.0;
            s_mse[a][idx(a, j)] = -leverage / (p * p);
        }
    }

    // 4: Average detection delay (uniform sensitivity).
    let s_avg_det: Vec<f64> = vec![-1.0 / (p * p); n_design];

    // 5: Worst-case detection (point mass at bottleneck arm 1, cell 0).
    let mut s_wc_det = vec![0.0; n_design];
    s_wc_det[idx(1, 0)] = -1.0 / (p * p);

    let sensitivities: Vec<SensitivityRow> = vec![
        s_regret,
        s_mse[0].clone(),
        s_mse[1].clone(),
        s_mse[2].clone(),
        s_avg_det,
        s_wc_det,
    ];

    // ---------------------------------------------------------------
    // Analyze
    // ---------------------------------------------------------------
    let analysis = analyze_redundancy(&sensitivities).unwrap();

    // Print eigenvalue spectrum.
    println!(
        "Eigenvalue spectrum (Gram matrix of {} objectives over {} design points):",
        analysis.num_objectives, analysis.num_design_points
    );
    println!(
        "{:>5}  {:>14}  {:>10}  {:>12}",
        "k", "eigenvalue", "% variance", "cumulative %"
    );
    println!("{}", "-".repeat(48));
    let total: f64 = analysis.eigenvalues.iter().sum();
    let mut cumul = 0.0;
    for (i, &ev) in analysis.eigenvalues.iter().enumerate() {
        let pct = if total > 0.0 { 100.0 * ev / total } else { 0.0 };
        cumul += pct;
        println!(
            "{:>5}  {:>14.4}  {:>9.2}%  {:>11.2}%",
            i + 1,
            ev,
            pct,
            cumul
        );
    }

    // Effective dimension at several thresholds.
    println!("\nEffective dimension:");
    for &eps in &[0.10, 0.05, 0.01, 0.001] {
        let dim = analysis.effective_dimension(eps);
        println!("  eps={:.3}: {} independent axes", eps, dim);
    }
    println!(
        "  Pareto front dimension bound: {}",
        analysis.pareto_dimension_bound(1e-6)
    );

    // Pairwise cosine similarity.
    let m = analysis.num_objectives;
    println!("\nPairwise cosine similarity:");
    print!("{:>22}", "");
    for name in &objective_names {
        print!("  {:>8}", &name[..name.len().min(8)]);
    }
    println!();
    for (i, name) in objective_names.iter().enumerate() {
        print!("{:>22}", name);
        for j in 0..m {
            let c = analysis.cosine(i, j).unwrap();
            if i == j {
                print!("  {:>8}", "---");
            } else {
                print!("  {:>8.3}", c);
            }
        }
        println!();
    }

    // Redundant pairs.
    let pairs = analysis.redundant_pairs(0.95);
    if pairs.is_empty() {
        println!("\nNo nearly-redundant pairs (|cosine| > 0.95).");
    } else {
        println!("\nNearly-redundant pairs (|cosine| > 0.95):");
        for (i, j, c) in &pairs {
            println!(
                "  {} <-> {}: cosine = {:.4}",
                objective_names[*i], objective_names[*j], c
            );
        }
    }

    // Trace sanity check.
    let ev_sum: f64 = analysis.eigenvalues.iter().sum();
    let trace = analysis.trace();
    println!(
        "\nTrace sanity: eigenvalue_sum={:.4}, gram_trace={:.4}, delta={:.2e}",
        ev_sum,
        trace,
        (ev_sum - trace).abs()
    );
}
