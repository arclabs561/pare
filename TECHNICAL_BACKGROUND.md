# Technical background: Pareto frontiers, skyline queries, crowding distance, hypervolume

This note captures the technical concepts behind `pare`’s API and implementation: dominance / skyline (Pareto frontier), diversity via crowding distance, and quality measurement via hypervolume.

## Pareto dominance and the skyline / frontier

Let each candidate be an **objective vector** \(x \in \mathbb{R}^d\). Each dimension has an optimization direction:

- **maximize**: higher is better
- **minimize**: lower is better

With “all maximize” objectives, a point \(a\) **dominates** \(b\) (written \(a \succ b\)) iff:

- \(a_i \ge b_i\) for all \(i\), and
- \(a_i > b_i\) for at least one \(i\).

With mixed maximize/minimize objectives, the comparisons are flipped per dimension (this is what `pare::dominates` does).

The **Pareto frontier** (a.k.a. **skyline**) is the set of points that are **not dominated** by any other point. In database terms, a skyline query returns exactly those undominated tuples. Chomicki et al. give the standard skyline definition and basic properties in a compact survey.  
See: Chomicki et al., “Skyline Queries, Front and Back” (SIGMOD Record, 2013) PDF at `https://databasetheory.org/sites/default/files/2016-06/chomicki.pdf`.

### The nuance: dominance is a partial order (so “best” is usually ill-defined)

Dominance induces a **partial order**: many points can be **incomparable**.

In 2D maximize space, if \(a = (10, 0)\) and \(b = (0, 10)\), neither dominates the other. Both are “Pareto-optimal” with respect to each other, and both can appear on the skyline.

This matters because:

- A frontier can contain many points (even a large fraction of your dataset), especially in high dimensions or in anti-correlated distributions.
- Any procedure that tries to pick “the best” point must introduce *extra structure* (weights, lexicographic priorities, utility functions, reference points, etc.). Without that, “best” is not a well-posed question.

### Algorithmic baseline

The naive frontier computation is \(O(n^2 d)\): for each point, check whether any other point dominates it.

Practical skyline/frontier implementations use pruning and/or indexing to avoid the quadratic worst case on typical data distributions, but the worst-case can still be quadratic in \(n\) (see skyline literature referenced in the survey above).

`pare`’s primary data structure is an **incremental frontier**:

- To insert a new point \(p\), reject it if any current frontier point dominates \(p\).
- Otherwise remove any frontier points dominated by \(p\), then add \(p\).

This is a common “online skyline” pattern; it tends to work well when the maintained frontier is much smaller than the full candidate set.

### When the frontier explodes (and why this surprises people)

Two effects make frontiers/skyline results “blow up” in size:

- **More dimensions ⇒ more incomparability**: as \(d\) increases, it becomes easier for points to trade off across many axes and thus avoid being dominated.
- **Anti-correlation ⇒ more skyline points**: when objectives trade off strongly (e.g., “better quality implies higher price”), dominance becomes rarer, so the skyline grows.

The anti-correlation point shows up explicitly in skyline literature; Shang & Kitsuregawa discuss skyline behavior on anti-correlated distributions and why “nice average-case assumptions” break on real data.  
See: Shang & Kitsuregawa, “Skyline Operator on Anti-correlated Distributions” (PVLDB) PDF at `http://www.vldb.org/pvldb/vol6/p649-shang.pdf`.

The pragmatic implication for system design is that “compute the skyline then let the user pick” stops being viable when the skyline is large; you need a second-stage preference model or an approximation/archiving strategy.

#### What you can do instead (design menu)

There isn’t a single universally-right answer, but these are common options:

- **Scalarization (utility function)**: map \(x \mapsto u(x)\) and take top‑k by \(u\). This is the simplest way to force a total order, but you must accept the modeling assumptions.
- **Multiple scalarizations**: sample many weight vectors and take the union of the corresponding winners; this is a cheap way to get “coverage” without keeping the whole skyline.
- **ε‑archiving / grid archiving**: keep at most one representative per cell of a discretized objective space to cap archive size at a desired resolution.
- **Relaxed dominance variants**:
  - `pare` includes **k‑dominance** helpers (`pareto_indices_k_dominance`) which require being strictly better in at least \(k\) dimensions while being non-worse in all dimensions. This reduces frontier size, but it is a different relation than Pareto dominance, so interpret it as an approximation/heuristic rather than “the” frontier.
- **Post-filter diversity**: compute some frontier/approx-front and then select a diverse subset (crowding distance, farthest-point sampling, clustering).
- **Preference interaction**: ask the decision maker for a few comparisons or constraints and refine (common in decision analysis; less common in pure DB skyline).

### Floating point realities: strict dominance, ties, and epsilon tolerances

Real systems rarely have perfect real numbers:

- **Exact ties**: if \(a=b\), neither point strictly dominates the other by the definition above (because “strictly better in at least one dimension” fails). Many libraries choose a tie-breaking rule (e.g., dedup identical points, keep earliest, keep latest, keep both).
- **Near-ties**: when values are within floating error, “\(>\)” and “\(\ge\)” become unstable. A small tolerance \( \varepsilon \) makes comparisons more robust but changes the induced order.

`pare` uses a small numeric tolerance `eps` for dominance checks. Conceptually, it implements “strictly better by more than eps in at least one dimension, and not worse by more than eps in any dimension.”

Important distinction:

- This **numeric epsilon** is *not* the same thing as **ε-dominance archiving** (a deliberate coarsening of objective space to bound archive size). The former is about float stability; the latter is about approximation/representation trade-offs.

Also note: `pare::ParetoFrontier::<usize>::try_new` rejects NaN/∞ values up front (because dominance comparisons with NaN make the partial order meaningless).

## Crowding distance (NSGA-II)

Many multi-objective optimizers want not only a non-dominated set, but also a **diverse** spread along the front. NSGA-II popularized **crowding distance** as a density heuristic computed within a non-dominated front:

- For each objective, sort points by that objective value.
- Boundary points get infinite crowding distance.
- Each interior point accumulates a normalized neighbor gap (distance to previous + next along that objective).

NSGA-II primary reference:

- Deb et al., “A Fast and Elitist Non-Dominated Sorting Genetic Algorithm for Multi-Objective Optimization: NSGA-II” (KanGAL Report No. 200001) PDF at `http://repository.ias.ac.in/83498/1/2-a.pdf`.
- A typeset IEEE reprint is also widely mirrored as a PDF; one copy is `https://www.cse.unr.edu/~sushil/class/gas/papers/nsga2.pdf`.

`pare::ParetoFrontier::crowding_distances()` implements the “boundary is infinite; sum normalized gaps across objectives” variant.

### The nuance: crowding distance is geometry-sensitive and scale-sensitive

Crowding distance is **not** invariant to reparameterizations:

- If one objective has a range 1000× larger than another, it can dominate the distance computation unless you normalize.
- If your front is highly curved/nonlinear, “equal distance in objective coordinates” is not the same as equal diversity in decision space (the thing you actually sample).

NSGA-II uses crowding distance primarily as a **tie-breaker** within a non-dominated front when selecting a fixed population size; it is not a principled metric of global goodness.

## Hypervolume indicator (S-metric)

The **hypervolume** of a set \(P\) measures the volume of objective space **dominated** by \(P\), relative to a chosen **reference point** \(r\). For all-maximize objectives, each point \(p\) defines a box \([r_1, p_1] \times \cdots \times [r_d, p_d]\) (clamped to non-negative side lengths), and hypervolume is the Lebesgue measure of the union of these boxes.

For mixed maximize/minimize objectives, one common trick (used by `pare`) is to transform objectives into a unified “improvement over reference” maximize space:

- maximize dim: \(\max(0, p_i - r_i)\)
- minimize dim: \(\max(0, r_i - p_i)\)

Then compute hypervolume against the origin in this oriented space.

Key practical points:

- **Reference point matters**: hypervolume values are only comparable for the same \(r\) (and same orientation/scaling).
- **Computational cost grows quickly with dimension**: exact algorithms are easy in 2D, harder in higher \(d\), and the general problem is expensive in the worst case.

Good open-access survey:

- Guerreiro, Fonseca, Paquete, “The hypervolume indicator: Problems and algorithms” (2020) PDF at `https://arxiv.org/pdf/2005.00515.pdf`.

`pare::ParetoFrontier::hypervolume()`:

- orients values relative to `ref_point`,
- drops points with zero contribution in any dimension,
- filters to a non-dominated set in maximize space,
- then computes exact hypervolume (special-cases 1D and 2D; uses recursive slicing in higher dimensions).

### The nuance: hypervolume is Pareto-compliant but not scale-free

Hypervolume has a strong theoretical property (strict monotonicity w.r.t. set dominance) that makes it attractive in multiobjective optimization. Practically, though:

- **Units matter**: “latency in ms” and “accuracy in [0,1]” interact multiplicatively in volume. If you rescale latency from ms to seconds, you rescale hypervolume.
- **Normalization is a modeling decision**: normalizing objectives (e.g., to [0,1]) can improve comparability, but it also injects assumptions (which min/max? over what population? fixed bounds or dynamic bounds?).
- **Reference point selection is a preference**: choosing \(r\) implicitly says what region of objective space you consider relevant. “Bad” reference points can produce hypervolume values that are dominated by a single extreme point, masking diversity.

### The nuance: why 2D is “easy” and higher dimensions are not

In 2D maximize space, for a non-dominated set sorted by \(x\) ascending, the \(y\) values are non-increasing. Hypervolume can then be computed as a sum of rectangle strips.

In \(d \ge 3\), the union-of-boxes geometry becomes much more complex; exact computation typically uses recursive decomposition/slicing or specialized data structures, and worst-case complexity grows quickly. The Guerreiro et al. survey is the best “one stop” map of these algorithmic trade-offs.

### Hypervolume contributions and subset selection (and a non-obvious failure mode)

For a set \(P\), the **hypervolume contribution** of a point \(p \in P\) is commonly defined as:

\[
\mathrm{HVC}(p \mid P) = \mathrm{HV}(P) - \mathrm{HV}(P \setminus \{p\})
\]

This is useful in “keep exactly \(m\) points” workflows:

- If you start from a candidate set and repeatedly drop the point with the smallest contribution, you often preserve most of the dominated volume with far fewer points.
- This idea appears in hypervolume-based EMOAs and in the “hypervolume subset selection problem” (HSSP) literature (surveyed by Guerreiro et al.).

The subtlety is that “drop least-contributing point” does **not** guarantee that the *observed* hypervolume sequence behaves the way you intuitively expect once you add practical complications (notably adaptive reference points). Judt et al. show **non-monotonicity of obtained hypervolume** in a 1-greedy S‑metric selection setting under reference point adaptation, even in low dimensions.  
See: Judt et al., “Non-monotonicity of Obtained Hypervolume in 1-greedy S-Metric Selection” PDF at `https://www.gm.th-koeln.de/ciopwebpub/Judt11a.d/Judt11a.pdf`.

Takeaway for `pare` usage:

- If you plan to use hypervolume as a progress metric, be explicit and stable about the reference point (and about any rescaling).
- If you plan to use hypervolume contributions for downselection, test the behavior you care about (monotonicity, diversity retention) on representative data; the indicator is principled, but your overall procedure may not be.

## ε-dominance and archiving (why it exists)

In many-objective optimization, the true non-dominated set can grow very large. **Archiving** strategies reduce this set while preserving coverage, often by introducing a resolution parameter \( \varepsilon \) (an “indifference” threshold) so points that are too close are treated as effectively equivalent.

One place to start for background on archiving and indicator-based evaluation is Laumanns’ dissertation, which discusses hypervolume-related indicators and archiving ideas in depth:

- Laumanns, dissertation PDF at `https://pub.tik.ee.ethz.ch/people/thiele/paper/diss_laumanns.pdf`.

`pare` does not currently implement ε-archiving as a first-class feature (it uses an `eps` tolerance for floating comparison in dominance), but the concept is relevant if you later want to bound frontier size or enforce a “grid” resolution.

## Mapping back to `pare` (what to expect)

- **Frontier maintenance**: `ParetoFrontier::push` is an online skyline update; worst case is linear in current frontier size times dimension.
- **Comparisons**: `dominates` is direction-aware with an epsilon tolerance; this affects edge cases when values are extremely close.
- **Scoring**: `scalar_score` normalizes per-dimension to \([0,1]\) using min/max over the current frontier and flips for `Minimize`; it is a convenience for picking a single point given weights, not a substitute for multi-objective reasoning.
- **Hypervolume**: exact hypervolume is useful for evaluation but can be expensive at higher dimension; consider approximation or contributions if you extend `pare`.

### One more nuance: frontier order is not “sorted”

When you call `ParetoFrontier::<usize>::try_new(...).indices()`, the returned indices are in the frontier’s **internal insertion order**, not in any guaranteed sorted-by-objective order. If you need a stable order, sort by a key explicitly (e.g., by one objective or by `data`).

## References (open access)

- Chomicki et al. (2013), “Skyline Queries, Front and Back” (SIGMOD Record) PDF: `https://databasetheory.org/sites/default/files/2016-06/chomicki.pdf`
- Deb et al. (2000/2002), NSGA-II technical report PDF: `http://repository.ias.ac.in/83498/1/2-a.pdf`
- Deb et al. (2002), NSGA-II IEEE reprint PDF (mirror): `https://www.cse.unr.edu/~sushil/class/gas/papers/nsga2.pdf`
- Guerreiro et al. (2020), “The hypervolume indicator: Problems and algorithms” PDF: `https://arxiv.org/pdf/2005.00515.pdf`
- Shang & Kitsuregawa (PVLDB), “Skyline Operator on Anti-correlated Distributions” PDF: `http://www.vldb.org/pvldb/vol6/p649-shang.pdf`
- Judt et al. (2011), “Non-monotonicity of Obtained Hypervolume in 1-greedy S-Metric Selection” PDF: `https://www.gm.th-koeln.de/ciopwebpub/Judt11a.d/Judt11a.pdf`
- Vinati (2022), “Flexible Skyline: one query to rule them all” PDF: `https://arxiv.org/pdf/2201.05096`
- Lupi (2022), “Multi-objective optimization… through flexible skyline queries” PDF: `https://arxiv.org/pdf/2202.09857`
