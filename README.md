# Lex-BAC: Lexicographic Bellman Atlas Continuation

This repository implements Lex-BAC for finite discounted homotopy MDPs:

\[
P^\lambda=(1-\lambda)P^0+\lambda P^1,\qquad
r^\lambda=(1-\lambda)r^0+\lambda r^1,\qquad \lambda\in[0,1].
\]

The exact certificate layer uses `sympy.Rational`.  Floating-point routines are
used only for plotting, timing, and the SC-IPR predictor.  The independent
verifier never reads solver caches.

Current main-learning boundary.  The learning algorithm studied as the
non-enumerative route is the CRAC residual actor-critic family:
`certificate_residual_flow.py` for exact finite-state analysis and
`neural_residual_actor.py` for sampled actor-critic policies.  This main learning
algorithm does not enumerate deterministic policies, does not maintain an
offline policy library, and does not first solve for a known good policy and
then imitate it.  Legacy atlas, cover, and ODD-library modules remain in the
repository as exact verification baselines, deployment-library tools, and
negative/contrast experiments; they are not the claimed online neural learning
mechanism.

## Files

- `mdp.py`: rational finite MDP data structures.
- `certslack.py`: exact Bellman slack rational functions.
- `solver.py`: global exact Lex-BAC atlas solver.
- `solver_online.py`: local sampled/high-probability online route; no hard certificates.
- `verifier.py`: independent exact verifier, \(O(n^3+m)\).
- `certified_policy_generator.py`: counterexample-guided online policy generation; no policy library.
- `certificate_residual_flow.py`: continuous Bellman-residual policy dynamics with safety-preserving line search.
- `neural_residual_actor.py`: Certified Residual Actor-Critic (CRAC) with critic proposals and high-probability monotonic/safety gates.
- `phasediagram.py`: phase diagram query and visualization.
- `robustness_report.py`: robust intervals, fragile region, curriculum nodes.
- `predictor.py`: numerical SC-IPR log-barrier center-path predictor.
- `corrector.py`: Lex-BAC hard-certificate corrector.
- `pc_bac.py`: PC-BAC fusion interface.
- `examples.py`: reproducible rational MDP examples.
- `innovations.py`: A1/A3 mechanisms plus legacy diagnostics.
- `epsilon_cover.py`: corrected A2 epsilon-cover by few policies.
- `adaptive_cover.py`: Witness-BAC and Bernstein-expanded Witness-BAC policy covers.
- `bernstein_cert.py`: root-free Bernstein-Bellman interval certificates.
- `certificate_compression.py`: minimum certificate-preserving interval cover.
- `switch_cover.py`: switch-aware certified chart selection.
- `risk_weighted_cover.py`: risk-weighted covers and uncertainty-minimax certified selection.
- `semantic_odd_cover.py`: semantic ODD policy-cover certificates for deployment.
- `odd_experiments.py`: drone/highway ODD runtime-selection experiments.
- `highway_env_validation.py`: public `highway-env` validation experiments.
- `ev_autonomous_validation.py`: EV energy/SOC validation on public highway-env tasks.
- `ev_autonomous_analysis.py`: confidence intervals and Pareto analysis for EV validation.
- `experiments.py`: timing and phase-diagram scripts.
- `model_free_benchmarks.py`: drone/highway-style model-free safety benchmarks.
- `crac_paper_benchmarks.py`: CRAC paper-style performance, mechanism, safety, and dependency matrices.
- `tests/`: correctness tests.

## Core Theorems

### 0. Main Result: Solve-Verify Asymmetry

**Theorem 0 (Solve-Verify Asymmetry, model-dependent).** The asymmetry claim has
three separate parts.  It does **not** assert that computing one optimal policy
for a single endpoint MDP \(M_1\) is exponentially hard; finite discounted MDPs
have standard polynomial-time solution methods.  The exponential lower-bound
claims below are restricted to explicit atlas-output and oracle-query models.

**Theorem 0a (Polynomial independent verification).** For rational finite
discounted \(M=(P,r,\gamma)\) and deterministic policy \(\pi\), define:

\[
\operatorname{Accept}(M,\pi)
\iff
\left[
V=(I-\gamma P_\pi)^{-1}r_\pi
\wedge
\forall(s,a),\ V(s)-r(s,a)-\gamma P_aV\ge0
\right].
\]

Soundness:

\[
\forall M,\pi,\ \operatorname{Accept}(M,\pi)\Rightarrow \pi\in\Pi^*(M).
\]

Completeness for deterministic optima:

\[
\forall M,\pi,\ \pi\in\Pi^*(M)\Rightarrow \operatorname{Accept}(M,\pi).
\]

Non-uniqueness:

\[
\operatorname{Accept}(M,\pi)
\]

certifies only \(\pi\in\Pi^*(M)\), not that \(\pi\) is unique and not that it is
the LexOpt representative.  The verifier performs one exact \(n\times n\) linear
solve plus \(m\) exact slack checks, so the arithmetic operation count is

\[
O(n^3+m).
\]

**Theorem 0b (Output-sensitive exponential atlas size).** In the
critical-region enumeration/oracle model, the compact family
`IndependentFlipMDP(k)` has \(k\) binary policy bits and

\[
K(k)=2^k
\]

regions.  Region \(j\) has policy label \(\operatorname{Gray}(j)\), and each
neighboring pair differs in exactly one bit.  The critical points are

\[
\lambda_j=j/2^k,\qquad j=1,\dots,2^k-1.
\]

The encoding length is polynomial in \(k\): store \(k\), the Gray-code rule, and
denominators of bit length \(O(k)\).  Any algorithm that outputs the full atlas
must output \(K(k)\) regions, hence requires \(\Omega(2^k)\) output-sensitive
time in this model.

Compatibility with the degree bound: no single slack curve is responsible for
exponentially many roots.  Each boundary is assigned a distinct policy-pair
witness slack id, so the exponential count comes from exponentially many policy
configurations in the arrangement.  This is exactly the source allowed by
\(\deg p^\pi_{s,a}\le n+1\).

**Theorem 0c (Oracle query lower bound for ATLAS-SWITCH).** In the
optimal-action oracle model, an algorithm may query
\(\mathcal O(\lambda)\), which returns the optimal action label at that
\(\lambda\).  Define ATLAS-SWITCH:

\[
\text{decide whether state }s_0\text{ switches at least }2^{ck}\text{ times.}
\]

For \(c=1/2\), after any set of \(q\) queried \(\lambda\)-points with
\(q\le 2^k-2^{k/2}-1\), the adversary in `atlas_lower_bound.py` constructs two
oracles that agree on every queried point:

- NO oracle: action \(0\) everywhere, switch count \(0\);
- YES oracle: queried cells fixed to \(0\), unqueried cells alternate actions,
  giving at least \(2^{k/2}\) switches.

Therefore any oracle algorithm that always decides ATLAS-SWITCH requires
\(2^{\Omega(k)}\) queries in this model.  This is a model-dependent oracle lower
bound, not an unconditional super-polynomial lower bound for explicit MDP
optimization.

Figure 1 is generated at `outputs/figure1_solve_verify_asymmetry.png`; its
source data are in `outputs/asymmetry_report.csv`.  The figure combines exact
Lex-BAC solver timings where enumeration is run, the oracle-query lower-bound
extension, and independent verifier timings.

The explicit atlas-size witness is plotted in
`outputs/exponential_atlas_regions.png`, with source data in
`outputs/exponential_atlas_regions.csv`.

### 1. Bellman Slack Certificate

For a deterministic policy \(\pi\),

\[
V^\pi_\lambda=(I-\gamma P^\lambda_\pi)^{-1}r^\lambda_\pi.
\]

Let

\[
q^\pi(\lambda)=\det(I-\gamma P^\lambda_\pi).
\]

Since \(P^\lambda_\pi\) is stochastic and \(0<\gamma<1\),
\(I-\gamma P^\lambda_\pi\) is a nonsingular M-matrix for every
\(\lambda\in[0,1]\), so \(q^\pi(\lambda)\neq0\) on the homotopy interval.

Using the adjugate formula,

\[
q^\pi(\lambda)V^\pi_\lambda
=
\operatorname{adj}(I-\gamma P^\lambda_\pi)r^\lambda_\pi.
\]

The Bellman slack

\[
\Delta^\pi_{s,a}(\lambda)
=
V^\pi_\lambda(s)-r^\lambda(s,a)-\gamma P^\lambda_aV^\pi_\lambda
\]

is therefore represented exactly as

\[
\Delta^\pi_{s,a}(\lambda)=p^\pi_{s,a}(\lambda)/q^\pi(\lambda).
\]

The numerator is constructed by exact common-denominator arithmetic:

\[
p^\pi_{s,a}(\lambda)
=
N_s(\lambda)
-r^\lambda(s,a)q^\pi(\lambda)
-\gamma\sum_{s'}P^\lambda_a(s,s')N_{s'}(\lambda),
\]

where

\[
N(\lambda)=\operatorname{adj}(I-\gamma P^\lambda_\pi)r^\lambda_\pi.
\]

Degree bound:

- entries of \(I-\gamma P^\lambda_\pi\) are affine in \(\lambda\);
- \(q^\pi\) has degree at most \(n\);
- adjugate entries have degree at most \(n-1\);
- \(r^\lambda_\pi\) is affine, so each \(N_s\) has degree at most \(n\);
- \(N_s\) contributes degree \(\le n\);
- \(r^\lambda(s,a)q^\pi\) contributes degree \(\le n+1\);
- \(P^\lambda_a(s,s')N_{s'}\) contributes degree \(\le n+1\).

Therefore

\[
\deg p^\pi_{s,a}\le n+1.
\]

This is implemented in `certslack.py` and tested on randomized rational MDPs.

### 2. Exact Optimality Verifier

`verifier.py` receives only \(M_1,\gamma,\pi\).  It solves

\[
(I-\gamma P^1_\pi)V=r^1_\pi
\]

using exact rational arithmetic, then checks

\[
V(s)-r^1(s,a)-\gamma P^1_aV\ge0,\qquad \forall(s,a).
\]

If all inequalities hold, then

\[
V(s)=r^1(s,\pi(s))+\gamma P^1_{\pi(s)}V
\]

and all other actions are no better.  Hence \(V=T_1V\).  The Bellman optimality
operator is a \(\gamma\)-contraction in sup norm, so its fixed point is unique.
Therefore \(V=V^*_1\) and \(\pi\) is globally optimal.

Conversely, any globally optimal deterministic policy satisfies the same
inequalities, so the verifier is complete for deterministic optimal policies.
The verifier decides only whether \(\pi\in\Pi^*_1\).  It does not decide whether
\(\pi\) is the LexOpt representative; if several deterministic policies are
optimal, all of them are accepted.

The verifier performs one \(n\times n\) linear solve and one pass over \(m\)
inequalities:

\[
O(n^3+m).
\]

### 3. Finite Atlas Without Connected-Chart Assumption

The solver does not assume that a policy's optimality set

\[
C_\pi=\{\lambda:\Delta^\pi_{s,a}(\lambda)\ge0,\ \forall(s,a)\}
\]

is connected.  It may be disconnected.

Instead, the global exact atlas enumerates deterministic policies and collects
all real roots in \([0,1]\) of all non-policy slack numerators
\(p^\pi_{s,a}\).  These roots partition \([0,1]\) into finitely many open
intervals.  On each such interval every nonzero slack numerator has constant
sign, hence the set of optimal policies is constant.

For an interval \((\alpha_i,\alpha_{i+1})\), where \(\alpha_{i+1}\) is the
global next critical point in the sorted root arrangement \(\mathcal R\), the
solver computes
\(\operatorname{LexOpt}(\alpha_i^+)\) by evaluating every policy at

\[
\alpha_i+\delta_{\text{probe}},
\qquad
0<\delta_{\text{probe}}<(\alpha_{i+1}-\alpha_i)/2,
\]

using exact rational arithmetic.  This is a right-limit choice, not a closed
point choice at \(\alpha_i\).  Permanently tied actions with
\(p^\pi_{s,a}\equiv0\) do not create roots, but they are still included in the
slack checks used to construct the right-limit optimal policy set.

The worst-case bound used by the implementation is

\[
1+(n+1)\sum_s(|A_s|-1)\prod_s|A_s|.
\]

For uniform action count \(A\), this becomes

\[
1+(n+1)n(A-1)A^n.
\]

This is exponential in \(n\).  The code reports the bound and never hides this
cost.

### 4. Lexicographic Tie-Breaking

The implementation does not perturb the scalar rewards used by the verifier.
Lexicographic tie-breaking is a selection rule over each exact right-limit
optimal policy set:

\[
\operatorname{LexOpt}(\alpha_i^+)=
\min_{\text{lex}}\Pi^*(\alpha_i^+).
\]

Therefore it cannot change the original MDP's optimal value set.  In tied test
MDPs, every tied policy is still verifier-accepted and has the same exact value;
Lex-BAC only chooses the canonical representative.

Because the exact solver uses a global arrangement, it does not perform a pivot
sequence and has no cycling mode to rule out.  Any future local online tracker
must live separately from `solver.py` and carry its own perturbation and
anti-cycling proof.

### 5. PC-BAC Certificate Semantics

`predictor.py` implements a numerical log-barrier center path:

\[
\min_V c^\top V-\mu\sum_{s,a}\log b_V(s,a),
\qquad
b_V(s,a)=V(s)-r^\lambda(s,a)-\gamma P^\lambda_aV.
\]

The predictor gives a soft dual-gap certificate.  The hard certificate is
provided only by `corrector.py`, which calls the exact Lex-BAC atlas, detects
which exact phase roots the predictor crossed, and then calls the independent
verifier.  `PCBACResult.segment_certificates` records this per predictor
segment: the segment's soft dual-gap bound plus the exact crossed phase roots
for that same lambda interval.

The model-free entry point uses `solver_online.py`: Robbins-Monro empirical
updates, pessimistic slack lower bounds, and a lexicographic symbolic
perturbation order for local tie-breaking.  The online selector refuses to
return a policy already used at the same local candidate set, which is the
anti-cycling mechanism for that local route.  It is explicitly marked
high-probability only and does not claim exact certification.

### 6. Model-Free Lex-BAC Certificate Semantics

The repository enforces three certificate layers:

- `exact`: only model-based rational arithmetic plus the independent verifier;
- `high_prob`: sampled/model-free statements with explicit confidence
  \(1-\delta\), sample counts \(N\), and confidence radius parameters;
- `none`: diagnostics, baselines, or heuristic curves without certification.

Every CSV produced by the experiment scripts contains a nonempty
`certificate_type` column in `{exact, high_prob, none}`.  Model-free outputs are
tested to reject accidental `exact` labels.

**Lemma B1 (Pessimistic sampled slack).** For a fixed value vector \(V\), define
one-step TD slack samples

\[
X_i(s,a)=V(s)-r_i-\gamma V(s'_i),\qquad s'_i\sim P(\cdot|s,a).
\]

Let \(\hat b(s,a)\) be the sample mean over \(N(s,a)\) fresh holdout samples and

\[
\tilde b(s,a)=\hat b(s,a)
-\kappa\sqrt{\frac{\log(SA/\delta)}{N(s,a)}},
\qquad \kappa=\gamma V_{\max}.
\]

By Hoeffding and a union bound over all state-action pairs, with probability at
least \(1-\delta\), \(\tilde b(s,a)\le b_{\text{true}}(s,a)\) for every
\((s,a)\).  Hence any feasible set constructed from \(\tilde b\ge\tau\) is
contained in the true feasible set on that event.  This is implemented by
`OnlineBACSolver.td_pessimistic_slack_lower_bounds`.

**Lemma B2 (Estimated phase tracking and anti-cycling).** Online phase changes
are detected only as estimated sign-change brackets of pessimistic slack; they
are not exact roots.  At each local candidate set, `OnlinePhaseTracker` uses
the symbolic lexicographic selector, equivalent to the infinitesimal reward
perturbation \(r(s,a)+\eta^{\operatorname{idx}(s,a)}\) with \(\eta\to0^+\),
and refuses to return a policy already used at that same local point.  Because a
finite local candidate set contains only finitely many deterministic policies,
the local tracker either returns a new policy or exhausts the set in finite
time.  Under Lemma B1's high-probability event, this finite tracking occurs
inside the pessimistic feasible region.

**Theorem B3 (Projected Robbins-Monro gap split).** The online route updates
\(V\) with Robbins-Monro steps and projects onto the convex polytope
\(\{V:B V-r\ge\tau\}\).  For \(K\) iterations, center-path parameter \(\mu_K\),
and \(N\) holdout samples per state-action pair, the implemented bound is

\[
J^*-J(\pi_K)
\le
\frac{(SA+m)\mu_K}{1-\gamma}
+\frac{C V_{\max}}{1-\gamma}
\sqrt{\frac{\log(SA/\delta)}{N}}.
\]

The first term is the center-path bias; the second is the sampling error.  The
CSV `outputs/b5_model_free_gap.csv` reports these two terms separately and
labels the certificate as `high_prob`.

**Theorem B4 (Anytime safety margin).** For safety cost \(g_i\), limit \(G_i'\),
and barrier multiplier upper scale \(\bar\lambda\), the safety certificate checks

\[
\sum_s d_{\mu_k}(s)g_i(s)\le G_i'-\mu_k/\bar\lambda.
\]

Thus every center-path iterate that passes the check is strictly safe with
explicit margin \(\mu_k/\bar\lambda\).  This differs from CPO/Lagrangian-style
baselines in the benchmark scripts: their curves are plotted as uncertified
baselines (`certificate_type=none`), while online Lex-BAC safety curves are
`high_prob`.

**Theorem B5 (Certified Residual Actor-Critic, CRAC).**
`neural_residual_actor.py` upgrades the residual-flow route to a sampled
actor-critic.  The critic proposes; the certificate gate accepts.  Concretely,
`ResidualCritic.fit` estimates transition/reward statistics from training
samples and computes a Bellman advantage residual \(\widehat A_\psi(s,a)\).
The softmax actor \(\pi_\theta(a|s)\) then takes the continuous logit update

\[
\theta_{s,a}^{+}=\theta_{s,a}
+\eta\left(\widehat A_\psi(s,a)
-\sum_b\pi_\theta(b|s)\widehat A_\psi(s,b)\right),
\]

which is the logit-space version of the residual/replicator direction.  This
is an end-to-end actor update, not a stored policy library and not a
state-action replacement rule.  The critic is not a certificate; it is only the
proposal generator.  Every accepted CRAC update is certified by a separate
holdout gate.

Continuous-control CRAC uses a TD3-style critic proposal backbone by default:
two critics, target-Q bootstrapping, clipped target-policy noise, and a
pessimistic minimum-Q actor objective.  This changes the candidate generator
only.  It does not weaken or replace the holdout certificate; MuJoCo rows
therefore record both `critic_mode` and `lcb_method`.

The MuJoCo path now also includes an online residual-action shooting generator.
When enabled, CRAC does not load a policy library.  At each training round it
generates fresh coordinate and CEM-style action-bias residual candidates from
the current actor, screens them with a small uncertified rollout budget, and
then sends only the selected candidate to the independent holdout gate.  The
screening step is explicitly not a certificate; CSV rows therefore record the
separate `proposal_source` field (`critic`, `axis_shooting_*`, or
`bias_shooting_*`) while `certificate_type` remains `high_prob` only when the
independent gate is used.

The candidate actor is accepted only after a separate holdout rollout gate.
For continuous-control experiments the implemented gate uses paired
common-random-number rollouts and the per-step clipped improvement
\(X_i=\bar r_i(\pi_{\theta^+})-\bar r_i(\pi_\theta)\).  If rewards are clipped
to \([-c,c]\), then \(X_i\in[-2c,2c]\), so the range length is \(4c\).  With
\(N\) paired rollouts and per-bound confidence budget \(\delta_t\), the
implemented Hoeffding lower confidence bound is

\[
\mathbb E[X]
\ge
\widehat X
-4c\sqrt{\frac{\log(2/\delta_t)}{2N}}.
\]

The implementation also exposes an `empirical_bernstein` option.  For sample
variance \(\widehat V\), it uses the conservative one-sided empirical Bernstein
radius

\[
\sqrt{\frac{2\widehat V\log(2/\delta_t)}{N}}
+
\frac{7(4c)\log(2/\delta_t)}{3(N-1)}.
\]

This is still a high-probability soft certificate, not an exact Bellman
certificate.  With small holdout budgets its constant term can be larger than
Hoeffding; the CSV therefore records `lcb_method`, `clipped_lcb_radius`, and the
diagnostic `required_eval_episodes_for_positive_lcb` rather than claiming that
Bernstein always improves the gate.  In the finite-control CRAC benchmark, the
same idea is reported as a rollout lower bound on return improvement.  The
accepted update carries the high-probability monotonic-improvement lower bound

\[
J(\pi_{\theta^+})-J(\pi_\theta)
\ge
\widehat J(\pi_{\theta^+})-\widehat J(\pi_\theta)
-\operatorname{rad}_R(\theta^+)-\operatorname{rad}_R(\theta)
-\operatorname{tail}_R(\theta^+)-\operatorname{tail}_R(\theta).
\]

For every safety cost \(c_i\), the same holdout gate computes an upper
confidence bound

\[
\widehat C_i(\pi_{\theta^+})+\operatorname{rad}_{c_i}
+\operatorname{tail}_{c_i}\le G_i.
\]

Only candidates passing both gates are executed.  Therefore, on the joint
holdout event, the accepted CRAC actor sequence is approximately monotone and
does not break the certified safety limits.  The theorem is explicitly
model-free/high-probability: it does not claim an exact certificate for critic
weights, actor weights, or unknown dynamics, and every output row is labeled
`certificate_type=high_prob`.

Backtracking and multiple testing.  The implementation treats line search as a
finite sequence of candidate policies.  If at most \(B\) backtracking candidates
can be tried per iteration, and there are \(q\) safety constraints, the
per-bound confidence budget is chosen conservatively as

\[
\delta_t =
\frac{\delta}{K B (2+q)}
\]

for \(K\) planned iterations.  The factor \(2+q\) covers the current-return
bound, candidate-return bound, and candidate safety-cost upper bounds; the
factor \(B\) prevents the accepted candidate from benefiting from repeated
testing.  The CSV records both `max_line_search_trials` and
`line_search_trial` so this accounting can be audited.

The drone benchmark uses the tabular state description
`(position, velocity, battery)` with obstacle-distance and battery constraints.
The highway benchmark is a CI-friendly highway-env-style tabular surrogate with
TTC and speed constraints; the external `highway-env` Python package is not a
runtime dependency.

The stronger public-library validation is in `highway_env_validation.py`, which
uses the actual `highway-env` Gymnasium package.  It evaluates:

- `highway-fast-v0` for obstacle avoidance / collision-free survival;
- `intersection-v0` for goal-reaching via `arrived_reward`;
- random, idle, aggressive, cautious, and two Lex-BAC ODD variants.

The two Lex-BAC variants use the same exact ODD policy library but different
runtime risk calibrations: a progress-oriented calibration and a safer
calibration.  For `intersection-v0`, the progress calibration uses a smaller
ODD risk scale so it matches the aggressive/idle arrival rate in the current
run, while the safe calibration trades arrival rate for the highest
collision-free rate among the listed controllers.  This is reported as a
safety/success Pareto tradeoff, not as a claim that one setting dominates all
baselines.

`highway_env_analysis.py` post-processes the per-episode data with Wilson
intervals for Bernoulli rates, bootstrap confidence intervals for mean reward
and episode length, and Pareto non-dominance over collision-free rate, success
rate, reward, and distance proxy.

The public highway-env harness now also includes a certified risk-calibration
sweep.  It runs the same exact ODD certificate while varying only the runtime
risk scale, then plots the resulting safety/success frontier against the
hand-written random, idle, aggressive, and cautious baselines.  This is a
mechanism validation for certificate-guided risk calibration: certified rows
remain `certificate_type=odd_exact`, baselines remain `certificate_type=none`,
and the result is interpreted as a Pareto frontier rather than a claim that one
single calibration dominates every baseline in every metric.

`safety_gymnasium_validation.py` adds a public Safety-Gymnasium layer.  Because
Safety-Gymnasium pins an older Gymnasium/MuJoCo stack than the main MuJoCo and
highway-env harnesses, the local run used an isolated compatibility environment
for the public benchmark and kept the main Python environment unchanged.  On
`SafetyCarCircle1-v0`, the safety gate rejects two reward-improving candidates
whose cost UCB exceeds the limit, then accepts a later candidate with positive
reward LCB and cost UCB below the limit.  The resulting `CRAC-safety-gated`
row has lower reward than the unsafe no-gate candidate, but much lower cost and
a higher constraint-satisfaction rate.  On `SafetyPointGoal1-v0`, no candidate
achieves positive reward LCB under the current sample budget, so the certified
controller correctly stays conservative.

`validation_dashboard.py` aggregates the completed finite-control, model-free,
long-horizon, MuJoCo, SB3, Safety-Gymnasium, highway-env, and EV outputs into
`top_level_algorithm_comparison.csv` and `top_level_reward_curves.csv`, plus a
compact curve index figure.  These dashboard rows are post-processing views;
they preserve the original `certificate_type`, `confidence`, `delta`, and
`samples_N` fields rather than creating a new certificate.

### 6.1 CRAC Paper-Style Validation Matrix

`crac_paper_benchmarks.py` is the current high-standard mechanism/performance
validation harness for CRAC.  It runs five finite-control scenarios over
multiple seeds:

- `reward_bandit`: pure reward learning;
- `safe_bandit`: reward improvement under a hard safety budget;
- `delayed_chain`: long-horizon delayed reward;
- `risky_shortcut`: high reward is unsafe, so the safety gate should reject it;
- `noisy_recovery`: stochastic transitions with a safety budget.

The matrix compares:

- full `CRAC`;
- `CRAC-no-safety-gate`;
- `residual-no-certificate`;
- `no-critic`;
- `conservative-safe`;
- `uniform-random`.

The expected mechanism pattern is not that CRAC has the highest raw reward in
every unsafe task.  Rather, CRAC should improve over no-critic/uniform when
there is certified safe improvement, and it should refuse reward-increasing
updates when the high-probability safety UCB would violate the limit.  The
uncertified residual and no-safety-gate ablations are allowed to get higher
raw reward only by violating safety; that is precisely the mechanism test.

The script also records external benchmark dependency status.  In this local
run, `gymnasium`, `mujoco`, `highway_env`, `stable_baselines3`, and `torch` are
available, while `safety_gymnasium` and `metadrive` are not installed.  Those
rows are labeled `certificate_type=none`; they are dependency evidence, not
algorithm certificates.

`mujoco_crac_benchmarks.py` is the continuous-action MuJoCo smoke harness.  It
uses real Gymnasium MuJoCo environments and compares:

- `CRAC-gated`: continuous Gaussian actor-critic with the high-probability
  holdout gate;
- `CRAC-no-gate`: the same critic-proposed candidate executed without the gate;
- `zero-action`: a strong neutral-action baseline, included so that good
  initialization is not mistaken for algorithmic progress;
- `random`: random action baseline.

The current local smoke run covers `HalfCheetah-v5` and `Hopper-v5`.  It is a
path-validation and conservatism diagnostic, not yet a full MuJoCo paper run.
The actor now starts from the neutral action, and the CSV reports both raw
return and the per-step clipped certificate metric used by the high-probability
gate.  With only a few holdout episodes, the strict bounded high-probability
LCB remains conservative and currently rejects every critic-proposed candidate.
This is reported honestly through `acceptance_rate=0.0`; the no-gate ablation
shows that the rejected critic update degrades performance in the present smoke
run, while the gated CRAC row shows what the certificate allows under the
current budget.

The same MuJoCo harness now also writes a compact multi-round training curve
for `CRAC-gated-train` (twin-Q + paired Hoeffding),
`CRAC-gated-bernstein-train` (twin-Q + paired empirical Bernstein),
`CRAC-gated-bias-train` (online axis/CEM residual shooting + paired empirical
Bernstein), `CRAC-gated-singleq-train` (single-Q ablation), and
`CRAC-no-gate-train`.
These rows expose the candidate generator separately from the certificate gate:
the CSV records the deployed return, raw candidate improvement, empirical
clipped improvement, `critic_mode`, `proposal_source`, `lcb_method`, LCB
radius, and the number of holdout episodes that would be required to certify a
positive LCB under the current bound.  This is a mechanism diagnostic, not yet
a full benchmark claim: in the current run the gate prevents unstable no-gate
updates, while the small training-curve holdout budget remains too conservative
for most nontrivial accepted MuJoCo updates.

To separate "candidate is weak" from "certificate budget is too small", the
harness also writes a certificate budget probe.  It reruns the first CRAC
candidate with a larger independent holdout budget.  In the current local run,
`Hopper-v5` becomes accepted under `certificate_type=high_prob` once the
empirical-Bernstein holdout budget is raised to 512 episodes, while
critic-only `HalfCheetah-v5` remains rejected at 512 episodes because the
critic candidate's paired clipped improvement is essentially zero.  With the
online network-shooting generator enabled, the selected HalfCheetah proposal
comes from `network_shooting_17`, improves raw return by about `+41.43`, and is
accepted once the independent holdout budget is raised to 1536 episodes
(`clipped_improvement_lcb > 0`).  This is the current clearest diagnosis of the
MuJoCo gap: critic-only HalfCheetah is candidate-quality-limited, but online
candidate generation can repair the proposal while preserving the certificate
semantics.

The harness also writes a certified accepted-update curve.  Unlike the budget
probe, this curve actually applies accepted candidates and continues training.
In the current local run, `HalfCheetah-v5` accepts the first online
network-shooting update and deploys a policy with raw return about `41.21`; the
next candidate is rejected because its LCB is negative, so the deployed policy is not
silently degraded.  `Hopper-v5` shows the same mechanism: the first critic
candidate is accepted, while a later large negative-return candidate is
rejected; the same gate continues to reject harmful critic proposals.  This is
still a compact validation curve rather than a full MuJoCo
leaderboard claim, but it demonstrates that the certificate can drive real
updates, not only annotate one-step probes.

For certified seed robustness, the harness additionally reruns certified
updates over seeds `0,1,2` on `HalfCheetah-v5`, `Hopper-v5`, and
`Walker2d-v5`.  The current table records the final deployed policy after the
accepted high-probability updates and counts the independent holdout samples
used by accepted and rejected certificate checks.  HalfCheetah reaches mean raw
return about `45.89` with minimum clipped-improvement LCB `0.113`.  Hopper now
uses a two-stream certified proposal portfolio: each stream is checked at
`delta=0.025`, and the selected stream is therefore a union-bound
`confidence=0.95` deployment.  This raises Hopper to mean raw return about
`170.13`, with minimum deployed-stream LCB `0.021`.  Walker2d uses two
certified update rounds and reaches mean raw return about `108.70`, with
minimum LCB `0.006`.  All three environments have acceptance rate `1.0` for
the deployed certified update sequence, so the table is now a
multi-environment seed-robustness check rather than a single accepted-update
smoke result.  The stream-level Hopper evidence is written to
`outputs/mujoco_crac_portfolio_streams.csv` and `.png`.

The candidate generator ablation separates the proposal mechanism from the
certificate gate.  On `HalfCheetah-v5`, critic-only proposals have essentially
zero mean raw improvement, axis/bias shooting improves by about `+5.64`, and
network shooting improves by about `+47.77` under the same ablation harness.
On `Walker2d-v5`, critic-only remains rejected, while both axis/bias shooting
and network shooting are accepted by the high-probability gate at the ablation
budget.  This supports the mechanism claim that online proposal generation is
doing real algorithmic work; the certificate is not merely blessing an ordinary
critic update.

The continuous-control safety gate ablation uses action-energy cost as a simple
MuJoCo safety budget.  On `HalfCheetah-v5`, the same high-reward network
proposal has positive reward LCB and would be accepted by the no-safety variant,
but its high-probability safety UCB is above the limit, so `CRAC-safety-gate`
rejects it.  On `Walker2d-v5`, the selected proposal has positive reward LCB
and safety UCB below the same limit, so the safety gate accepts it.  The output
therefore separates `certificate_type=high_prob` for reward improvement from
`safety_certificate_type`, which is `none` for the no-safety ablation and
`high_prob` for the safety-gated row.

The horizon-transfer validation evaluates the deployed certified actors at
80, 150, and 300 environment steps.  The update certificate itself only covers
the 80-step window, so longer horizons are labeled with
`evaluation_certificate_type=none`.  In the current local run, the accepted
CRAC actor remains above the zero-action baseline at all reported horizons on
both `HalfCheetah-v5` and `Walker2d-v5`; for example the 300-step raw returns
are about `180.69` and `239.69`, respectively.  This does not extend the formal
certificate beyond its stated horizon, but it checks that the short-horizon
proposal is not merely exploiting a one-window artifact.

The MuJoCo statistical report adds paired effect sizes and bootstrap 95%
confidence intervals over the existing CSV outputs.  It covers horizon-transfer
CRAC-vs-zero differences, candidate-generator paired effects, and the safety
gate mechanism.  These rows use `certificate_type=none` because the report is
statistical post-processing rather than a new certificate.  In the current
local run, every horizon-transfer CRAC-vs-zero paired CI is strictly positive,
and the key candidate-generator effects over critic-only proposals are also
positive.

The same script also includes SB3 baselines for `PPO`, `SAC`, and `TD3`.  The
smoke runner remains a dependency/code-path check.  The stronger
sample-budget comparison runs an SB3 budget sweep at 800, 5000, and 20000
training steps with the same 80-step evaluation horizon used by the CRAC
certificate window, then writes `mujoco_sample_budget_comparison.csv`.  The
current local sweep covers `HalfCheetah-v5`, `Hopper-v5`, and `Walker2d-v5`,
with three seeds for every SB3 budget point.  On HalfCheetah,
`CRAC-certified` reaches mean raw return about `45.89`, above the best
completed SB3 row (`TD3-20000`, about `42.70`).  On Hopper,
the certified proposal portfolio reaches about `170.13`, above the best current
SB3 raw row (`SAC-5000`, about `166.63`).  On Walker2d, the two-round
certified policy reaches about `108.70`, slightly above the current
`TD3-20000` raw-return row (`108.41`).  This is still not a final MuJoCo
leaderboard claim because CRAC uses many more independent holdout samples, but
it is no longer a single-environment or 800-step strawman comparison.

`mujoco_baseline_gap_report.csv` makes the raw-return comparison explicit by
pairing each CRAC-certified row with the strongest completed SB3 row in the
same environment.  The report is labeled `certificate_type=none` because it is
post-processing, not a new certificate, and it includes an approximate 95%
interval for the raw-return gap from the reported standard errors.  In the
current run, CRAC leads the best SB3 row in mean raw return on HalfCheetah,
Hopper, and Walker2d.  The gap intervals still cross zero in all three
environments, while the certified clipped-improvement margins remain positive.

`ev_autonomous_validation.py` extends the public `highway-env` validation with
an electric-vehicle evaluation layer.  It does not claim that `highway-env`
contains full EV drivetrain physics.  Instead, it uses the public autonomous
driving task dynamics and records EV decision-level metrics:

- episode energy use and regenerative braking energy;
- final and minimum state of charge (SOC);
- battery reserve violations;
- TTC-like proximity proxy;
- EV-feasible success, requiring task success and no battery violation.

Certificate semantics are unchanged: Lex-BAC ODD rows keep `odd_exact` only for
the certified policy-selection layer, while EV energy/SOC metrics are validation
measurements rather than hard Bellman certificates.

The same script also writes an EV stress matrix over initial SOC and runtime
risk calibration.  This explicitly probes low-battery and risk-estimation
regimes rather than reporting only a single nominal EV setting.

### 7. Algorithmic Novelty Beyond Parametric LP

Classical parametric LP / multiparametric LP focuses on enumerating exact
critical regions and adjacent bases.  The following mechanisms use Bellman
certificates to answer RL-specific certified-control questions without treating
full atlas enumeration as the only objective.

**A0R. Certificate Residual Flow.** A0G repairs one state-action at a time.
A0R is the continuous policy-generation route.  For a stochastic policy
\(\pi(\cdot|s)\), compute the exact Bellman advantage residual

\[
A_\pi(s,a)=Q_\pi(s,a)-V_\pi(s).
\]

Then evolve the policy on the probability simplex by the replicator-type
certificate flow

\[
\dot\pi(a|s)=\pi(a|s)A_\pi(s,a).
\]

This is not a policy bank and not an action-replacement rule: every action
probability moves continuously according to its certified Bellman residual.

Theorem A0R (Monotone Residual Flow).  Let

\[
J_\rho(\pi)=\rho^\top V^\pi
\]

for a start distribution \(\rho\).  Along the continuous flow,

\[
\frac{d}{dt}J_\rho(\pi_t)
=
\frac{1}{1-\gamma}
\sum_s d^\pi_\rho(s)
\sum_a \pi(a|s)A_\pi(s,a)^2
\ge 0.
\]

Thus the Bellman-residual flow monotonically improves the policy objective and
is stationary only when every positive-probability action has zero advantage.
If the maximum positive advantage is zero, the Bellman optimality inequalities
hold and the exact verifier accepts.

Convergence statement.  On a finite MDP, the stochastic-policy simplex is
compact and \(J_\rho\) is bounded above.  The display above is therefore a
Lyapunov identity: \(J_\rho(\pi_t)\) converges, and every omega-limit point of
the interior continuous flow lies in the invariant set where

\[
\sum_s d^\pi_\rho(s)\sum_a \pi(a|s)A_\pi(s,a)^2=0.
\]

When the start distribution reaches every state and the limiting policy has
full support on every action, this invariant set is exactly the Bellman
optimality certificate set \(A_\pi(s,a)=0\) for all \((s,a)\).  More generally,
the implementation uses the stronger termination test
\(\max_{s,a} A_\pi(s,a)\le\tau\), so a converged discrete run carries an exact
\(\tau\)-optimality residual certificate rather than relying on support-only
stationarity.

Discrete certified implementation.  `certificate_residual_flow.py` uses exact
rational policy evaluation and an Euler step with backtracking.  A step is
accepted only if:

- the policy remains in the probability simplex;
- \(J_\rho\) does not decrease;
- every supplied discounted safety constraint remains below its limit.

Therefore every emitted step carries an exact certificate of monotone
improvement and safety preservation.  If no safe monotone step above the
minimum step size exists, the run stops rather than claiming progress.

Safety preservation.  For safety cost \(c_i\) and limit \(G_i\), define
\(J^c_i(\pi)=\rho^\top V^{\pi,c_i}\).  The implementation checks

\[
J^c_i(\pi_{k+1})\le G_i
\]

exactly before accepting the step, so all accepted iterates remain safe
whenever the initial policy is safe.

Originality boundary.  Replicator dynamics and policy-gradient identities are
known.  The new algorithmic object here is the proof-carrying residual flow:
the Bellman certificate residual is the vector field, and the line search
turns each continuous actor update into an independently checkable monotone
and safety-preserving certificate.  This is closer to a classical online RL
update than the policy-library branch below.

**A0G. Certificate-Guided Policy Generator.** The policy-library mechanisms
below are not the only route.  `certified_policy_generator.py` implements a
single-current-policy generator closer to classical policy iteration and actor
improvement.  It does not pre-store a set of policies and does not enumerate
all deterministic policies.  It maintains one current policy \(\pi_k\), calls
the independent Bellman verifier, and if verification fails, uses the most
negative Bellman slack as an exact counterexample:

\[
(s_k,a_k)
\in
\arg\min_{s,a}
\left[
V^{\pi_k}(s)-r(s,a)-\gamma P_aV^{\pi_k}
\right].
\]

If the minimum slack is negative, the generator repairs only that state:

\[
\pi_{k+1}(s_k)=a_k,\qquad
\pi_{k+1}(s)=\pi_k(s)\ \text{for }s\ne s_k.
\]

Theorem A0G.  For a finite discounted MDP, every failed certificate emitted by
`generate_certified_policy` is an exact Bellman counterexample to the current
policy.  The repair step is a strict policy-improvement step.  Therefore the
algorithm cannot revisit a previous policy and terminates after finitely many
repairs with an independently verified exact optimal deterministic policy.

Proof sketch.  A negative slack means
\(r(s_k,a_k)+\gamma P_{a_k}V^{\pi_k}>V^{\pi_k}(s_k)\).  The policy improvement
theorem for discounted MDPs implies \(V^{\pi_{k+1}}\ge V^{\pi_k}\), with strict
improvement in at least one state reachable from the repaired state.  There are
finitely many deterministic policies, so no strictly improving sequence can
cycle.  When no negative slack remains, the Bellman optimality inequalities
hold and the independent verifier accepts.

Actor interface.  A score actor can seed the current policy by choosing the
highest-scoring action in each state.  The certificate loop then corrects the
actor's output online.  This is still model-based exact certification, not a
neural-network hard certificate.

Originality boundary.  This mechanism deliberately borrows the improvement
logic of policy iteration.  The new contribution is the proof-carrying control
loop: every update is triggered by a concrete Bellman certificate failure and
the trace contains either exact counterexamples or a final exact optimality
certificate.  It is not a precomputed policy bank.

**A1. Certificate-Guided Pruning.** If the user needs only the target endpoint
policy \(\pi^*_1\) and its robustness interval, `innovations.py` first verifies
\(\pi^*_1\) at \(\lambda=1\), then builds only that policy's slack certificate
and scans its own roots from right to left.

Theorem A1. Let \(\pi\) be independently verified optimal at \(M_1\).  Let
\(R_\pi\) be the sorted roots in \([0,1]\) of only
\(\{p^\pi_{s,a}:a\ne\pi(s)\}\).  Scanning intervals induced by \(R_\pi\) from
right to left returns the maximal interval \([a,1]\) on which \(\pi\) is
globally optimal.  Its cost depends on \(|R_\pi|\le(m-n)(n+1)\), not on the
global atlas region count \(K\).

Proof sketch.  For fixed \(\pi\), all signs of all slack numerators are constant
between consecutive roots in \(R_\pi\).  The Bellman slack optimality criterion
is necessary and sufficient for \(\pi\)'s optimality.  Starting from
\(\lambda=1\), the first interval where a slack sign becomes negative is exactly
the left boundary of \(\pi\)'s robust interval.

Why this is not natural parametric LP enumeration: multiparametric LP normally
asks for all critical regions/bases.  A1 uses the endpoint Bellman certificate
as a query-specific target and never constructs unrelated regions.

**A2. Epsilon Policy Cover with Verifiable Guarantee.** Exact atlases can be
exponential because exact policy-region identities can change combinatorially.
The corrected positive result is therefore not a polynomial bound on exact
chart count and not a value-difference merge theorem.  It is an epsilon-cover
by few policies: construct a set \(\Pi_\epsilon\) such that every
\(\lambda\in[0,1]\) has some \(\pi\in\Pi_\epsilon\) satisfying

\[
\|V^*_\lambda - V^\pi_\lambda\|_\infty\le \epsilon.
\]

Let

\[
R_{\max}=\max_{s,a,i}|r^i(s,a)|,\quad
D_r=\max_{s,a}|r^1-r^0|,\quad
D_P=\max_{s,a}\|P^1_{s,a}-P^0_{s,a}\|_1,
\]

and

\[
L=\gamma R_{\max}D_P+D_r.
\]

Lemma A2.1.  The optimal value path is Lipschitz:

\[
\|V^*_{\lambda_1}-V^*_{\lambda_2}\|_\infty
\le
\frac{L}{(1-\gamma)^2}|\lambda_1-\lambda_2|.
\]

Proof sketch.  \(T_\lambda\) is a \(\gamma\)-contraction and, on the value ball
\(\|V\|_\infty\le R_{\max}/(1-\gamma)\), its dependence on \(\lambda\) is at
most \((D_r+\gamma D_P R_{\max}/(1-\gamma))|\lambda_1-\lambda_2|\).  The
contraction fixed-point perturbation lemma gives the extra factor
\(1/(1-\gamma)\), yielding the displayed \(L/(1-\gamma)^2\) bound.

Lemma A2.2.  For any fixed deterministic policy \(\pi\),

\[
\|V^\pi_{\lambda_1}-V^\pi_{\lambda_2}\|_\infty
\le
\frac{L}{(1-\gamma)^2}|\lambda_1-\lambda_2|.
\]

The proof is identical because \(T^\pi_\lambda\) is also a
\(\gamma\)-contraction.

Lemma A2.3.  If \(\pi\) is \(\epsilon/2\)-optimal at \(\lambda_t\), then it is
\(\epsilon\)-optimal for all

\[
|\lambda-\lambda_t|
\le
\frac{(1-\gamma)^2\epsilon}{4L}.
\]

This follows by applying Lemma A2.1 and A2.2 to the optimal and fixed-policy
values, so the loss can grow by at most
\(2L|\lambda-\lambda_t|/(1-\gamma)^2\).

Theorem A2.  Let

\[
h=\frac{(1-\gamma)^2\epsilon}{2L},\qquad
T=\lceil 1/h\rceil.
\]

At grid points \(\lambda_t=t/T\), compute an exact optimal deterministic policy
\(\pi_t\).  Then

\[
\Pi_\epsilon=\{\pi_0,\ldots,\pi_T\}
\]

is an epsilon-cover of \([0,1]\), and

\[
|\Pi_\epsilon|\le T+1
=O\!\left(\frac{L}{(1-\gamma)^2\epsilon}\right).
\]

The bound has no \(n\), no \(A^n\), and no exact-region count \(K\) term except
through the model Lipschitz magnitude \(L\).  This is how A2 is compatible with
IndependentFlip-style exact atlas lower bounds: exact regions can be
exponential while the number of policies needed for value-\(\epsilon\) coverage
is controlled by \(L\), \(\epsilon\), and \(1/(1-\gamma)\).
The dependence is explicitly \(L/(1-\gamma)^2\), not \(L/(1-\gamma)\).

Each generated chart carries `certificate_type='eps_exact'`.  The relaxed
checker `verify_eps_chart` uses exact arithmetic at the anchor point and the
Lipschitz radius above.  It is an epsilon certificate, not a hard optimality
certificate and not a sampled high-probability certificate.

Why this is not natural parametric LP enumeration: parametric LP typically
preserves exact optimal bases/regions.  A2 deliberately discards exact
region identity and certifies only performance coverage by a small policy set.

**A2W. Witness-BAC Counterexample-Guided Policy Cover.** This is algorithmic
novelty, not deployment-system novelty.  A2 uses a uniform Lipschitz grid.  The
Witness-BAC algorithm instead maintains a certified policy set
\(\Pi_t\), finds a witness \(\lambda\) in the largest currently uncovered
interval, computes an exact optimal policy \(\pi_\lambda\), and adds the
centered Bellman-Lipschitz chart certified around that witness.  It stops only
when the union of certified charts covers \([0,1]\).

Let

\[
C=\frac{L}{(1-\gamma)^2}.
\]

If \(\pi_t\) is exactly optimal at a witness \(\lambda_t\), then for every
\(\lambda\) satisfying

\[
|\lambda-\lambda_t|\le \rho,\qquad \rho=\frac{\epsilon}{2C},
\]

\[
\|V^*_\lambda-V^{\pi_t}_\lambda\|_\infty\le\epsilon.
\]

Theorem A2W.  `build_witness_policy_cover` returns a finite policy set
\(\Pi_W\) and charts whose union covers \([0,1]\).  Every returned chart has
`certificate_type='eps_exact'`, is independently checked by exact anchor
Bellman optimality plus the radius inequality above, and therefore
\(\Pi_W\) is an \(\epsilon\)-policy cover.  The loop terminates within the
same worst-case order as A2,

\[
O\!\left(\frac{L}{(1-\gamma)^2\epsilon}\right)
\]

witness insertions, while adapting to long stable stretches and duplicate
optimal policies in practice.

Proof sketch.  Each witness chart is sound by Lemma A2.1 and Lemma A2.2:
the optimal value and fixed-policy value can separate by at most
\(2C|\lambda-\lambda_t|\).  The loop adds a chart centered in an uncovered gap;
the uniform A2 grid is a finite cover at the same length scale, so the greedy
witness process cannot require more than a constant-factor number of certified
intervals before no uncovered gap remains.  Once no gap remains, every
\(\lambda\) lies inside at least one sound witness chart.

Originality boundary.  Counterexample-guided synthesis and adaptive sampling
are known general paradigms.  The claim here is narrower: Witness-BAC uses
Bellman verification failure as a policy-cover witness, injects the exact
optimal deterministic policy at that witness, and emits independently
verifiable \(\epsilon\)-optimality certificates.  It is not exact
parametric-LP region enumeration, because it never traverses adjacent bases or
critical-region geometry, and it is not a value-merge theorem because every
chart still names a single certified policy.

**A2WB. Bernstein-expanded Witness-BAC.** A2W still uses a conservative global
Lipschitz radius.  A2B is a proof checker.  The constructive algorithm in
`build_bernstein_expanded_witness_cover` combines them: after a witness
\(\lambda_t\) selects its exact optimal policy \(\pi_t\), the algorithm tries
to expand the chart left and right using root-free Bernstein-Bellman interval
certificates.  The Lipschitz epsilon chart is kept only as fallback.

Theorem A2WB.  Every returned `bernstein_exact` chart is an exact optimality
certificate by Theorem A2B.  Every returned `lipschitz_eps` chart is an
\(\epsilon\)-optimality certificate by Theorem A2W.  Therefore the union of the
returned charts is an \(\epsilon\)-policy cover of \([0,1]\), and the algorithm
inherits the A2W termination guarantee

\[
O\!\left(\frac{L}{(1-\gamma)^2\epsilon}\right)
\]

because the fallback chart is always available at each witness.  In favorable
cases, Bernstein expansion recovers long exact-optimality intervals and uses
strictly fewer charts than the fixed-radius witness cover.

Proof sketch.  Soundness is by disjunction of two independently verified
certificate types: exact Bernstein positivity of all Bellman slack numerators,
or endpoint optimality plus the A2 Lipschitz radius.  The largest-uncovered-gap
witness loop is unchanged from A2W, so a witness always contributes at least
the Lipschitz fallback interval even when Bernstein expansion rejects.

Why this is not natural parametric LP enumeration: the algorithm never solves
critical boundary equations and does not follow adjacent bases.  It turns a
policy witness into a root-free interval certificate and uses the certificate
as the expansion oracle.

**A2D. Certificate-Maximal Witness Selection.** Degenerate witness points can
have several exact optimal deterministic policies.  A lexicographic tie-break
is mathematically valid but can choose a policy whose certified interval is
short.  `select_certificate_maximal_policy` enumerates all policies accepted by
the independent verifier at the witness, expands each candidate with A2WB, and
selects the one with maximum exact Bernstein-certified interval width.

Theorem A2D.  For a fixed witness \(\lambda_t\), finite discounted MDP, and
fixed Bernstein expansion parameters, `select_certificate_maximal_policy`
returns an anchor-optimal policy whose exact certified interval width is
maximal among all deterministic policies optimal at \(\lambda_t\).  The
returned policy remains exact optimal at the witness and its chart certificates
remain sound by A2B/A2WB.

Proof sketch.  The set of deterministic policies is finite.  The independent
Bellman verifier exactly filters the subset optimal at \(\lambda_t\).  The
Bernstein expansion oracle computes a deterministic certified interval for
each surviving policy.  Taking the maximum over this finite list is therefore
an exact optimum for the stated certificate-width objective.

Originality boundary.  A2D is not a claim about globally optimal policy-search
under arbitrary objectives.  It is a certificate-aware tie selector for
homotopy witnesses, designed to avoid arbitrary lexicographic choices at
degenerate Bellman optima.

**A2C. Certificate-Preserving Cover Compression.** A2W/A2WB generate
certificate-carrying charts.  Some charts can be redundant.  The module
`certificate_compression.py` selects a minimum-cardinality subset of already
certified intervals that still covers the requested \(\lambda\)-domain.

Theorem A2C.  For a finite family of certified intervals
\(\mathcal I=\{[l_i,r_i]\}\) covering \([0,1]\), the greedy algorithm that,
at the current uncovered point \(x\), chooses an interval with \(l_i\le x\)
and maximum \(r_i\), returns a cover with the minimum possible number of
intervals.  Because it only discards charts, every remaining chart keeps its
original `certificate_type` and certificate proof.

Executable minimality certificate.  The verifier `verify_cover_minimality`
computes a greedy frontier sequence \(x_0,x_1,\ldots\), where \(x_t\) is the
farthest point reachable by any \(t\)-chart certified cover.  If the selected
cover has \(k\) charts, covers \([0,1]\), and \(x_{k-1}<1\), then no cover with
fewer than \(k\) charts exists.  This turns the minimum-cardinality claim into
a checkable lower-bound witness, not just a prose proof.

Proof sketch.  Let \(I_g\) be the greedy first interval and \(I_o\) the first
interval of an optimal cover.  Since both must start at or before 0 and
\(I_g\) reaches at least as far as \(I_o\), replacing \(I_o\) by \(I_g\) keeps
the remaining target suffix no harder to cover.  Induct on the uncovered
suffix.  This exchange argument gives exact minimum-cardinality optimality.

Originality boundary.  This is not general policy-space compression, and it
does not solve arbitrary set cover over policies.  The result is deliberately
narrow: once Bellman certificates have reduced the RL problem to certified
one-dimensional intervals, the chart library can be compressed optimally
without weakening any mathematical guarantee.

**A2E. Switch-Aware Certified Cover.** Minimum chart count is not always the
right runtime objective.  A drone or vehicle may prefer one extra certified
chart if it avoids a large policy jump.  `switch_cover.py` treats certified
charts as intervals labeled by deterministic policies and minimizes

\[
\left(
\sum_t d_H(\pi_t,\pi_{t+1}),\quad \#\text{charts}
\right)
\]

lexicographically, where \(d_H\) is Hamming distance between action choices.

Theorem A2E.  Build a directed graph whose nodes are certified charts.  Add a
source edge to every chart starting before the target left endpoint.  Add an
edge \(i\to j\) when chart \(j\) starts before the current covered frontier
and extends it.  Give edge \(i\to j\) cost
\((d_H(\pi_i,\pi_j),1)\) and source edges cost \((0,1)\).  A shortest path to
any chart reaching the target right endpoint is exactly a certified cover with
minimum total Hamming switch cost, and among those, minimum chart count.

Proof sketch.  Every valid ordered cover induces such a path, because each next
chart must overlap the current frontier and extend it.  Every such path induces
a continuous certified cover.  Edge costs add exactly the objective terms, so
Dijkstra's algorithm over nonnegative lexicographic costs returns the global
optimum.

Originality boundary.  Shortest paths are classical.  The new algorithmic
object is the certificate graph produced by Bellman/Bernstein witness charts,
where runtime smoothness is optimized without weakening exact or epsilon
certificates.  This is algorithmic control selection, not deployment-system
novelty.

**A2F. Runtime Hysteresis Certified Selector.** Offline smooth covers reduce
planned policy jumps, but runtime \(\lambda\)-estimates can jitter.  The
selector `select_runtime_hysteresis_chart` receives an uncertainty interval
\([\hat\lambda-\eta,\hat\lambda+\eta]\), a current chart index, and a library
of certified charts.

Theorem A2F.  If the current chart still covers the whole uncertainty interval,
the selector keeps it and incurs zero switch cost.  Otherwise, among all charts
covering the uncertainty interval, it selects one with minimum Hamming switch
cost from the current policy, breaking ties by largest certificate margin.  If
no chart covers the uncertainty interval, it returns `certificate_type='none'`
and refuses to claim a certified action.

Proof sketch.  Coverage is checked by interval inclusion
\(l_i\le\hat\lambda-\eta\le\hat\lambda+\eta\le r_i\).  If the current chart
passes this check, retaining it is trivially feasible and has minimum possible
switch cost \(0\).  If not, the feasible chart set is finite, and the selector
explicitly minimizes Hamming switch cost over that finite set.

Originality boundary.  Hysteresis and runtime assurance are known concepts.
A2F's narrow contribution is embedding hysteresis into Bellman/Bernstein
certificate-chart selection, so anti-chattering behavior never invents an
uncertified policy.

**A2G. Risk-Weighted Certified Cover.** Uniform \(\epsilon\)-covers treat every
homotopy parameter as equally important.  Safety-critical control often does
not: \(\lambda\) may encode road density, obstacle proximity, battery reserve,
or weather severity.  A2G changes the algorithmic objective itself.  Given a
piecewise-constant risk profile \(w(\lambda)>0\), construct charts that
guarantee

\[
w(\lambda)\|V^*_\lambda-V^\pi_\lambda\|_\infty\le\epsilon.
\]

Equivalently, high-risk regions receive a smaller local regret budget
\(\epsilon/w(\lambda)\) and therefore a denser certified policy cover.

Let \(C=L/(1-\gamma)^2\).  On a risk interval
\([a_j,b_j]\) with constant weight \(w_j\), split the interval into

\[
N_j=\left\lceil \frac{2Cw_j(b_j-a_j)}{\epsilon}\right\rceil
\]

subintervals.  At the left endpoint of each subinterval compute an exact
optimal deterministic policy and attach `certificate_type='eps_exact'`.

Theorem A2G.  For every returned chart \([l,r]\) inside risk interval \(j\),
the anchor policy \(\pi_l\) satisfies

\[
\forall \lambda\in[l,r],\qquad
w_j\|V^*_\lambda-V^{\pi_l}_\lambda\|_\infty\le\epsilon.
\]

The chart count obeys

\[
\sum_j N_j
\le
J+\frac{2C}{\epsilon}\sum_j w_j(b_j-a_j),
\]

where \(J\) is the number of risk intervals.  The bound depends on weighted
risk measure and horizon through \(C=L/(1-\gamma)^2\), not on exact atlas
region count, not on \(A^n\), and not on state count except through the model
Lipschitz magnitude \(L\).

Proof sketch.  At anchor \(l\), \(\pi_l\) is exactly optimal.  Lemma A2.1 and
Lemma A2.2 imply

\[
\|V^*_\lambda-V^{\pi_l}_\lambda\|_\infty
\le 2C|\lambda-l|.
\]

The interval width is at most \(\epsilon/(2Cw_j)\), hence multiplying by
\(w_j\) gives the stated weighted regret certificate.

Why this is not natural parametric LP enumeration: parametric LP critical
regions are basis identities and do not encode a weighted regret metric over
ODD risk.  A2G solves a different synthesis problem: allocate certified policy
resolution according to safety weight while retaining Bellman-verifiable
performance bounds.  This is also not SAC, PPO, or TD3; it is not a stochastic
actor-critic update rule, but a certificate-carrying policy synthesis method
whose stability comes from exact Bellman inequalities and Lipschitz geometry.

**A2H. Uncertainty-Minimax Certified Selector.** A2F is strict: it requires
one certified chart to contain the whole runtime uncertainty
interval \([\hat\lambda-\eta,\hat\lambda+\eta]\).  That can reject even when
a useful conservative policy choice is still certifiable.  A2H solves a
different runtime problem: among all anchor certificates in a risk-weighted
cover, choose the policy minimizing a certified worst-case weighted regret
upper bound over the whole uncertainty interval.

For a chart with anchor \(a\), policy \(\pi_a\), and runtime interval \(U\),
define

\[
B(a,U)=
\max_j\ 
2Cw_j\max_{\lambda\in U\cap[a_j,b_j]}|\lambda-a|,
\qquad C=\frac{L}{(1-\gamma)^2}.
\]

`select_uncertainty_minimax_policy` returns the chart minimizing \(B(a,U)\),
breaking ties by Hamming switch cost from the current policy.  It marks the
selection accepted only if \(B(a,U)\le\epsilon\); otherwise it still returns
the best exact upper bound with `certificate_type='eps_exact'` and reason
`best certified minimax bound exceeds budget`.

Theorem A2H.  For the selected policy \(\pi_a\),

\[
\sup_{\lambda\in U}
w(\lambda)\|V^*_\lambda-V^{\pi_a}_\lambda\|_\infty
\le B(a,U),
\]

and no policy anchor in the supplied certificate library has a smaller such
computed upper bound.  If \(B(a,U)\le\epsilon\), the action selected for the
uncertain ODD interval is certified \(\epsilon\)-admissible in the weighted
regret metric.

Proof sketch.  Anchor optimality gives zero loss at \(a\).  Lemma A2.1 and
Lemma A2.2 give
\(\|V^*_\lambda-V^{\pi_a}_\lambda\|_\infty\le2C|\lambda-a|\).
Multiplying by the risk weight on each risk interval and taking the maximum
over \(U\) yields \(B(a,U)\).  The library is finite, so explicit enumeration
and lexicographic tie-breaking gives the minimum certified bound over that
library.

Why this is not natural parametric LP enumeration: A2H does not ask which exact
critical region contains \(\hat\lambda\).  It optimizes a robust certificate
functional over a noisy ODD interval and can still produce an exact bound when
no single chart contains the whole interval.  This is algorithmic selection
under certified regret, not deployment plumbing.

**A2I. Budget-Optimal Risk Allocation.** A2G answers: for a requested
\(\epsilon\), how many certified charts are sufficient?  A vehicle or drone may
face the inverse constraint: only \(B\) policies can be stored, audited, or
switched among at runtime.  A2I solves that inverse problem inside the
certificate family itself.

For risk interval \(j\), let

\[
a_j=w_j(b_j-a_j).
\]

Allocate \(n_j\ge1\) centered charts to interval \(j\), with
\(\sum_j n_j\le B\).  The anchor of each chart is the midpoint of its cell, so
the farthest point in the cell is \((b_j-a_j)/(2n_j)\).  By Lemma A2.1 and
Lemma A2.2, the worst weighted regret certificate on interval \(j\) is

\[
2Cw_j\frac{b_j-a_j}{2n_j}
=
C\frac{a_j}{n_j}.
\]

The allocation problem is therefore the exact integer minimax program

\[
\min_{n_j\in\mathbb N,\ \sum_j n_j\le B}
\max_j \frac{a_j}{n_j}.
\]

Theorem A2I.  `allocate_minimax_risk_budget` returns an allocation
\((n_1,\ldots,n_J)\) attaining the optimum of the program above.  The cover
`build_budget_optimal_risk_weighted_cover` then returns \(B\) centered
anchor-policy certificates whose worst weighted regret upper bound is

\[
\epsilon_B^*
=
C\min_{\sum_j n_j\le B}\max_j\frac{a_j}{n_j}.
\]

Proof sketch.  For any threshold \(\tau>0\), interval \(j\) needs exactly
\(\lceil a_j/\tau\rceil\) charts to satisfy \(a_j/n_j\le\tau\).  Thus
\(\tau\) is feasible iff

\[
\sum_j \left\lceil a_j/\tau\right\rceil\le B.
\]

The optimum threshold must equal \(a_j/q\) for some interval \(j\) and integer
\(1\le q\le B\).  The algorithm enumerates these rational candidates in
increasing order and picks the first feasible one, then spends any leftover
charts without increasing the objective.  This is an exact discrete
water-filling certificate.

Originality boundary.  Resource allocation and water-filling are classical.
The new object is the Bellman-certificate budget itself: A2I optimizes how a
finite onboard policy library should be distributed across a risk-weighted ODD
so that the certified regret bound is minimax optimal.  It is not a training
heuristic, and it is not parametric-LP region enumeration.

**A2J. Minimax Anchor Library Design.** A2I gives the optimal number of
certified charts per risk interval.  A2J makes the remaining geometric step
explicit: where should the policy anchors be placed?  For each interval
\([a_j,b_j]\) with \(n_j\) assigned anchors, split it into equal cells and place
each anchor at the cell midpoint.

Theorem A2J.  For fixed \(n_j\), the midpoint equal-cell design minimizes

\[
\sup_{\lambda\in[a_j,b_j]}w_j\min_t|\lambda-\alpha_t|
\]

over all \(n_j\) anchors \(\alpha_t\in[a_j,b_j]\), with optimum value

\[
\frac{w_j(b_j-a_j)}{2n_j}.
\]

Combining A2J with the A2I allocation yields a full library design minimizing
the certified upper bound

\[
\sup_\lambda w(\lambda)\|V^*_\lambda-V^{\pi(\lambda)}_\lambda\|_\infty
\]

within the Bellman-Lipschitz anchor-certificate family and chart budget \(B\).
The implementation emits a `RiskLibraryDesign` object and an independent
`verify_minimax_risk_library_design` checker.

Proof sketch.  With \(n_j\) anchors, the \(n_j\) Voronoi cells cover an interval
of length \(b_j-a_j\), so at least one cell has length at least
\((b_j-a_j)/n_j\).  Any point-anchor representative of that cell has maximum
distance at least half its length, yielding the lower bound
\((b_j-a_j)/(2n_j)\).  Equal cells with midpoint anchors attain this bound.
Multiplying by \(w_j\) gives the weighted result.

Originality boundary.  One-dimensional k-center is classical.  The new role is
as an exact proof-carrying design layer for a finite Bellman policy library:
the algorithm returns not only policies, but a verifiable certificate that
their anchor locations are minimax optimal for the risk-weighted regret
geometry.  This is still algorithmic synthesis, not deployment infrastructure.

**A2K. Vector-Risk Pareto Envelope.** A vehicle or drone rarely has only one
risk notion.  Collision risk, battery reserve, speed comfort, and mission time
can disagree.  A2K lifts A2I/A2J from one risk profile to several aligned risk
profiles

\[
w^{(q)}(\lambda),\qquad q=1,\ldots,Q,
\]

defined on the same ODD partition.  For an allocation
\(n=(n_1,\ldots,n_J)\), define the certified vector bound

\[
b_q(n)
=
C\max_j
\frac{w^{(q)}_j(b_j-a_j)}{n_j},
\qquad C=\frac{L}{(1-\gamma)^2}.
\]

The factor \(C\) appears because A2J's midpoint design has weighted radius
\(w^{(q)}_j(b_j-a_j)/(2n_j)\) and Bellman-Lipschitz regret is bounded by
\(2C\) times distance.

Theorem A2K.  For fixed chart budget \(B\) and shared partition, enumerate all
positive allocations \(\sum_j n_j=B\).  `compute_vector_risk_pareto_envelope`
returns exactly the non-dominated allocations under componentwise comparison of
the vector \(b(n)\), and for every dominated allocation it records a concrete
dominating allocation.  `verify_vector_risk_pareto_envelope` independently
recomputes the candidate set, Pareto front, and dominance certificates.

Proof sketch.  The feasible allocation set is finite.  For each allocation,
A2J gives the exact minimax anchor certificate for every risk component, hence
the displayed vector \(b(n)\) is exact inside the centered Bellman-Lipschitz
library family.  Componentwise dominance over a finite set is decidable by
direct pairwise comparison; therefore the returned front is exact iff no
candidate outside the front dominates a front point and every omitted candidate
has a listed dominator.

Originality boundary.  Multiobjective RL and Pareto fronts are broad existing
areas.  The narrow contribution here is a proof-carrying Pareto envelope for
finite Bellman policy libraries under ODD risk profiles: every plotted point is
an exact certificate vector, and every discarded design has an explicit
dominance witness.  This is not SAC, PPO, or TD3 with multiple rewards; it is
offline certified policy-library synthesis.

**A2L. Distributionally Robust Risk Envelope.** A2G-K assume the ODD risk
weights are known.  In autonomous driving or drone navigation those weights are
often calibrated from perception, weather, battery aging, or traffic estimates.
A2L treats each interval weight as uncertain:

\[
w_j\in[\underline w_j,\overline w_j].
\]

For any fixed anchor library, the certified regret bound is monotone in every
weight.  Therefore the worst case over this box uncertainty set is attained at
the upper envelope \(\overline w_j\).

Theorem A2L.  Let \(\mathcal W\) be the interval uncertainty set above.  The
distributionally robust certified design problem

\[
\min_{\text{library } \mathcal L}
\sup_{w\in\mathcal W}
\sup_\lambda
w(\lambda)\|V^*_\lambda-V^{\pi_\mathcal L(\lambda)}_\lambda\|_\infty
\]

inside the A2J centered Bellman-Lipschitz library family is solved by applying
A2I/A2J to the upper-envelope risk profile
\(\overline w=(\overline w_1,\ldots,\overline w_J)\).  The resulting bound is

\[
2C\cdot
\min_{\sum_j n_j\le B}
\max_j
\frac{\overline w_j(b_j-a_j)}{2n_j}.
\]

`compute_distributionally_robust_risk_envelope` constructs this certificate,
and `verify_distributionally_robust_risk_envelope` checks the uncertainty set,
the upper-envelope design, and the bound equality.

Proof sketch.  For any fixed allocation and anchors, A2J bounds the weighted
radius by \(w_j r_j\) on interval \(j\).  This expression is coordinatewise
monotone in \(w_j\), so the supremum over the uncertainty box occurs at
\(\overline w_j\).  Once reduced to the upper-envelope profile, A2I/A2J give
the exact minimax allocation and anchor placement.

Originality boundary.  Robust optimization is classical.  The new role here is
to make ODD risk calibration uncertainty part of the Bellman certificate
itself: a design can now state exactly which risk-weight box it covers and
which worst-case regret bound is certified.  This remains a model-based
`eps_exact` certificate, not a sampled confidence claim.

**A2M. Robust Minimum Memory Sizing.** A2L answers: for a fixed chart budget,
what worst-case regret bound can be certified?  A2M solves the inverse problem:
given a required robust bound \(\epsilon\), what is the minimum number of
onboard policies/charts that must be stored and audited?

For the upper-envelope weighted lengths

\[
a_j=\overline w_j(b_j-a_j)
\]

and Bellman-Lipschitz constant \(C=L/(1-\gamma)^2\), the centered library bound
is \(C a_j/n_j\) on interval \(j\).  Therefore interval \(j\) needs

\[
n_j\ge \frac{C a_j}{\epsilon}.
\]

Theorem A2M.  The exact minimum robust chart budget for target bound
\(\epsilon\) is

\[
B^*(\epsilon)
=
\sum_j
\left\lceil
\frac{C\overline w_j(b_j-a_j)}{\epsilon}
\right\rceil.
\]

`compute_minimum_robust_memory_sizing` returns these counts and
`verify_minimum_robust_memory_sizing` recomputes the ceiling lower bound and
the achieved worst-case regret certificate.

Proof sketch.  Necessity follows from \(C a_j/n_j\le\epsilon\) for every
interval, which implies the ceiling lower bound on each \(n_j\).  Sufficiency
is achieved by choosing exactly those counts and applying the A2J midpoint
library design.  Summing the per-interval lower bounds gives the minimum total
budget.

Originality boundary.  This is a sizing theorem for the certified Bellman
library family, not a claim about arbitrary neural policy compression.  Its
practical value is that a vehicle or drone can convert a required certified
robust regret threshold directly into the smallest auditable policy-library
size before deployment.

**A2N. Anytime Robust Refinement Schedule.** A2M is static: it sizes a library
for one target bound.  A vehicle or drone may instead receive additional
storage, audit budget, or policy slots over time.  A2N constructs a nested
sequence of robust libraries

\[
\mathcal L_{J}\subset\mathcal L_{J+1}\subset\cdots\subset\mathcal L_B,
\]

where \(J\) is the number of ODD risk intervals and each step adds exactly one
chart.

At each step, the algorithm tries each possible single-interval refinement and
chooses one minimizing the next certified bound

\[
C\max_j\frac{\overline w_j(b_j-a_j)}{n_j}.
\]

Theorem A2N.  The returned sequence is nested, its certified robust regret
bounds are non-increasing, and every budget level in the sequence attains the
same optimum as A2I/A2M for that budget.  `verify_anytime_robust_refinement_schedule`
checks all three properties by recomputing the exact A2I optimum at every
budget.

Proof sketch.  Each step only increments one \(n_j\), so the libraries are
nested.  Increasing \(n_j\) cannot increase any term
\(\overline w_j(b_j-a_j)/n_j\), hence the max bound is monotone
non-increasing.  Per-budget optimality is certified independently by comparing
the current allocation against the exact threshold enumeration in A2I.

Originality boundary.  This is not a new generic greedy theorem for every
resource allocation problem.  It is an anytime certificate-maintaining
refinement protocol for the Bellman robust policy-library objective, designed
so an deployed controller can grow its audited library without invalidating
older certificates.

**A2B. Bernstein-Bellman Root-Free Interval Certificates.** This is also
algorithmic novelty, not deployment-system novelty.  Exact Lex-BAC can certify
an interval by finding the roots of every Bellman slack numerator.  A2B instead
certifies an interval without root solving.  For a claimed policy \(\pi\) on
\([a,b]\), build its exact slack certificate

\[
\Delta^\pi_{s,u}(\lambda)=p^\pi_{s,u}(\lambda)/q^\pi(\lambda).
\]

Write \(p^\pi_{s,u}((1-t)a+tb)\) in the Bernstein basis:

\[
p^\pi_{s,u}((1-t)a+tb)
=
\sum_{i=0}^d \beta_i B_i^d(t),
\qquad
B_i^d(t)=\binom di t^i(1-t)^{d-i}.
\]

Theorem A2B.  If every Bernstein coefficient \(\beta_i\) is nonnegative for
every state-action slack numerator, then \(\pi\) is globally optimal for every
\(\lambda\in[a,b]\).  The verifier uses exact symbolic arithmetic and returns
`certificate_type='exact'`.

Proof sketch.  Bernstein basis functions are nonnegative and sum to one on
\([0,1]\).  Thus nonnegative Bernstein coefficients imply
\(p^\pi_{s,u}(\lambda)\ge0\) throughout \([a,b]\).  The denominator
\(q^\pi(\lambda)=\det(I-\gamma P^\lambda_\pi)\) is positive for discounted
stochastic \(P^\lambda_\pi\), so all Bellman slacks are nonnegative on the
whole interval.  Bellman optimality inequalities are necessary and sufficient
for deterministic policy optimality in discounted finite MDPs.

The recursive variant splits intervals whose coefficients are not yet all
nonnegative.  It is sound for every accepted interval and conditionally
complete under a strict positive slack margin; boundary-touching degeneracy may
still need root-aware reasoning.  This limitation is deliberate and is tested:
bad intervals reject rather than silently becoming certificates.

Why this is not natural parametric LP enumeration: parametric LP typically
discovers exact adjacent critical regions by solving boundary equations.  A2B
is a root-free Bellman proof checker for a proposed policy interval; it can
verify charts generated by A1, A2, or Witness-BAC without traversing the global
critical-region graph.

**A3. Incremental Re-Certification Under Perturbation.** Given a certified
target policy \(\pi\), a changed target MDP can be re-certified by running the
independent verifier on the changed endpoint only; no atlas rebuild is needed.
For unknown bounded perturbation sets, a sufficient margin certificate is also
available.

Let

\[
g_\pi=\min_{s,a\ne\pi(s)}\Delta^\pi_{s,a}(1)
\]

be the target action gap.  If reward perturbations obey
\(\|\delta r\|_\infty\le \epsilon_r\) and transition perturbations obey
\(\|\delta P_{s,a}\|_1\le\epsilon_P\), then every non-policy slack changes by at
most

\[
L_{\text{env}}
=(1+\gamma)\frac{\epsilon_r+\gamma\epsilon_P R_{\max}/(1-\gamma)}{1-\gamma}
+\epsilon_r+\gamma\epsilon_P R_{\max}/(1-\gamma).
\]

Theorem A3. For a concrete perturbed endpoint, \(\pi\) remains optimal if and
only if the independent verifier accepts it on the perturbed model.  For an
unknown perturbation ball, \(g_\pi>L_{\text{env}}\) is a sufficient uniform
certificate that \(\pi\) remains optimal for every model in the ball.

The "if and only if" statement is restricted to concrete re-certification.  The
Lipschitz margin is a sufficient robust condition, not a
necessary one for every benign perturbation.

Why this is not natural parametric LP enumeration: the mechanism certifies a
previously deployed policy under local data drift, using Bellman slack margins
and affected-pair checks rather than re-enumerating a full parameter atlas.

**A4. Semantic ODD-Cover for Deployment.** The strongest deployment-oriented
extension is implemented in `semantic_odd_cover.py`.  It builds a finite policy
library for an operational design domain (ODD) parameter \(\lambda\), such as
wind/obstacle severity for drones or traffic/friction stress for driving.  Each
chart certifies three things at once:

- value loss at most \(\epsilon\);
- a semantic discounted safety cost below a user limit \(G\);
- robustness to an interval-valued runtime estimate
  \([\hat\lambda-\eta,\hat\lambda+\eta]\).

This differs from common safe-RL shields and control-barrier filters.  A
shield/CBF modifies actions online.  ODD-Cover instead
pre-computes certificate-carrying policies and refuses to claim safety when the
runtime ODD uncertainty interval is not contained in a certified chart.

Let \(C_V=L_r/(1-\gamma)^2\) be the A2 performance Lipschitz constant and let
\(C_g=L_g/(1-\gamma)^2\) be the same fixed-policy Lipschitz constant for a
semantic safety cost \(g^\lambda\).  For a chart anchored at \(\lambda_t\), let
\(\pi_t\) be exactly optimal at \(\lambda_t\) and suppose

\[
|\lambda-\lambda_t|\le\rho,\qquad
2C_V\rho\le\epsilon,\qquad
J_g^{\pi_t}(\lambda_t)+C_g\rho\le G.
\]

Theorem A4.  For every \(\lambda\) in that chart,

\[
\|V^*_\lambda-V^{\pi_t}_\lambda\|_\infty\le\epsilon,
\qquad
J_g^{\pi_t}(\lambda)\le G.
\]

If a runtime estimate interval
\([\hat\lambda-\eta,\hat\lambda+\eta]\) is contained in the chart, then the same
two guarantees hold for every true \(\lambda\) in that interval.  The verifier
`verify_odd_chart` checks the anchor optimality, exact anchor safety cost, and
the two Lipschitz-radius inequalities using exact arithmetic.  Its
`certificate_type` is `odd_exact`.

Originality boundary.  The project does not claim that safety filters, CBFs, or
runtime assurance are new.  The new object here is the combination of
Bellman-atlas epsilon policy cover, semantic discounted safety costs, exact
chart verification, and interval-valued ODD runtime selection without exact
atlas enumeration.

## Running Tests

```bash
python -m pytest -q
```

## Running Experiments

```bash
python experiments.py --out-dir outputs --n-min 3 --n-max 12 --actions 3
python pc_experiments.py
python innovation_experiments.py
python model_free_benchmarks.py
python odd_experiments.py
python highway_env_validation.py
python ev_autonomous_validation.py
```

The CSV marks exact solver runs that are skipped because the requested
`max_policies` limit was exceeded.  This is intentional: exact atlas generation
is exponential, while final-policy verification remains polynomial.

`pc_experiments.py` writes:

- `outputs/pc_bac_comparison.csv` and `.png`;
- `outputs/long_horizon_iterations.csv` and `.png`.
- `outputs/model_free_demo.csv`.

The long-horizon CSV includes an explicit `sc_ipr_converged` column.  If a
Newton run reaches the iteration cap, the row is not treated as a convergence
claim.

The model-free demo includes `hard_certificate_claimed=False`.  Its slack values
are pessimistic sampled lower bounds, not exact Bellman certificates.

`innovation_experiments.py` writes:

- `outputs/a0r_certificate_residual_flow.csv` and `.png`;
- `outputs/a0g_certified_policy_generator.csv` and `.png`;
- `outputs/a1_pruning_vs_global.csv` and `.png`;
- `outputs/a2_epsilon_policy_cover.csv` and `.png`;
- `outputs/a2_independentflip_exact_vs_cover.csv` and `.png`;
- `outputs/a2_cover_vs_gamma.csv` and `.png`;
- `outputs/a2_cover_vs_n.csv` and `.png`;
- `outputs/a2b_bernstein_interval_certificates.csv` and `.png`;
- `outputs/a2d_certificate_maximal_selector.csv` and `.png`;
- `outputs/a2e_switch_aware_cover.csv` and `.png`;
- `outputs/a2f_hysteresis_runtime_selector.csv` and `.png`;
- `outputs/a2g_risk_weighted_cover.csv` and `.png`;
- `outputs/a2h_uncertainty_minimax_selector.csv` and `.png`;
- `outputs/a2i_budget_optimal_risk_allocation.csv` and `.png`;
- `outputs/a2j_minimax_anchor_library_design.csv` and `.png`;
- `outputs/a2k_vector_risk_pareto_envelope.csv` and `.png`;
- `outputs/a2l_distributionally_robust_risk_envelope.csv` and `.png`;
- `outputs/a2m_robust_minimum_memory_sizing.csv` and `.png`;
- `outputs/a2n_anytime_robust_refinement_schedule.csv` and `.png`;
- `outputs/a2w_witness_cover_comparison.csv` and `.png`;
- `outputs/a3_recertification.csv` and `.png`.

`model_free_benchmarks.py` writes:

- `outputs/b5_drone_safety.csv`, `b5_drone_constraint_violations.png`, and
  `b5_drone_safety_margin.png`;
- `outputs/b5_highway_safety.csv`, `b5_highway_constraint_violations.png`, and
  `b5_highway_safety_margin.png`;
- `outputs/b5_long_horizon_advantage.csv` and `.png`;
- `outputs/b5_model_free_gap.csv` and `.png`;
- `outputs/b5_crac_actor_critic.csv` and `.png`.

`crac_paper_benchmarks.py` writes:

- `outputs/paper_crac_benchmark_summary.csv`;
- `outputs/paper_crac_benchmark_aggregate.csv`;
- `outputs/paper_crac_mechanism_steps.csv`;
- `outputs/paper_crac_dependency_status.csv`;
- `outputs/paper_crac_performance_matrix.png`;
- `outputs/paper_crac_safety_matrix.png`;
- `outputs/paper_crac_mechanism_matrix.png`.

`mujoco_crac_benchmarks.py` writes:

- `outputs/mujoco_crac_smoke_summary.csv`;
- `outputs/mujoco_crac_smoke_aggregate.csv`;
- `outputs/mujoco_crac_smoke_returns.png`;
- `outputs/mujoco_crac_training_curve.csv`;
- `outputs/mujoco_crac_training_aggregate.csv`;
- `outputs/mujoco_crac_training_curve.png`;
- `outputs/mujoco_crac_certificate_budget_probe.csv`;
- `outputs/mujoco_crac_certificate_budget_probe.png`;
- `outputs/mujoco_crac_certified_update_curve.csv`;
- `outputs/mujoco_crac_certified_update_curve.png`;
- `outputs/mujoco_crac_certified_seed_robustness.csv`;
- `outputs/mujoco_crac_certified_seed_robustness_aggregate.csv`;
- `outputs/mujoco_crac_certified_seed_robustness.png`;
- `outputs/mujoco_crac_candidate_generator_ablation.csv`;
- `outputs/mujoco_crac_candidate_generator_ablation_aggregate.csv`;
- `outputs/mujoco_crac_candidate_generator_ablation.png`;
- `outputs/mujoco_crac_safety_gate_ablation.csv`;
- `outputs/mujoco_crac_safety_gate_ablation_aggregate.csv`;
- `outputs/mujoco_crac_safety_gate_ablation.png`;
- `outputs/mujoco_crac_horizon_transfer.csv`;
- `outputs/mujoco_crac_horizon_transfer_aggregate.csv`;
- `outputs/mujoco_crac_horizon_transfer.png`;
- `outputs/mujoco_statistical_report.csv`;
- `outputs/mujoco_statistical_report.png`;
- `outputs/mujoco_sb3_budget_sweep_summary.csv`;
- `outputs/mujoco_sb3_budget_sweep_aggregate.csv`;
- `outputs/mujoco_sb3_budget_sweep.png`;
- `outputs/mujoco_sample_budget_comparison.csv`;
- `outputs/mujoco_sample_budget_comparison.png`;
- `outputs/mujoco_baseline_gap_report.csv`;
- `outputs/mujoco_baseline_gap_report.png`;
- `outputs/mujoco_crac_portfolio_streams.csv`;
- `outputs/mujoco_crac_portfolio_streams.png`;
- `outputs/mujoco_sb3_smoke_summary.csv`;
- `outputs/mujoco_sb3_smoke_aggregate.csv`;
- `outputs/mujoco_sb3_smoke_returns.png`.

`mujoco_paper_scale_benchmark.py` is the paper-scale MuJoCo runner.  It is
separate from the smoke and 20k-step budget sweep.  The target matrix is
`HalfCheetah-v5`, `Hopper-v5`, and `Walker2d-v5` crossed with `PPO`, `SAC`,
and `TD3`, using seeds `0..4`.  Every run is trained for at least `1,000,000`
environment steps.  If the 1M checkpoint does not satisfy the script's
convergence criterion, that run continues to `2,000,000` steps.  The runner
records checkpoint learning curves every 100k steps and reports the metrics
common in recent deep-RL evaluation practice: final return, best return, AUC,
sample efficiency to 90% of the best observed return, mean/median/IQM,
bootstrap 95% intervals, optimality gap, performance-profile curves,
probability of improvement, last-three-evaluation stability, final-vs-best
gap, wall-clock time, and FPS.  Each run summary and progress manifest also
records whether the run met the 1M minimum, whether it satisfied the 1M
convergence gate, and whether it was therefore extended toward 2M.  SB3 rows
remain `certificate_type=none`.

The reporting set follows the RLiable-style evaluation checklist used in
modern deep-RL papers: aggregate IQM/median/mean with bootstrap intervals,
performance profiles, optimality gaps, and pairwise probability of improvement
(https://github.com/google-research/rliable).  This is why the paper-scale
runner keeps all seed-level curves instead of reporting only a single point
estimate.

`extended_control_benchmark_catalog.py` prepares the next paper-scale control
and robotics matrix requested for algorithm-paper validation.  It keeps the
same 1M/2M rule and the same PPO/SAC/TD3 seed protocol.  The local runnable
extension currently contains Gymnasium MuJoCo arm/high-dimensional control
tasks:

- `Reacher-v5` for robot-arm reaching;
- `Pusher-v5` for arm/object manipulation;
- `Ant-v5` for high-dimensional locomotion control;
- `Humanoid-v5` for high-dimensional whole-body control.

The same catalog records install-then-run targets that are common in recent
control/robotics RL papers but are not installed in this local environment:
DeepMind Control Suite (`dm_control`), Meta-World, Gymnasium-Robotics Fetch,
ManiSkill, and MyoSuite.  These rows are dependency evidence and remain
`certificate_type=none`.

The current long run is sharded so workers do not contend on one CSV.  Per-shard
outputs live under `outputs/paper_scale_shards/shard_*`, while
`monitor_mujoco_paper_scale.py --merge` merges all shard curves into:

- `outputs/mujoco_paper_scale_eval_curve.csv`;
- `outputs/mujoco_paper_scale_run_summary.csv`;
- `outputs/mujoco_paper_scale_algorithm_summary.csv`;
- `outputs/mujoco_paper_scale_rliable_metrics.csv`;
- `outputs/mujoco_paper_scale_performance_profile.csv`;
- `outputs/mujoco_paper_scale_probability_of_improvement.csv`;
- `outputs/mujoco_paper_scale_metric_catalog.csv`;
- `outputs/mujoco_paper_scale_progress_manifest.csv`;
- `outputs/mujoco_paper_scale_worker_status.csv`;
- `outputs/mujoco_paper_scale_report.md`;
- `outputs/mujoco_paper_scale_learning_curves.png`;
- `outputs/mujoco_paper_scale_final_return.png`;
- `outputs/mujoco_paper_scale_performance_profile.png`.

The extended control/robotics plan writes:

- `outputs/extended_control_benchmark_catalog.csv`;
- `outputs/extended_control_dependency_status.csv`;
- `outputs/extended_control_jobs.csv`;
- `outputs/extended_control_jobs/shard_*.csv`;
- `outputs/extended_control_plan.md`.

`odd_experiments.py` writes:

- `outputs/odd_drone_runtime_cover.csv` and `.png`;
- `outputs/odd_highway_runtime_cover.csv` and `.png`;
- `outputs/odd_application_summary.csv`.

`highway_env_validation.py` writes:

- `outputs/highway_env_episode_metrics.csv`;
- `outputs/highway_env_summary.csv`;
- `outputs/highway_env_statistical_report.csv`;
- `outputs/highway_env_pareto_front.csv`;
- `outputs/highway_env_collision_free.png`;
- `outputs/highway_env_success_rate.png`;
- `outputs/highway_env_time_steps.png`;
- `outputs/highway_env_reward.png`;
- `outputs/highway_env_pareto_*.png`;
- `outputs/highway_env_calibration_episode_metrics.csv`;
- `outputs/highway_env_calibration_sweep.csv`;
- `outputs/highway_env_calibration_pareto.csv`;
- `outputs/highway_env_calibration_pareto.png`;
- `outputs/highway_env_calibration_tradeoff.png`.

`ev_autonomous_validation.py` writes:

- `outputs/ev_autonomous_episode_metrics.csv`;
- `outputs/ev_autonomous_summary.csv`;
- `outputs/ev_autonomous_statistical_report.csv`;
- `outputs/ev_autonomous_pareto_front.csv`;
- `outputs/ev_autonomous_success.png`;
- `outputs/ev_autonomous_energy.png`;
- `outputs/ev_autonomous_soc_min.png`;
- `outputs/ev_autonomous_ttc.png`;
- `outputs/ev_autonomous_battery_violations.png`.
- `outputs/ev_autonomous_pareto_*.png`.
- `outputs/ev_stress_episode_metrics.csv`;
- `outputs/ev_stress_summary.csv`;
- `outputs/ev_stress_success_by_soc.png`;
- `outputs/ev_stress_battery_by_soc.png`;
- `outputs/ev_stress_soc_min_by_soc.png`.

`safety_gymnasium_validation.py` writes:

- `outputs/safety_gymnasium_episode_metrics.csv`;
- `outputs/safety_gymnasium_summary.csv`;
- `outputs/safety_gymnasium_gate_report.csv`;
- `outputs/safety_gymnasium_reward.png`;
- `outputs/safety_gymnasium_cost.png`;
- `outputs/safety_gymnasium_dependency_status.csv` when the dependency is not
  runnable in the active Python environment.

`validation_dashboard.py` writes:

- `outputs/top_level_algorithm_comparison.csv`;
- `outputs/top_level_reward_curves.csv`;
- `outputs/top_level_reward_curves.png`.

Every CSV written by these scripts includes `certificate_type`; the tests scan
generated CSVs and reject model-free rows labeled as `exact`.
