# Language Control or Trajectory Index?

## Initial Research Project Plan

**Status:** Working plan  
**Project stage:** Problem definition and diagnostic design  
**Primary objective:** Determine whether language causally controls how a robot acts, or mainly identifies familiar behaviors already favored by the visual observation and training distribution.

---

## 1. Working research question

> When a robot policy is physically capable of multiple valid behaviors in the same scene, can language reliably select and modify the requested behavior, including in unseen combinations and during execution?

The project should initially remain architecture-agnostic. Route selection, bimanual arm choice, action order, approach direction, and other behaviors are candidate diagnostic probes rather than fixed commitments.

### Working hypothesis

Many vision-language-action policies use language effectively for coarse task or object identification but do not consistently ground language as a causal, compositional, and persistent control signal over execution.

In the limiting case, language behaves like a task or trajectory index: it helps retrieve a familiar behavior cluster but cannot reliably modify how that behavior is executed under a new combination of constraints.

### Competing hypothesis

Language is grounded in reusable behavioral factors. When a relevant phrase changes, the requested component of behavior changes predictably while unrelated components remain stable, including for unseen factor combinations.

---

## 2. Why this question matters

Task success alone does not prove language control. A policy can succeed because the scene uniquely determines the expected behavior, even if the instruction is ignored. Conversely, a policy can follow the requested behavioral choice but fail later because of insufficient motor capability.

This project therefore separates three properties:

1. **Capability:** Can the policy physically execute each candidate behavior?
2. **Controllability:** Can a language intervention select the requested behavior?
3. **Execution:** After selecting it, can the policy complete the task?

The separation prevents a motor failure from being misclassified as a language-grounding failure and prevents aggregate task success from concealing language neglect.

---

## 3. Operational definition of language control

A policy is language-controllable along behavioral factor \(f\) when:

1. the scene and goal permit at least two valid behaviors;
2. changing only the instruction changes factor \(f\) in the requested direction;
3. unrelated behavioral factors remain approximately unchanged;
4. paraphrases produce equivalent effects;
5. the effect persists in unseen but feasible combinations; and
6. an instruction change during execution can alter the remaining behavior when physically possible.

### Indexing behavior versus grounded control

| Diagnostic | Trajectory/task indexing | Grounded language control |
|---|---|---|
| Arbitrary ID versus natural language | Similar performance | Language generalizes beyond arbitrary IDs |
| Paraphrases | Unstable or memorized | Semantically equivalent behavior |
| Counterfactual instruction in the same scene | Default visual behavior persists | Requested behavior changes |
| Held-out combination | Snaps to a trained combination | Familiar factors recombine |
| Mid-episode correction | Weak or delayed response | Remaining behavior changes predictably |
| Single-factor edit | Whole trajectory changes or no response | Only the requested factor changes |
| Behavior absent from all training | Usually impossible | Not required for the core claim; this tests generation, not control |

---

## 4. Scope and non-goals

### Initial scope

- Simulation-first manipulation experiments.
- At least one language-conditioned action model.
- Counterfactual pairs with the same or closely matched visual observations.
- Behavior-level metrics in addition to task success.
- Familiar behaviors recombined in held-out task-control combinations.
- Static instructions and selected mid-episode interventions.

### Candidate behavioral factors

| Factor | Example instruction pair | Possible metric |
|---|---|---|
| Object | “Pick the left block” / “Pick the right block” | Selected object |
| Destination | “Place in red target” / “Place in blue target” | Selected target |
| Order | “A then B” / “B then A” | First completed subtask |
| Motion direction | “Move left” / “Move right” | Signed displacement |
| Approach | “Approach from above” / “Approach from the side” | Approach vector/angle |
| Orientation | “Rotate clockwise” / “Rotate counterclockwise” | Signed angular displacement |
| Route | “Go around left” / “Go around right” | Route class or crossing side |
| Manner | “Move slowly” / “Move quickly” | Velocity profile |
| Online correction | “Stop,” “switch,” “use the other side” | Post-intervention behavior change |
| Bimanual role | “Left holds, right acts” / reverse | Actor-role assignment |

### Non-goals for the first stage

- Proving that all VLAs are vision-dominated.
- Requiring the policy to invent a motor behavior absent from its entire training history.
- Claiming that any response to a visual obstacle demonstrates physical reasoning.
- Committing to path topology, bimanual manipulation, or a two-stage architecture before the diagnostic evidence supports that choice.
- Treating unsafe language compliance as desirable when an instruction conflicts with physical feasibility.

---

## 5. Core research questions

### RQ1: Does language causally affect behavior?

With observation and goal held fixed, do counterfactual instructions produce the requested behavioral difference?

### RQ2: Which semantic factors are controllable?

Is language more effective for selecting objects and goals than for specifying approach, order, route, orientation, speed, or coordination?

### RQ3: Is language control compositional?

Can the policy combine a familiar task with a familiar control factor when that exact pairing was withheld from training?

### RQ4: Is language influence persistent?

Does an instruction affect only the initial action, or constrain the behavior throughout execution?

### RQ5: What causes failure?

Possible causes include insufficient counterfactual data, language-action fusion, visual shortcuts, catastrophic forgetting, action-chunk inertia, weak motor capability, or failure to represent the requested control factor.

### RQ6: What is the smallest effective intervention?

Can the diagnosed problem be addressed through data, objectives, conditioning, inference-time guidance, explicit control representations, or hierarchy?

---

## 6. Experimental design principles

### Principle A: Establish capability first

Before testing language control, verify that the policy can execute both alternatives under some supported condition. If a policy has never successfully used the left arm, a failed “use the left arm” trial does not isolate language control.

### Principle B: Make language necessary

Both requested behaviors should be visually and physically feasible. The scene alone should not determine the correct choice.

### Principle C: Use paired counterfactual trials

Clone the same simulator state and change only the instruction. Preserve object positions, camera state, robot state, lighting, and random seed whenever possible.

### Principle D: Score the requested factor directly

Measure route, actor, order, approach, or direction independently from final success.

### Principle E: Separate recombination from invention

The main generalization test should withhold a **combination**, not every example of the requested behavior. For example, teach both arms across generic tasks but train one target task with only the right arm.

### Principle F: Include positive language controls

Demonstrate that the instruction channel can affect at least one factor for the same checkpoint. This distinguishes selective semantic failure from a dead language input.

### Principle G: Avoid ambiguous safety conflicts

The main controllability result should use two feasible options. Physical-conflict conditions can be reported separately as safety or arbitration tests.

---

## 7. Proposed diagnostic benchmark

### Stage A: Small pilot

Select two or three factors that are easy to classify automatically:

1. object or destination selection;
2. action order or approach direction; and
3. route selection or bimanual role assignment.

The factors should span increasing control complexity:

- **Discrete endpoint selection**: what is manipulated or where it goes.
- **Execution attribute**: how the manipulation is performed.
- **Temporally extended constraint**: how behavior must remain organized over time.

### Stage B: Counterfactual factorial design

For every selected factor, construct a matrix such as:

| Task | Control A | Control B |
|---|---:|---:|
| Task 1 | Train | Train |
| Task 2 | Train | Train |
| Target task | Train | **Held-out test** |

The held-out cell tests recombination. Additional evaluations should include:

- synonymous instructions;
- irrelevant language changes;
- empty instruction;
- scrambled instruction, reported only as distribution-shift evidence;
- matched visual distractors;
- mid-episode instruction changes;
- feasible vision-language conflicts;
- unseen object/layout/camera variations.

### Optional route probe

Use the DETOUR experiment as one temporally extended diagnostic:

- both left and right routes demonstrated somewhere in training;
- only one route demonstrated for the target configuration;
- both lanes open for the language-only test;
- solid and collider-free blockers as visual interventions;
- route choice scored separately from completion;
- full route-withdrawal ablation to distinguish repertoire selection from synthesis.

### Optional bimanual probe

Prefer relational coordination over simple arm selection:

- left holds / right manipulates versus the reverse;
- left-to-right versus right-to-left handoff;
- simultaneous versus sequential lifting;
- leader/follower role reversal;
- held-out actor-object or actor-role binding.

Simple left/right arm choice is useful but should not be the sole bimanual contribution.

---

## 8. Models and baselines

Do not choose the proposed architecture until the initial audit identifies the failure mode. Begin with diagnostic baselines.

### Minimum baselines

1. **Standard VLA:** ordinary task-level instruction conditioning.
2. **No-language policy:** estimates what vision and state alone can accomplish.
3. **Arbitrary-ID conditioning:** replaces meaningful instructions with task/control identifiers.
4. **Fine-grained relabeling:** adds execution descriptions without architectural changes.
5. **Counterfactual-pair training:** balances instructions and actions under matched observations.
6. **One recent steerable or language-motion approach**, if implementable in the chosen environment.

### Candidate intervention families

Only pursue these after diagnosis:

- counterfactual data augmentation;
- action-aligned fine-grained captions;
- stronger or repeated language conditioning during action generation;
- language-conditioned residual or guidance branch;
- factorized control variables;
- interpretable control bottleneck;
- hierarchical planner/executor;
- language-conditioned spatial or temporal cost field;
- regularization preserving controllability during task fine-tuning.

### Selection rule

Choose the simplest intervention that targets the demonstrated failure. Do not introduce hierarchy or explicit planning if balanced counterfactual supervision already solves the problem.

---

## 9. Metrics

### Primary metrics

**Factor compliance**

\[
P(B_f = b_f^{\text{requested}})
\]

**Paired causal effect**

\[
CE_f = P(B_f=1\mid do(l_f=1),o)-P(B_f=1\mid do(l_f=0),o)
\]

**Task success**

Report final task completion separately.

**Conditional execution success**

\[
P(\text{success}\mid B_f = b_f^{\text{requested}})
\]

This distinguishes correct selection followed by failed execution.

### Secondary metrics

- paraphrase consistency;
- irrelevant-language invariance;
- held-out composition accuracy;
- instruction-switch latency;
- post-switch compliance;
- collision or constraint violations;
- completion time and path length;
- unintended changes to non-target factors;
- calibration or confidence, if available.

### Statistical plan

- Use paired analyses when simulator states are cloned.
- Report confidence intervals, not only percentages.
- Treat independently trained checkpoints as the main replication unit.
- Use at least three train seeds where feasible.
- Predefine behavioral classifiers and ambiguous-zone handling.
- Report per-condition results rather than relying only on pooled success.

---

## 10. Diagnostic decision tree

### Finding 1: The model cannot perform one alternative anywhere

Interpretation: capability failure. Add motor coverage or choose a different factor before making a language-control claim.

### Finding 2: It performs both alternatives, but language cannot select them

Interpretation: controllability failure. Test counterfactual paired data and language-action conditioning.

### Finding 3: Counterfactual paired training fixes the problem

Interpretation: primarily a data-identifiability failure. A data-centric method may be sufficient.

### Finding 4: Control works in-distribution but fails on held-out combinations

Interpretation: compositional representation or task-specific overfitting failure. Investigate factorization, regularization, or hierarchy.

### Finding 5: Control works initially but disappears during execution

Interpretation: temporal persistence or action-chunk conditioning failure. Investigate repeated conditioning, replanning, or constraint-state tracking.

### Finding 6: Requested factor is correct but the task fails

Interpretation: execution failure. Improve the low-level policy without relabeling the event as a grounding failure.

---

## 11. Initial implementation plan

### Phase 0: Reproduce and instrument

- [ ] Freeze one baseline checkpoint and environment version.
- [ ] Implement deterministic simulator-state cloning.
- [ ] Implement automatic behavior classifiers.
- [ ] Record action, state, image, instruction, episode seed, and classifier output.
- [ ] Verify that language reaches the model exactly as intended.
- [ ] Create a standardized rollout and video-generation script.

**Exit criterion:** Repeated trials are reproducible and behavioral factors can be scored without relying on final success.

### Phase 1: Capability map

- [ ] Choose 2-3 candidate behavioral factors.
- [ ] Verify that each alternative can be executed by the baseline policy.
- [ ] Identify invalid or physically asymmetric conditions.
- [ ] Establish expert feasibility and completion time.

**Exit criterion:** Every language-control comparison uses alternatives the policy or matched expert can physically execute.

### Phase 2: Language-control audit

- [ ] Run paired counterfactual instructions in matched scenes.
- [ ] Compare natural language with arbitrary IDs and no-language input.
- [ ] Test paraphrases and irrelevant text variations.
- [ ] Introduce mid-episode switches at predefined decision points.
- [ ] Score factor compliance, task success, and conditional completion.

**Exit criterion:** At least one reproducible controllability failure is localized to a semantic factor rather than motor capability.

### Phase 3: Generalization audit

- [ ] Create held-out task-control combinations.
- [ ] Vary layout, object, visual appearance, and camera parameters.
- [ ] Compare in-distribution control with recombinational generalization.
- [ ] Repeat across independent checkpoints.

**Exit criterion:** Determine whether the failure is in-distribution grounding, OOD composition, or both.

### Phase 4: Mechanism ablations

- [ ] Balance counterfactual instruction-action pairs.
- [ ] Increase instruction granularity.
- [ ] Compare early-only versus persistent language conditioning.
- [ ] Probe or ablate visual and language features where technically possible.
- [ ] Test full fine-tuning versus frozen/regularized components if task adaptation is used.

**Exit criterion:** Obtain evidence favoring one or more causal explanations.

### Phase 5: Proposed method

- [ ] Select the smallest intervention justified by Phase 4.
- [ ] Define the method before inspecting final test results.
- [ ] Compare with data-matched and compute-matched baselines.
- [ ] Evaluate all diagnostic factors, not only the factor used to design the method.
- [ ] Measure tradeoffs with ordinary task success and safety.

**Exit criterion:** The intervention improves causal language control without materially degrading motor performance.

### Phase 6: Validation and paper preparation

- [ ] Replicate across seeds and at least one additional policy family if feasible.
- [ ] Freeze evaluation classifiers and test splits.
- [ ] Conduct failure analysis with representative videos.
- [ ] Release benchmark definitions, splits, and evaluation code.
- [ ] Write limitations and distinguish negative search evidence from novelty proof.

---

## 12. Suggested first four-week sprint

### Week 1: Definitions and instrumentation

- Select two simple factors and one temporally extended factor.
- Implement state cloning and behavior classifiers.
- Write a one-page preregistration of hypotheses and exclusions.

### Week 2: Capability and paired controls

- Verify both alternatives for every condition.
- Run matched instruction pairs on the baseline.
- Confirm positive and negative language controls.

### Week 3: Composition split

- Construct a small factorial train/test split.
- Train at least two independent checkpoints.
- Compare task success with factor compliance.

### Week 4: First diagnosis

- Run counterfactual relabeling as the simplest intervention.
- Decide whether the main bottleneck appears to be data, conditioning, persistence, or composition.
- Select the next model direction only after this review.

### First milestone deliverable

A compact result table answering:

1. Can the policy execute both alternatives?
2. Does language select between them in matched scenes?
3. Does selection generalize to a held-out task-control combination?
4. Does a mid-episode language change alter behavior?
5. Does counterfactual paired training improve the result?

---

## 13. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Instruction is out of distribution | Failure is unsurprising | Include trained wording, paraphrases, and action-aligned relabeling |
| One behavior is physically harder | Controllability is confounded | Match feasibility and report conditional execution |
| Vision changes between paired trials | Language effect is not isolated | Clone simulator states and random seeds |
| Behavior classifier is subjective | Results are unstable | Predefine geometric rules and an ambiguous category |
| Only one checkpoint is tested | Rollout counts overstate evidence | Replicate across train seeds |
| Proposed method memorizes benchmark phrases | Apparent improvement lacks semantics | Test paraphrases, arbitrary IDs, and unseen compositions |
| Language compliance causes unsafe behavior | Control conflicts with feasibility | Separate controllability from safety arbitration |
| Scope becomes too broad | Weak depth across many factors | Pilot broadly, then select 2-3 factors for the main paper |

---

## 14. Provisional paper structure

1. **Introduction:** Does language control how a robot acts?
2. **Related work:** VLA grounding, visual shortcuts, steerable policies, multimodal imitation, compositional control.
3. **Definition:** Capability, controllability, composition, and execution.
4. **Benchmark:** Paired counterfactual scenes and behavioral-factor metrics.
5. **Audit:** Current policies across semantic factors.
6. **Diagnosis:** Data, fusion, persistence, and composition ablations.
7. **Method:** Intervention selected from the diagnosis.
8. **Evaluation:** In-distribution control, held-out combinations, switches, and robustness.
9. **Limitations:** Feasibility, policy coverage, safety, embodiment, and simulator dependence.

---

## 15. Working contribution statement

> We introduce a controlled framework for determining whether language causally controls robot execution rather than merely indexing familiar task or trajectory modes. The framework separates motor capability, behavioral-factor compliance, and final task execution under paired counterfactual instructions, held-out task-control combinations, and online interventions. Based on the resulting diagnosis, we develop and evaluate a targeted method for improving persistent and compositional language control.

This statement should remain provisional until the baseline audit and method results are available.

---

## 16. Seed literature

- [When Vision Overrides Language / LIBERO-CF](https://arxiv.org/abs/2602.17659)
- [CAST: Counterfactual Labels Improve Instruction Following](https://arxiv.org/abs/2508.13446)
- [Steerable Vision-Language-Action Policies](https://arxiv.org/abs/2602.13193)
- [FineVLA](https://arxiv.org/abs/2605.27284)
- [RT-H: Action Hierarchies Using Language](https://arxiv.org/abs/2403.01823)
- [AC-VLA: Robust OOD Action Execution via Compositional Learning](https://arxiv.org/abs/2607.15714)
- [Behavior Transformers](https://arxiv.org/abs/2206.11251)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [Causal Confusion in Imitation Learning](https://arxiv.org/abs/1905.11979)

---

## 17. Research log template

### Experiment ID

**Date:**  
**Checkpoint/data seed:**  
**Environment version:**  
**Behavioral factor:**  
**Capability verified:** Yes / No  
**Training combination:** Seen / Held out  
**Instruction intervention:**  
**Visual intervention:**  
**Number of trials:**  

### Results

- Factor compliance:
- Task success:
- Success conditional on correct factor:
- Collision/constraint violations:
- Ambiguous trials:

### Interpretation

- Capability failure, controllability failure, composition failure, or execution failure?
- Strongest alternative explanation:
- Required follow-up control:

### Decision

- Continue / modify / stop this experiment:
- Reason:

