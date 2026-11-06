# DaRiAn — Frame-level ball possession

## Approach

The submitted method is possession v4: a team-symmetric ensemble of histogram gradient-boosting
classifiers followed by offline temporal decoding. It produces one H/A/D/X label per tactical
video frame. No manual match labels, hand corrections, match-specific transition constants,
private ground truth, or competition-public-label tuning are used.

## Assets used

Inference uses the provided tactical ATD metadata and tracking TXT. Player positions and ball
positions are converted to metres. Distances aggregate by team; persistent player identities
are not assumed. Features include nearest-player distances, recent and future control evidence,
ball visibility and movement, pitch-edge distance, and collective player movement. Each half
is processed independently using the published timeline boundaries.

The classifier omits Smart Tagging, homography, and video because their incremental value was
not independently established in the available validation data. A separate conservative
broadcast-alignment experiment recovered only 22 additional ball observations and was excluded.

## Training data and labels

Source: Metrica Sports Sample Data, https://github.com/metrica-sports/sample-data, revision
e706dd506b360d69d9d123d5b8026e7294b13996. These public anonymised matches predate the 2024
competition match. Metrica publishes the tracking and event annotations; individual annotators
and their complete annotation workflow are not identified in the source. The team does not
manually annotate any match.

Games 1 and 2 supply training and model-selection data. Event annotations are mechanically
converted into proxy possession labels: deliberate actions and recoveries establish a team,
while ball-out actions, goals/out shots, and received fouls establish dead ball. State carries
to the next marker within that period. Game 3 additionally supplies explicit carry events for
its proxy reference. These proxies are approximate and can disagree with corrected frame-level
possession semantics, especially on fleeting recoveries and exact restart boundaries.

Model 1 trains on Game 1; model 2 trains on Game 2. Their augmented training row counts are
219,368 and 216,824 respectively, including team mirroring. Four training conditions are used:
clean tracking, contiguous ball gaps, correlated position errors with brief false tracks and
missing players, and combined gaps/position errors. Distances are recalculated from perturbed
geometry. Perturbations are seeded algorithms that do not accept label arrays.

Training samples every fifth frame and excludes a small region around approximate event
boundaries. Models use 120 boosting iterations, learning rate 0.08, at most 15 leaves per tree,
minimum 80 samples per leaf, L2 regularisation 5, no random validation split, and seed 2026.
Home/away mirroring is used in training and prediction. Inference averages both models.

## Possession decisions

The classifier estimates H/A/D evidence. A three-state sequence decoder integrates evidence
over each entire half and applies a penalty for label changes. Future observations can resolve
earlier gaps, so the pipeline is offline. Emission losses are summed into five-frame groups;
the CSV still contains every original frame. The selected switch penalty is 1.0 with no hard
minimum possession duration. X is emitted outside the published halves.

The method models possession continuity rather than assigning ownership to every nearest-player
touch. It can still mistake a deflection for control, merge genuine short possessions, or infer
an incorrect stoppage. There is no manual rule or correction for a specific match timestamp.

## Model selection and external evidence

Two leave-one-game-out folds on Games 1/2 select decoder settings. Each development prediction
uses the model trained on the other game. Selection weights mean spellF1 and the weakest
corruption-condition mean equally. The predeclared search includes switch penalties
0.25/0.5/1/2 and minimum durations 0/0.4/0.8 seconds.

Game 3 is excluded from fitting and this selection. Using its event-derived proxy reference:

| Input condition | Submitted model spellF1 |
|---|---:|
| Clean | 0.6203 |
| Ball gaps | 0.5543 |
| Position errors | 0.6080 |
| Combined errors | 0.4983 |

These are four conditions of one external match, not four independent matches and not official
hackathon scores. A later calibration/separate-transition-cost candidate was fitted and selected
on Games 1/2, then checked against the reused Game 3 reference. Its average spellF1 was 0.5689
versus 0.5702 for v4; it was rejected. That later comparison is a repeated external confirmation,
not a new blind test. It did not alter the submitted v4 model.

## Reference-quality correction found during the audit

The earlier proxy builder omitted fouls encoded as CHALLENGE qualifiers. A separate corrected
event-only reader now recognises those fouls and explicit offside/handball/end-half annotations,
preserves advantage, and does not treat recovery of a dead ball as a restart. No tracking signal
or prediction is used to make this correction. It changes external reference labels, so its
scores must not be mixed with the historical table above.

The corrected-reference experiment did not pass the development promotion gate. The release
retains the frozen v4 weights and their original weak-reference training procedure. Proxy
uncertainty remains a limitation. Earlier implementations and experimental audit tools are
not part of this final submission repository.

## Reproduce

Extract the source archive into a public repository or local folder, install Python 3.11, and run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python starter/run_darian.py --data-dir "/path/to/match-bundle" --output DaRiAn.csv
```

The bundle root must contain tactical/ with the supplied ATD XML and TXT. The included trained
model uses public OTHER-match data only; no restricted match data is included. The runner checks
model/source hashes, runs inference and the official format validator, then checks exact output
SHA-256. Final-model training runs independently with
`python starter/train_possession_v4.py --download`; manifests pin all public inputs.
No earlier model or training stage is required. Shared parsing and diagnostics were extracted
into neutral modules without changing the packaged weights. Historical source paths embedded
in the weights record their original training provenance; DARIAN_CONFIG.json records current
release source hashes. Retraining records current source hashes, so the serialized model's
metadata/hash can differ from the original release even when predictions agree.

Release verification includes synthetic feature/decoder/perturbation tests, official format
validation, and reproduction of the complete CSV with the packaged weights. The expected
submission hash below is checked automatically by the release runner.

Model SHA-256: `a9c55312ce995cd6a8bf4c83dc03fb61b02db568a6db601b8bcb6b4126f243df`

Submission SHA-256: `b93cae4a40ba0e5db253b2ac7b8263699ff7b1ee0913adb165bb582b56a22d62`

## Limitations

Only three external matches and approximate event-based references are available. Synthetic
errors do not capture every real tracking failure. Model probabilities are not calibrated
guarantees. Ambiguous restarts, short possessions, long occlusions, and distribution differences
remain failure modes. Better external references and more independent automatic-tracking games
are needed to substantiate stronger accuracy claims. The private competition score is unknown.
