# CausalMem Phase 5.3 — Evaluation Protocol and Reproducibility

**Status:** Protocol prepared; no Phase 5.3 evaluation was run in this task.  
**Purpose:** Define a repeatable evaluation plan for temporal retrieval, causal
evidence validation, abstention, user isolation, and answer provenance. This
document is a protocol, not an evaluation result.

## 1. Evidence classes and separation

Report these evidence classes separately. Never combine their counts into one
quality score.

| Evidence class | Current source | What it measures | Reporting rule |
|---|---|---|---|
| Deterministic tests and policy fixtures | `tests/`; `evaluation/research_v1.jsonl`; `research_evaluator.py` | Code paths, fixed-response validation, persistence, isolation, interval boundaries, and failure handling | Report pytest counts and policy-evaluator metrics as deterministic. Do not call them live model performance. |
| Synthetic live-pipeline evaluation | `evaluation/live_research_v1.jsonl`; `research_live_evaluator.py` | Actual Mem0/Ollama ingestion, retrieval, and answer generation on a small fixed synthetic set | Report completed, failed, skipped, and interrupted cases separately; report proxy metrics with counts and denominators. |
| Manual live verification | Operator-observed Streamlit interaction | One observed UI behavior and its text-free diagnostics | Label as manual and user-reported; keep it separate from automated result JSON and do not assign a test ID or date unless recorded. |

The recorded Phase 5.1 deterministic policy artifact has 10 cases. Its
historical results report answerability 9/9, abstention precision/recall 4/4,
source-ID proxy F1 6/6, claim-source groups 6/6, temporal state matches 2/2,
causal transition-ID precision/recall 1/1, and retrieval-failure detection
1/1. Those results use fixed evidence and fixed model responses; they are not
live LLM or Mem0 ranking measurements.

The historical Phase 5.2 final six-case run is recorded in
`evaluation/README.md` and its isolated artifact. It reported 6 completed
cases, 0 failures, 0 skips, exact fixture-key evidence F1 8/8, answer/abstain
agreement 6/6, required-fact substring recall 7/7, forbidden-fact checks 0/4,
citation groups 6/6, as-of state keys 3/3, and causal/provenance fixture-key
proxies 1/1. These are small-sample proxies, not semantic correctness or
general reliability. A preceding run abstained on `asof-after-replacement`
after lexical support failed; the later all-pass run does not establish that
the issue cannot recur.

The operator has separately reported a successful Streamlit query,
“Why did I switch to swimming?” Its observed diagnostics were
`causal_gate_passed=true`, `required_transition_citation_accepted=true`,
`answer_reason=validated_claims`, and one accepted claim. This is a manual
verification only; it was not added to or inferred from a saved benchmark
artifact. No timestamp, test ID, or additional measurement is recorded here.

## 2. Evaluation cases

Use synthetic users and messages only. Use isolated temporary SQLite and
Mem0/Qdrant stores for integration and live cases. Deterministic cases should
use fixed model responses and mocked Mem0 responses; live cases should use the
unchanged, versioned live dataset unless a separately versioned dataset change
is reviewed. For every case, retain the runtime evidence IDs in the private
result artifact and score only the declared references; do not put expected
answers or labels in model evidence.

| Case | Purpose and setup | Expected behavior and pass/fail criteria | Required evidence |
|---|---|---|---|
| `cause-explicit-valid` | Same-user `running` and `swimming` states; one `EXPLICIT_CAUSE` transition with a recorded knee-injury trigger/reason; fixed answer cites the transition | Answer accepted only when it cites the transition, names supported cause/change content, and passes endpoint, relation, ownership, and lexical/polarity checks | Transition citation resolves to the same-user edge; both endpoint states resolve for that user; trigger/reason are nonempty and claim-supported |
| `cause-prompt-wording` | Use the actual transition structure and the validator regression wording “The recorded reason for the change was a knee injury” | Pass if the cause claim is accepted with a transition citation; this locks the supported causal-framing normalization | Same owned transition and endpoint evidence; text-free `validated_claims` result |
| `cause-unsupported-detail` | Explicit cause says knee injury; fixed candidate adds an unsupported symptom or consequence such as dizziness | Abstain/reject the claim; fail if unsupported added detail is rendered as established fact | Candidate rejected by causal lexical support; no accepted transition claim |
| `cause-not-established` | Same endpoints linked only by `TEMPORAL_SUCCESSION`, `UNKNOWN`, or inferred relation, with no explicit cause | Abstain before causal answer generation when no other valid cause exists | No eligible explicit-cause edge; gate rejection |
| `cause-ambiguous` | Two relevant same-user explicit-cause edges can answer the same question, without a unique discriminator | Preserve both candidates and abstain or mark ambiguity; fail if one cause is selected arbitrarily | Both owned transition records and explicit ambiguity outcome |
| `citation-integrity` | Compare correct transition citation with memory-only, state-only, unknown/mismatched ID, wrong relation, and foreign-owner citations | Only the exact valid transition reference can authorize the causal claim; every invalid form is rejected | Reference kind/ID resolution booleans and stable reason codes; no IDs or content in UI diagnostics |
| `endpoint-ownership` | Cite a same-user transition whose source or destination is missing or foreign; retain an unrelated valid edge only to ensure the global gate can run | Reject the cited edge; fail if foreign/missing endpoint labels validate it | Transition owner and endpoint-owner checks; unresolved citation must not render |
| `user-isolation` | Alice and Bob have deliberately similar facts and transitions; retrieve and answer as Alice | Alice's evidence and citations contain only Alice-owned records; fail on any Bob evidence entering the bundle or answer | Requested user scope, fixture attribution, and foreign-evidence leakage count |
| `asof-before` | Goal/preference replacement with an explicit recorded transition instant; request an instant immediately before it | Retrieve predecessor only; linked Mem0 record must agree with state validity | Actual retriever output, state key, linked memory attribution, and normalized requested timestamp |
| `asof-at-boundary` | Same replacement; request exactly the transition instant | Retrieve successor only under half-open intervals | Actual retriever output; old state/memory excluded and successor included |
| `asof-after` | Same replacement; request after the transition instant | Retrieve successor; no future state or transition beyond the request may leak in | Actual retriever output and transition timestamp ordering |
| `history-predecessor` | One user has exactly one relevant replacement edge; query asks what was true before the change | Retrieve predecessor and cite its valid state or uniquely linked memory; fail if successor is substituted | Same-user transition boundary links predecessor end and successor start; predecessor evidence resolves |
| `stale-or-contradictory-current` | Seed conflicting distinct Mem0 IDs and state history with an older value and a newer replacement; also test an intentionally stale `ACTIVE` predecessor row in a separate fixture | Preserve conflicting distinct memories; current query must not present a superseded value as current. If stored status/interval/transition disagree, mark the case inconsistent and fail pending diagnosis rather than silently choosing | Raw isolated SQLite status/interval/transition structure, retrieved state IDs, and source attribution in restricted artifacts |
| `unchanged-fact` | Add a replacement for one state dimension while a separate fact remains valid | Keep the unchanged fact available; do not close it due only to a broad state-type match | State type/value and interval of both unchanged and replaced states |
| `unsupported-abstention` | Ask for a personal fact absent from the user’s evidence | Return the existing abstention; fail if a personal fact is inferred from query keywords or unrelated evidence | Empty/mismatched evidence outcome and abstention diagnostics |
| `retrieval-source-failure` | Deterministically fault-inject Mem0, SQLite state, transition, or graph retrieval in separate tests | Report retrieval failure distinctly from an ordinary empty result; never score infrastructure failure as a valid abstention | Failure source, coarse error category, and case status; no raw exception text |

The current `live_research_v1.jsonl` contains 12 version-1.0 cases; the
`retrieval-failure-mock-only` case is intentionally not a live model case.
The current `research_v1.jsonl` contains 10 version-1.0 deterministic policy
cases. The separately versioned `research_stale_state_v1.jsonl` supplemental
fixture set covers an intentionally stale ACTIVE predecessor and expects
abstention when retrieval marks the replacement history inconsistent.
Its `dataset_version` remains the evaluator schema version (`1.0`); its
`fixture_set_version` identifies the supplemental case release. Retrieval
behavior is also covered by isolated SQLite tests, since this policy evaluator
uses fixed evidence and does not itself run the full retriever. The table above
is a protocol coverage map, not a claim that all listed cases are present in a
single dataset. Additions must be versioned and must not rewrite historical
datasets or result files.

## 3. Metrics and reporting rules

For every metric report numerator, denominator, computed value, and case IDs
included. A zero denominator is **undefined / N/A**, not zero and not a pass.
Do not include skipped, unavailable, failed, or interrupted cases in the
completed-case numerator. Report their counts separately and explain the
reason. An infrastructure failure is not a model abstention. A run with any
failed or interrupted case is incomplete and must not be labeled all-pass.

| Metric | Definition and denominator | Interpretation limit |
|---|---|---|
| Deterministic test outcome | Pytest passed / failed / errors / skipped counts; report exit code and warnings | Code regression evidence only; not answer quality |
| Answerability accuracy | Correct `answer`/`abstain` outcomes divided by completed, answerable evaluation cases | Labels correctness of answer-vs-abstain only, not semantic answer truth |
| Abstention precision | Expected-abstain and predicted-abstain cases divided by all predicted abstentions | N/A if no predicted abstentions |
| Abstention recall | Correctly abstained expected-abstain cases divided by all expected-abstain cases | N/A if no expected abstentions |
| Evidence/source-ID precision, recall, F1 | Exact expected fixture-key set: TP/(TP+FP), TP/(TP+FN), harmonic mean; report case and key counts | Fixture-key mapping proxy; not entailment |
| Citation-group agreement | Exact actual versus expected citation source group; denominator is all expected/actual groups according to evaluator rule | Attribution proxy; not proof each claim follows from cited text |
| Claim validation rate | Accepted supported claim groups / all proposed groups, with rejected reason counts | Policy adherence proxy; a validator passing is not human semantic verification |
| Temporal state accuracy | Exact expected state fixture-key set among states returned by the actual end-to-end retriever, divided by annotated end-to-end as-of cases | Do not oracle-filter returned evidence before scoring; keep oracle interval checks separate |
| Causal transition/provenance score | Exact transition-source fixture-key precision/recall/F1 over explicitly annotated cases; separately count valid transition-citation outcomes | Does not establish scientific causation or truth of user attribution |
| Unsupported causal assertion rate | Unsupported detected causal assertions / all detected causal assertions; report detection method and manual-review count | Lexical marker detector can miss implicit claims; not a complete semantic measure |
| Cross-user leakage rate | Foreign evidence items or citations / explicit foreign-evidence checks, reported by source and total | Any observed leak is a safety failure; unknown-owner records are not silently considered owned |
| Retrieval failure detection | Fault-injected failures surfaced / expected fault injections; also report false failure classifications | Deterministic/mock failure metric; keep separate from live performance quality |
| Latency | Report sample count and measurement boundary; mean/median and p95 only when sample count meets the evaluator's 20-observation threshold | Separate ingestion, retrieval/classification, answer generation, and whole-run timing; no cause inferred from latency alone |

The live evaluator currently reports fixture-key, substring, citation-group,
temporal, causal, and latency measures. Required/forbidden answer facts use
normalized case-insensitive substring matching. These are proxies, not
semantic entailment. Do not label a lexical match or transition-ID match as
proof of factual correctness or causation. Manual review should be reported as
a separate count with a documented review procedure; do not fabricate a
reviewer or consensus score.

## 4. Reproducibility and isolation requirements

For each run record, without secrets:

- dataset path, dataset version, evaluator version/schema version, SHA-256 of
  the unchanged dataset, selected case IDs in execution order, and subset rule;
- Python version, installed `mem0ai`, `qdrant-client`, `openai`, and Streamlit
  versions where available, plus the project `requirements.txt` revision;
- provider, chat model, embedding model, and whether Mem0, Qdrant, Ollama,
  retrieval, and answer generation actually ran live, were mocked, or were
  skipped;
- random seed (live evaluator currently fixes Python `random` to 0), run
  status, preflight status, run start/end if recorded, and exact command;
- isolated SQLite, Mem0/Qdrant, Mem0 configuration, progress, and result
  paths; verify each resolves under a new unique isolation root;
- per-case completed/failed/skipped/interrupted status and the latency boundary.

Do not record credentials or environment-variable values. Progress JSONL is
allowlisted/redacted; the final result JSON intentionally contains predictions
and evidence and is sensitive. Keep both, console logs, and stores inside the
fresh run root. Never reuse a storage directory or overwrite a result. Preserve
all historical datasets, results, and partial logs unchanged.

### Deterministic evaluation command

Use a new run directory and output file. `research_evaluator.py` applies the
benchmark's storage checks before importing storage-aware modules:

```powershell
$run = Join-Path (Get-Location) ('.tmp\research-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $run | Out-Null
$output = Join-Path $run 'results\phase5_3_deterministic.json'
New-Item -ItemType Directory -Path (Split-Path $output) | Out-Null
.\.venv\Scripts\python.exe .\research_evaluator.py `
  --isolation-dir "$run" `
  --dataset .\evaluation\research_v1.jsonl `
  --output "$output"
$evalExit = $LASTEXITCODE
Write-Output "EVALUATOR_EXIT=$evalExit"
```

### Isolated live evaluation command

Only run after the protocol is approved, preflight succeeds, Ollama already
has the configured chat and embedding models, and the owner accepts the
runtime cost. This command is documented for reproducibility; it was **not**
run while preparing this protocol.

```powershell
$run = Join-Path (Get-Location) ('.tmp\research-live-' + [guid]::NewGuid().ToString('N'))
$output = Join-Path $run 'results\phase5_3_live.json'
.\.venv\Scripts\python.exe .\research_live_evaluator.py `
  --isolation-dir "$run" `
  --dataset .\evaluation\live_research_v1.jsonl `
  --case-ids direct-fact updated-current-goal historical-predecessor `
    asof-before-replacement asof-at-replacement asof-after-replacement `
    explicit-cause succession-not-cause ambiguous-preference `
    unsupported-question foreign-user-isolation retrieval-failure-mock-only `
  --output "$output"
$evalExit = $LASTEXITCODE
Write-Output "LIVE_EVALUATOR_EXIT=$evalExit"
```

The mock-only diagnostic is expected to be skipped in live mode. Confirm the
actual case selection and statuses in the output; never count skipped or
failed rows as successful answers.

### Deterministic pytest workaround for this Windows environment

The project has encountered `tmp_path` directory creation/cleanup permission
failures under pytest's normal tmpdir plugin. The established workaround is a
fresh project-local UUID directory per test, created with `Path.mkdir()`,
isolated storage environment variables set before pytest starts, and pytest's
cache/tmpdir plugins disabled. The following is the validated pattern; use a
fresh `$testRoot` every run:

```powershell
$testRoot = Join-Path (Get-Location) ('.tmp\pytest-' + [guid]::NewGuid().ToString('N'))
$perTest = Join-Path $testRoot 'per-test'
$base = Join-Path $testRoot 'pytest-base'
New-Item -ItemType Directory -Path $perTest | Out-Null
$plugin = Join-Path $testRoot 'causalmem_pytest_tmp.py'
@'
import os, uuid
from pathlib import Path
import pytest
@pytest.fixture
def tmp_path():
    path = Path(os.environ['CAUSALMEM_TEST_TMP_ROOT']) / ('case-' + uuid.uuid4().hex)
    path.mkdir(parents=False, exist_ok=False)
    return path
'@ | Set-Content -Encoding utf8 $plugin
$env:CAUSALMEM_TEST_TMP_ROOT = $perTest
$env:CAUSALMEM_DB = Join-Path $testRoot 'storage\causalmem.sqlite3'
$env:MEM0_PATH = Join-Path $testRoot 'storage\mem0'
$env:MEM0_DIR = Join-Path $testRoot 'storage\mem0-config'
if ($env:PYTHONPATH) { $env:PYTHONPATH = "$testRoot;$env:PYTHONPATH" }
else { $env:PYTHONPATH = $testRoot }
.\.venv\Scripts\python.exe -m pytest -q tests `
  -p no:cacheprovider -p no:tmpdir -p causalmem_pytest_tmp `
  --basetemp="$base"
$testExit = $LASTEXITCODE
Write-Output "PYTEST_EXIT=$testExit"
```

Run focused tests before the full suite, but report only actually executed
commands and exit codes. Do not edit application code solely to work around a
temporary-directory failure. Classify failures as fixture/setup, filesystem,
assertion, or application errors from their tracebacks.

## 5. Analysis and stopping rules

1. Validate dataset schema, unique case IDs, annotations, and selected order
   before execution. Do not tune the system or alter labels after observing
   results in the same evaluation cycle.
2. Run deterministic tests/evaluator first. Any assertion failure blocks live
   execution until understood; an infrastructure skip is not a pass.
3. Run isolated live cases only after storage-path and service preflight checks
   pass. If a live case fails or the run is interrupted, preserve partial
   progress and stop; do not translate the failure into a valid abstention or
   reuse its storage directory for a retry.
4. Review each completed case's expected behavior, returned state/transition
   evidence, citation group, text-free diagnostics, forbidden-fact proxy, and
   latency boundary. Do not expose sensitive final results in general logs.
5. Treat the `EXPLICIT_CAUSE` label as recorded user attribution, not scientific
   causal proof. Chronology or `TEMPORAL_SUCCESSION` alone cannot establish a
   cause.

## 6. Known limitations and threats to validity

- The live dataset is small and synthetic; it does not establish
  generalization, user utility, or broad answer reliability.
- Exact fixture-key and citation-group metrics depend on curated mappings;
  substring fact checks do not measure semantic entailment.
- The current six-case results do not provide a paired live Mem0-only
  comparison. No superiority claim is supported.
- The previous six-case live run had an intermittent temporal lexical-support
  abstention before a later all-pass run. Recurrence remains possible.
- Real model outputs are nondeterministic even though the evaluator seed is
  fixed; the seed does not make Ollama inference deterministic.
- The 4.28-hour ingestion anomaly remains unexplained. Single-run timings do
  not establish its cause or a stable latency distribution.
- SQLite transactions do not include Mem0 writes; cross-store partial state is
  possible.
- User IDs are namespaces, not authentication or authorization.
- The current implementation uses recorded-time intervals, not a full
  event-time/bitemporal model.
- Dependency requirements are lower bounds, not a lockfile; Python
  compatibility is not formally pinned.

### Separate unresolved state-provenance concern

A prior live-data observation reported the `running` state as `ACTIVE` and
including the newer switch message among its source-memory IDs, despite a
recorded transition to `swimming`. This is not resolved by the successful
manual answer verification or by the six-case evaluation. Do not repair or
rewrite stored records as part of this protocol. Investigate separately with
a read-only, redacted structural audit of the row, all state write/close and
provenance-append paths, transition endpoints, and timestamps. Then reproduce
the intended lifecycle on a fresh temporary database. Determine whether the
record is stale historical data, another ingestion path, or a genuine
consistency defect before proposing a migration or correction.

## 7. Protocol readiness checklist

- [ ] Decide whether the evaluation uses the existing version-1.0 datasets
  unchanged or a separately reviewed new dataset version.
- [ ] Resolve the `running` status/provenance inconsistency through read-only
  inspection and isolated reproduction; preserve production data meanwhile.
- [ ] Confirm the current code revision and installed package versions; Git
  metadata is unavailable in the recorded environment, so a commit identifier
  must not be fabricated.
- [ ] Run the deterministic targeted tests and full suite using isolated
  storage; archive counts, warning summary, command, and exit code.
- [ ] Review privacy and storage isolation before any live run.
- [ ] Obtain approval for a new live run; use a fresh isolation root and
  versioned output; do not overwrite historical results.
- [ ] Report failures, skips, unavailable metrics, and incomplete runs without
  imputing values or changing ground truth.

**Next step:** perform the separate read-only state-provenance investigation,
then obtain approval for the deterministic Phase 5.3 run. This document does
not authorize or execute that run.
