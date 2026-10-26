# CausalMem evaluation fixture

The planned Phase 5.3 evaluation protocol, case matrix, metric definitions,
isolation requirements, and Windows test procedure are documented in
[PHASE_5_3_PROTOCOL.md](PHASE_5_3_PROTOCOL.md). It is a plan, not an execution
record.

`ground_truth.jsonl` is a manually authored synthetic dataset. Each line is an
independent case. `fixture_key` values identify source utterances and are the
only identifiers used as annotated evidence; runtime Mem0 IDs are mapped to
these keys after ingestion and are never ground-truth labels.
The benchmark selects this checked-in file by default when it exists; pass
`--dataset` to select another read-only set. `--generate` explicitly requests
the older synthetic fixture generator inside the isolated output directory.

The benchmark ingests `conversation` and `distractor_users` as user messages.
Expected-answer annotations (`expected_evidence_keys`, answer facts, temporal
expectations, and causal expectations) are attached to saved result rows only
after retrieval and generation. They are not passed to either model condition.

## Metric definitions

- Evidence precision/recall/F1 compare retrieved stable fixture keys with
  `expected_evidence_keys`. If both sets are empty, precision and F1 are 1 and
  recall is not applicable. If no evidence is retrieved despite expected keys,
  precision is not applicable and recall/F1 are 0. Cases with no expected keys
  are counted separately.
- Answer behavior compares exact project abstention text with
  `expected_behavior`. Coverage is the fraction of should-answer cases that
  received a non-abstaining answer. Fact-based incorrectness is reported
  separately from answer/abstain behavior.
- Required/forbidden facts use normalized, case-insensitive substring checks.
  This is a transparent lexical proxy, not semantic entailment; manual review is
  required for claims where correctness matters.
- Causal language is detected with a small explicit-marker pattern. Hybrid
  evidence eligibility uses the current deterministic causal-answer gate.
  Mem0-only eligibility uses an explicit-cause phrase plus query/evidence term
  overlap, which is a weaker proxy. The support method is recorded per result.
  Implicit causal claims and bare causal nouns can be missed, so causal results
  require manual review.
- Temporal current/history checks use returned state evidence and fixture-key
  mappings. As-of consistency applies aware ISO timestamps and the existing
  half-open recorded-time rule: `valid_from <= requested < valid_until`, with
  null `valid_until` open-ended and unknown/malformed `valid_from` excluded.
  `hybrid_retriever.retrieve` accepts an optional as-of timestamp and the six
  cases in `temporal_end_to_end.jsonl` score state keys returned by that actual
  retrieval path. The five `temporal_oracle` boundary cases remain oracle-only
  interval checks and are excluded from end-to-end temporal metrics. Persisted
  timestamps represent recording time, not parsed event time.
- Latency reports mean and median for available samples. P95 is reported only
  with at least 20 observations. Retrieval timing includes query classification
  for the hybrid condition; generation timing covers answer generation. Missing
  timing samples are omitted and counted rather than replaced.

The Mem0-only and hybrid conditions use the same case identity, conversation,
question, and annotations. The hybrid condition can additionally retrieve
SQLite states and explicit transitions. This dataset and evaluator do not by
themselves establish that either condition is better. Phase 5.1's deterministic
results are policy fixtures, not live benchmark results.

## Deterministic Phase 5.1 policy evaluation

`research_v1.jsonl` is a separate, versioned ten-case fixture set. It supplies
fixed evidence bundles and fixed structured model responses, then runs those
responses through the current answer-validation policy. It covers direct and
updated facts, historical state evidence, explicit cause, succession without
cause, ambiguity, unsupported questions, foreign-user evidence, multiple
citations, and a mocked SQLite retrieval failure. The fixed responses are test
inputs, not model predictions; this evaluation does not measure LLM quality,
retrieval ranking quality, live Mem0 persistence, or latency.

Run it with new paths for each invocation; the CLI uses the benchmark's
fail-closed storage isolation checks before importing application storage code:

```powershell
$run = Join-Path (Get-Location) ('.tmp\research-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe research_evaluator.py `
  --isolation-dir "$run" `
  --output "$run\results\phase5_1.json"
```

The JSON result records `schema_version`, `dataset_version`, predictions, and
metrics with numerators/denominators. Answerability and abstention metrics use
the expected answer/abstain labels. Source-ID precision/recall measures exact
identifier agreement; claim citation support compares each rendered citation
group against that claim's expected source group. Temporal scores compare cited state IDs and
verify recorded validity at the annotated timestamp where one is supplied.
Causal scores compare cited transition IDs with explicit labels. These ID and
lexical proxies do not establish semantic correctness. Null metrics indicate a
zero denominator; an empty fixture run receives no fabricated score.

## Phase 5.2 live-pipeline evaluation

`live_research_v1.jsonl` is a separate, versioned set of synthetic
conversations. The live runner seeds runnable cases through the existing Mem0
add and `state_extractor.ingest_message` paths, then invokes the actual full
query classifier, `hybrid_retriever.retrieve`, and answer generator. Expected
labels are copied to the result only after each case finishes; they are not
sent to the model. The mock-only retrieval-failure diagnostic is recorded as
skipped and is never counted as a live result.

Run it in a new directory for every attempt, after confirming that the local
Ollama service already has both the configured chat model and
`nomic-embed-text`. The runner checks installed models and never downloads one.
For PowerShell:

```powershell
$run = Join-Path (Get-Location) ('.tmp\research-live-' + [guid]::NewGuid().ToString('N'))
$out = Join-Path $run 'results\live_research_results.json'
.\.venv\Scripts\python.exe .\research_live_evaluator.py --isolation-dir "$run" --dataset .\evaluation\live_research_v1.jsonl --output "$out"
```

The runner establishes isolated SQLite, Mem0/Qdrant, and Mem0 configuration
paths before importing application storage modules. It refuses reused stores
and output files. The JSON result is versioned as `live_pipeline`, records
completed/failed/skipped cases separately, and includes an append-only progress
JSONL file beside the result. If setup or a case fails, failures are not scored
as model abstentions. An interrupted process may leave only the progress file;
a hard process termination cannot be converted into a final summary.

The live report reuses the benchmark's answerability, evidence-key, causal,
temporal, and latency calculations. Evidence and citation agreement are exact
fixture-key/ID mapping proxies, not semantic entailment. Latency covers each
full retrieval (including query classification) plus answer generation; it
excludes fixture ingestion, preflight, and file writing. P95 remains undefined
below the evaluator's 20-observation threshold. Mock-only source-failure tests
remain separate from live metrics. The six-case end-to-end temporal fixture in
`temporal_end_to_end.jsonl` continues to be a deterministic test and is not
mixed into this live result.

## Phase 5.2.1 status

Claim validation remains fail-closed. Its diagnostics now include text-free
rejection details identifying malformed claim fields and evidence-reference
shapes; they do not retain claim text, prompts, or raw model responses.

As-of retrieval passes the normalized requested timestamp and state-validity
context into answer generation. Mem0 records linked to SQLite states are
filtered using those states' validity intervals; unlinked records continue to
use their recorded `created_at` timestamp. State validity follows the
half-open rule `valid_from <= requested_timestamp < valid_until`, with a null
`valid_until` representing an open interval. At the exact replacement
boundary, the predecessor state and its linked memory are excluded, and the
replacement state and its linked memory are included.

The full pytest suite passed with 156 passed, 0 failed, and 0 skipped. Eight
existing SQLite datetime-adapter deprecation warnings were reported. The
Phase 5.2.1 live evaluation was interrupted; only partial progress results
are available. Live answer and retrieval quality therefore remain
unverified. The original Phase 5.2 baseline and partial progress logs are
preserved and are not replaced by these status notes.

## Phase 5.2 evaluation diagnostics follow-up

Temporal claim-validation diagnostics now separate lexical support from the
selected state's type/value, its explicitly linked same-user source memories,
and their combined text. They also indicate when both sources were needed or
neither source supported the claim. These are booleans only; they do not record
claim text, terms, evidence text, prompts, model output, or evidence IDs. The
existing combined lexical check remains the acceptance check, so this change
does not relax validation.

The `historical-predecessor` annotation now distinguishes the expected
historical state evidence from the replacement transition's provenance. Its
`expected_transition_provenance_keys` are included in the exact evidence target
only for explicitly annotated cases and are also scored as a separate exact
fixture-key metric. Historical state scoring still checks the expected state
key independently. Unannotated extra evidence continues to count as a false
positive.

The saved `explicit-cause` result contains one accepted claim citing the
transition while covering the case's two required fact proxies. Its annotation
therefore expects one citation group for that single claim. Citation-group
scoring is unchanged; this is a case-specific annotation correction, not a
change to causal validation or a general many-to-one scoring rule.

The latest deterministic validation also covers clause-local negation in
lexical support. A positive current preference is no longer rejected merely
because a different, superseded preference appears in a negated clause of the
same source text. Claim support remains fail-closed: claim terms must be
supported within a matching clause with compatible polarity. Regression tests
cover the current preference, the superseded preference, and unrelated
unsupported claims.

The live evaluator now accepts `--case-ids` to select cases from the unchanged
dataset without rewriting it. The selector preserves dataset order and rejects
duplicate or unknown IDs. This is an execution control only; it does not alter
the cases or scoring rules. The full deterministic suite after these changes
completed with **194 passed, 0 failed, 0 skipped**, exit code **0**, and the
same 8 SQLite datetime-adapter deprecation warnings. On this Windows
environment, pytest's normal `tmp_path` cleanup produced `PermissionError`
(`WinError 5`); the successful run disabled pytest's cache and tmpdir plugins
and supplied an isolated project-local `tmp_path` fixture and storage paths.

These deterministic tests validate policy and data-flow behavior with mocks;
they do not establish live answer quality. A separate six-case live run is
required for that limited observation, and even a perfect proxy score on six
synthetic cases is not evidence of general semantic correctness.

Focused regression validation for this follow-up completed with **14 passed,
113 deselected, 0 failed**, exit code **0**, and no warnings. The run used
pytest with `-p no:cacheprovider -p no:tmpdir` and a unique project-local
`--basetemp`; `CAUSALMEM_DB`, `MEM0_PATH`, and `MEM0_DIR` were redirected to a
new project-local test root. The focused command used a fresh project-local root (the completed run used
`.tmp/phase52-evaldiag-tests-ccb181263dee46eaa621a1df2c339f00`):

```powershell
$root = Join-Path (Get-Location) ('.tmp\phase52-evaldiag-tests-' + [guid]::NewGuid().ToString('N'))
$base = Join-Path $root 'pytest'
New-Item -ItemType Directory -Path $root | Out-Null
$env:CAUSALMEM_DB = Join-Path $root 'storage\test.sqlite3'
$env:MEM0_PATH = Join-Path $root 'storage\mem0'
$env:MEM0_DIR = Join-Path $root 'storage\mem0-config'
.\.venv\Scripts\python.exe -m pytest -q tests/test_hybrid_answers.py tests/test_evaluator.py tests/test_research_live_evaluator.py tests/test_benchmark_execution.py -k "temporal_validation_diagnostics or temporal_lexical_diagnostics_split or foreign_linked_memory or transition_provenance or ground_truth_jsonl or explicit_cause_annotation or benchmark_dataset_validates" -p no:cacheprovider -p no:tmpdir --basetemp="$base"
$testExit = $LASTEXITCODE
```

Observed result: 14 passed, 113 deselected, exit code 0, no warnings. At that
point no live evaluation had been run; the subsequent six-case run is recorded
below. Prior saved results were not changed.

### Six-case Phase 5.2.2 validation before provenance-scoring correction

The selected live run used the unchanged `evaluation/live_research_v1.jsonl`
dataset and the `--case-ids` selector. It completed all six cases with the
real isolated SQLite, Mem0/Qdrant, local Ollama, hybrid retrieval, and answer
generation components. The run exited with code **0** and wrote results and
progress under:

`.tmp/phase5222-quality-784641255beb4d54a4e1ae24f968b51d/`

The declared SQLite, Qdrant/Mem0, Mem0 configuration, result, and progress
paths all resolved under that new root. The evaluator reported 6 attempted, 6
completed, 0 failed, and 0 skipped. Five cases produced an answer and passed
their required-fact substring checks; `asof-after-replacement` abstained.

Measured proxies were: evidence precision/recall/F1 **1.00/1.00/1.00** (8
expected fixture keys, no false positives or false negatives); answer-or-
abstain accuracy **5/6 (83.3%)**; required-fact recall **6/7 (85.7%)**;
forbidden-fact violations **0/4**; temporal evidence precision/recall/F1
**1.00/1.00/1.00** across the three end-to-end as-of cases; causal evidence
precision/recall/F1 **1.00/1.00/1.00** for the single causal case; and exact
citation-group agreement **5/6 (83.3%)**. Mean query-pipeline latency was
**26.89 seconds** (retrieval **5.13 seconds**, answer generation **21.75
seconds**); p95 is unavailable because this run has six observations and the
metric requires at least 20. Required/forbidden answer facts and citation
groups are lexical/fixture-key proxies, not semantic entailment.

For the one abstention, privacy-safe diagnostics confirm that requested
timestamp normalization, owned state resolution, stored interval timestamp
normalization, and the half-open validity check passed. Lexical support failed
for the selected state's label/value and its linked same-user source memory;
the stable rejection was `temporal_lexical_support_failed`. The diagnostics do
not retain the proposed claim or unmatched terms, so this run cannot establish
whether the model proposed unsupported wording or a supported paraphrase that
the lexical validator rejected. No validation rule or annotation was changed
in response to this result. This run was not all-pass.

The run also emitted Mem0 optional spaCy/FastEmbed-unavailable notices and a
PostHog client/network warning. These did not stop evaluation; the live
components completed. This six-case synthetic run is a narrow observation,
not evidence of general reliability or semantic correctness.

### Final six-case validation after provenance-scoring correction

The benchmark provenance scorer now verifies both transition endpoint owners
from the isolated database's user-filtered state history. It no longer
requires the successor state to be included in answer evidence for a
predecessor query. Transition ownership and source-memory user checks remain
in place. A focused regression verifies that an omitted endpoint is credited
only when its database owner matches the requested user; a foreign owner is
still rejected.

The subsequent fresh run used the same six selected case IDs and exited **0**.
All 6 cases completed, with no failed or skipped cases. All six produced an
answer and passed the required-fact proxy. Evidence retrieval scored
precision/recall/F1 of **1.00** (8 true-positive fixture keys, 0 false
positives, 0 false negatives). Citation-group agreement was **6/6**. The
three end-to-end as-of states matched exactly (**3/3**), and the single
annotated transition-provenance case matched (**1/1**). The causal evidence
proxy was **1/1**. Answer-or-abstain accuracy and required-fact recall proxies
were **6/6** and **7/7** respectively; forbidden-fact violations were
**0/4**. These are lexical and fixture-key proxies, not semantic entailment.

Mean query-pipeline latency, excluding fixture ingestion, was **26.90
seconds** (retrieval **5.38 seconds**, answer generation **21.52 seconds**).
P95 is unavailable with six observations (the evaluator requires at least
20). The explicit-cause fixture ingestion took about **4.28 hours** on the
observed progress clock despite normal query latency. This extreme Mem0 write
delay is an operational concern; no timeout or retry policy was changed.
Optional spaCy/FastEmbed notices and PostHog client/network warnings appeared,
but all six evaluations completed.

The final run directory was
`.tmp/phase5222-quality-final-6ae7c0522e5d458d920ac5587c31d218/`. Its SQLite,
Qdrant/Mem0, Mem0 configuration, result, and progress paths were verified to
resolve beneath that root. Prior runs and the source dataset were not used as
output targets. This six-case synthetic run does not establish general
reliability or semantic correctness.

## Evidence and diagnostics recorded on 2026-10-24

The entries below preserve historical observations at their original scope.
They were not revalidated as part of this documentation update.

### Deterministic policy tests

The fixed-fixture evaluation in `research_v1.jsonl` and the automated tests
exercise deterministic validation, ownership, temporal intervals, explicit
causal evidence, abstention, retrieval error handling, and selected ingestion
flows using fixtures and mocks. They are policy/data-flow checks; they do not
measure live LLM answer quality, live Mem0 ranking quality, or production
latency. SQLite seeded directly by a test is not equivalent to end-to-end
conversation ingestion.

### Six-case live quality evaluation

The historical run at
`.tmp/phase5222-quality-final-6ae7c0522e5d458d920ac5587c31d218/` completed all
six selected synthetic cases. It reported 6/6 cases passing the configured
answer and proxy checks, with evidence precision/recall/F1 of 1.00, answer-
or-abstain accuracy 6/6, required-fact recall proxy 7/7, exact citation-group
agreement 6/6, exact as-of state matches 3/3, and the annotated transition
provenance and causal evidence proxies each 1/1. These are small-sample
fixture-key and lexical proxies, not semantic entailment or proof of general
correctness. The recorded mean query latency was 26.90 seconds; p95 was
unavailable because only six observations were collected.

An earlier six-case run at
`.tmp/phase5222-quality-784641255beb4d54a4e1ae24f968b51d/` had one abstention
(`asof-after-replacement`) after temporal lexical support failed. The later
all-pass result does not erase that intermittent failure or establish that it
cannot recur. The dataset and evaluator are synthetic research instruments;
the results should be interpreted as observations on these cases only.

### Single-case latency diagnostic

The separate isolated `explicit-cause` diagnostic at
`.tmp/phase521-explicit-cause-diag-cfdd9316895d496b8fff19447c406d84/` completed
one case in **215.384 seconds**. Its recorded Mem0 LLM memory-extraction stages
took **50.981 seconds** and **50.978 seconds**; answer generation took
**45.703 seconds**. Embedding generation, vector search, and vector insertion
were substantially shorter in that run. The saved Ollama runtime observation
reported the chat model and embedder executing on CPU. These timings locate
expensive stages in this one run but do not prove why inference was slow, nor
should they be interpreted as quality scores. This is a performance
diagnostic, separate from the six-case quality evaluation.

### Privacy, safety, and artifact handling

The live evaluator's progress JSONL is serialized through an allowlist. It
retains compact case status, safe timing/count fields, and stable reason or
error-category codes while excluding questions, answers, prompts, raw model
responses, evidence text/IDs, and user identifiers. This is a property of the
progress JSONL writer, not a guarantee that every console log or application
surface is content-free.

The final evaluation JSON intentionally contains predictions and evidence and
must be treated as sensitive. Keep results, progress files, captured console
logs, and isolated SQLite/Mem0/Qdrant stores inside a fresh run directory; do
not commit or share them without inspecting their contents. User IDs are
storage namespaces, not authentication credentials or an access-control
system.

Answer safety is not provided by the model prompt alone. Deterministic claim
validation, evidence ownership checks, temporal interval checks, and the
explicit-cause evidence gate supplement model-generated answers. They can
reject unsupported or ambiguous claims, but the lexical and fixture-key
evaluation proxies do not prove semantic correctness.

The local Mem0 configuration uses Mem0 OSS with an Ollama LLM and
`nomic-embed-text` embedder backed by local Qdrant storage; SQLite separately
stores typed states and transitions. Mem0 ingestion and SQLite state ingestion
are sequential systems without a shared atomic transaction. State intervals
are half-open on recorded time (`valid_from` inclusive, `valid_until`
exclusive); recorded timestamps are not necessarily real-world event times.
An explicit causal link requires recorded cause evidence. Temporal succession
alone is ordering, not causation. Ambiguous or unsupported requests may be
rejected or answered with the safe abstention behavior.

`docs/CODEBASE_AUDIT.md` and `docs/PHASE_7A_AUDIT.md` are historical,
point-in-time audits. They may describe earlier code findings and should not
be treated as authoritative descriptions of subsequent changes. The audit
reports and the measurements above were not rerun or revised for this note.

### Reproduction and limitations

For a new live run, use the isolated command in the Phase 5.2 section, a new
unique `--isolation-dir`, and a new output path under that root. The evaluator
refuses existing storage targets and output files. Confirm the required local
Ollama models are already available; the evaluator does not download them.
Keep the original dataset read-only. For a selected subset, pass `--case-ids`
with the requested IDs. A result file is an output artifact and must never be
reused as the next run's output target.

The recorded local CPU diagnostic is a single observation; no controlled
hardware/model/context comparison isolated the cause of its latency. The
six-case result is too small to establish general reliability, and its
substring, fixture-key, and citation-group scores are proxies. Live extraction
and answer quality, broader user populations, and independently adjudicated
semantic correctness remain incompletely evaluated.

## Reproducible local environment (Windows)

### Version scope

The repository has no declared `requires-python` range or tested-version
matrix. The versions below were observed in the project's local virtual
environment on 2026-10-24; they are an environment snapshot, not dependency
pins or a guarantee that another installation behaves identically. The
`requirements.txt` values are minimums:

| Component | `requirements.txt` minimum | Observed local version |
|---|---:|---:|
| Python | Not declared | 3.12.10 |
| mem0ai | 2.2.1 | 2.2.1 |
| qdrant-client | Not directly listed | 1.19.1 |
| ollama Python package | 0.4 | 0.6.3 |
| openai | 1.30 | 3.24.0 |
| streamlit | 1.37 | 1.65.0 |
| networkx | 3.3 | 3.7 |
| pydantic | 2.7 | 2.13.5 |
| python-dotenv | 1.0 | 1.2.4 |
| plotly | 5.22 | 7.1.0 |
| pytest | 8.0 | 9.1.1 |

The Ollama CLI observed locally was 0.40.9. `ollama list` showed
`llama3.2:3b` and `nomic-embed-text:latest`; model tags can move as local
models are updated, so this is a point-in-time observation rather than a
lock. The recorded app diagnostic used `llama3.2:3b`; `mem0_engine.py`
configures the Ollama embedding provider with `nomic-embed-text`.

### Install and configure Ollama

Install Ollama using its [official Windows download and PowerShell
instructions](https://ollama.com/download/windows). The official installer
command is:

```powershell
irm https://ollama.com/install.ps1 | iex
```

Open the Ollama app to start its local service. If the service is not already
running, start `ollama serve` in a separate terminal. Then pull the model tags
used by this project's recorded local setup and verify them:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama list
```

In `.env`, select `LLM_PROVIDER=ollama`, set `LLM_MODEL=llama3.2:3b`, and set
`OLLAMA_BASE_URL=http://localhost:11434` for the default local service. The
embedding model is configured in `mem0_engine.py`, not selected by an
environment variable. `.env.example` shows these values as a commented local
configuration while retaining the application's OpenAI default; copying the
sample does not automatically switch providers.

### One-case isolated live evaluation

Run only after the service and both models are available. The evaluator's
required arguments are `--isolation-dir` and `--output`; it also accepts
`--dataset` and `--case-ids`. It configures `CAUSALMEM_DB`, `MEM0_PATH`, and
`MEM0_DIR` within the isolation root before importing storage modules. The
Ollama Qdrant directory is the configured Mem0 path's `-ollama` sibling;
progress is written beside the result as `.progress.jsonl`. The console log
below is also captured under the root. The final JSON contains predictions
and evidence and is sensitive.

```powershell
$run = Join-Path (Get-Location) ('.tmp\research-live-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $run | Out-Null
$out = Join-Path $run 'results\one_case.json'
$console = Join-Path $run 'console.log'
& .\.venv\Scripts\python.exe .\research_live_evaluator.py `
  --isolation-dir "$run" `
  --dataset .\evaluation\live_research_v1.jsonl `
  --case-ids explicit-cause `
  --output "$out" *> "$console"
$evalExit = $LASTEXITCODE
Write-Output "Evaluation exit code: $evalExit"
```

Every run needs a new root and result path. The evaluator refuses reused
storage targets and outputs; do not point it at existing results. SQLite,
Mem0/Qdrant, Mem0 configuration, progress JSONL, console log, and result file
are then under the new root. Keep that root private and inspect artifacts
before sharing. A successful process exit means the evaluation completed; it
does not establish that a model answer was semantically correct. The evaluator
preflight checks local models and does not download missing ones.

### Focused tests with isolated Windows temporary directories

This workspace has experienced pytest temporary-directory cleanup
`PermissionError` failures. The workaround below supplies isolated storage,
disables pytest's cache and built-in tmpdir plugins, and substitutes a minimal
per-test `tmp_path` fixture rooted under a new project-local test directory.
It avoids deleting or reusing existing temporary folders. This is an
environment workaround, not a project pytest configuration change.

```powershell
$testRoot = Join-Path (Get-Location) ('.tmp\pytest-' + [guid]::NewGuid().ToString('N'))
$perTest = Join-Path $testRoot 'per-test'
$base = Join-Path $testRoot 'pytest-base'
New-Item -ItemType Directory -Path $perTest -Force | Out-Null
$plugin = Join-Path $testRoot 'causalmem_pytest_tmp.py'
@'
import os
import tempfile
from pathlib import Path
import pytest

@pytest.fixture
def tmp_path():
    root = Path(os.environ['CAUSALMEM_TEST_TMP_ROOT'])
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='case-', dir=root))
'@ | Set-Content -Encoding utf8 $plugin
$env:CAUSALMEM_TEST_TMP_ROOT = $perTest
$env:CAUSALMEM_DB = Join-Path $testRoot 'storage\causalmem.sqlite3'
$env:MEM0_PATH = Join-Path $testRoot 'storage\mem0'
$env:MEM0_DIR = Join-Path $testRoot 'storage\mem0-config'
if ($env:PYTHONPATH) { $env:PYTHONPATH = "$testRoot;$env:PYTHONPATH" }
else { $env:PYTHONPATH = $testRoot }
& .\.venv\Scripts\python.exe -m pytest -q `
  tests\test_core.py tests\test_hybrid_answers.py `
  -p no:cacheprovider -p no:tmpdir -p causalmem_pytest_tmp `
  --basetemp="$base"
$testExit = $LASTEXITCODE
Write-Output "Pytest exit code: $testExit"
```

Use a fresh PowerShell session or restore the environment variables after the
run. The workaround intentionally leaves its isolated files in place for
inspection; do not reuse its root for another run.

### Troubleshooting

- **Ollama unavailable:** start the Ollama app, or run `ollama serve` if no
  service is active; check that `OLLAMA_BASE_URL` matches the running local
  endpoint (default `http://localhost:11434`).
- **Model missing:** run `ollama list`; pull `llama3.2:3b` and
  `nomic-embed-text` if absent, then retry. The evaluator does not pull models.
- **Slow CPU inference:** the recorded one-case run took 215.384 seconds in
  total, including two Mem0 LLM extraction stages of 50.981 and 50.978 seconds
  and answer generation of 45.703 seconds. Ollama reported CPU execution for
  that run. These measurements do not isolate the cause, and other hardware,
  models, or workloads may differ.
- **Windows pytest cleanup error:** use the fresh project-local workaround
  above and preserve the failing traceback. Do not delete old temp directories
  or change permissions to work around it.
- **Evaluation completed but answer appears wrong:** exit code 0 indicates
  completion, not semantic correctness. Review the sensitive result artifact
  manually and interpret lexical/fixture-key metrics only within their stated
  scope.
