# Phase 7A: Reproducibility and Release-Readiness Audit

**Audit date:** 2026-10-23  
**Scope:** Read-only source, documentation, dependency, and storage-path review. This report is the only requested artifact created for this audit. No application code, dependencies, ground truth, databases, Mem0 stores, or model artifacts were changed. No live benchmark or full test suite was run during this audit.

## Summary

The project has useful safeguards: the benchmark establishes storage overrides before importing storage modules, refuses existing benchmark stores and result files, validates the local Qdrant path, and keeps outputs under an isolation root. SQLite state and transition writes share a transaction; transition endpoints are checked for same-user ownership; temporal and graph queries are user-scoped; and causal answers require recorded explicit-cause evidence.

The project is **not release-ready for general or hosted multi-user use**. The main blockers are stale benchmark instructions, import-time initialization of the configured SQLite database, no dependency lock or declared tested Python matrix, and several limits around causal answer verification, Mem0/SQLite consistency, attribution, and timestamp precision. These findings do not establish that production data was modified. There is no Git repository metadata in this workspace, and no before/after production-store snapshot was made.

## Storage and execution paths

| Store or output | Current path behavior | Evidence |
|---|---|---|
| Application SQLite | `CAUSALMEM_DB`; default `data/causalmem.sqlite3` | `config.py`, `database.py` |
| Mem0 configuration | `MEM0_DIR`; defaults to `mem0-config` beside `MEM0_PATH` | `config.py` |
| Mem0/Qdrant | `MEM0_PATH`; local Qdrant. For Ollama, `mem0_engine.py` uses the sibling `<MEM0_PATH-name>-ollama` and an Ollama-specific collection | `mem0_engine.py` |
| Generated legacy fixture | `dataset_generator.py` defaults to `data/evaluation_questions.json`; opens with exclusive-create mode and refuses overwrite | `dataset_generator.py` |
| Benchmark database and stores | Under `<isolation-dir>/storage/`: SQLite, Mem0, Ollama-Qdrant sibling, and Mem0 config paths | `benchmark.py::_configure_isolation`, `_load_runtime` |
| Benchmark results and generated dataset | Results/output must be inside the isolation root and separate from stores; an existing results file is refused | `benchmark.py::_resolve_evaluation_paths` |
| Ollama model files | Owned by the separate Ollama installation/service, not by CausalMem’s configured Qdrant path | `config.py`, `mem0_engine.py`; Ollama is called through its service URL |
| Python and pytest artifacts | `__pycache__`/`.pyc`, `.pytest_cache`, and observed pytest temp artifacts. `.gitignore` ignores Python bytecode and `.pytest_cache`, but not `.tmp` or `pytest-cache-files-*` | `.gitignore`, test-run environment |

The benchmark’s checks cover its declared SQLite, Qdrant, and Mem0 configuration stores. Its isolation root itself may already exist if the specific store targets do not; output folders may also pre-exist if the result file does not. Unique, new run roots remain the safest reproducible convention. The benchmark does not download Ollama models.

## Confirmed strengths

- `benchmark.py::_configure_isolation` rejects a run if storage modules were already imported, checks paths against `.env`/environment-configured application stores, refuses existing store targets, then sets storage environment variables before runtime imports. `_load_runtime` rechecks SQLite, Mem0, Qdrant, and `MEM0_DIR` paths and fails closed if the configured local Qdrant backend cannot be verified.
- `benchmark.py::_resolve_evaluation_paths` confines result output to the run root, separates outputs from stores, and refuses to overwrite an existing result file or generated dataset.
- `database.py::connect` commits on successful exit and rolls back on exceptions. `state_extractor.py::ingest_message` passes one connection to the state, transition, and source-closure writes. `Database.save_transition` validates that both endpoints exist and belong to the transition user.
- `Database.states`, `states_at`, and `transitions` filter by user. `causal_graph.py` builds graphs from user-filtered state and transition reads. `mem0_engine.py` passes `filters={"user_id": user_id}` to Mem0 search and get-all calls.
- As-of retrieval accepts an optional timezone-aware timestamp, normalizes it to UTC, filters state validity with a half-open interval, and filters future transition endpoints/timestamps. Active-state retrieval remains the no-timestamp path.
- The causal answer gate in `answer_generator.py` requires a query subject anchor, same-user endpoint labels, an `EXPLICIT_CAUSE` relation, and nonempty recorded trigger/reason evidence. Other relationship types do not satisfy that gate.
- Current tests cover extraction guards, succession versus explicit cause, provenance, transition endpoint ownership, transaction rollback, temporal intervals, user isolation, causal abstention, evaluator proxies, benchmark storage checks, and the Phase 6E mocked ingestion-to-retrieval cases.

## Findings

### High

#### H1 — Benchmark and temporal evaluation documentation is stale

**Evidence:** `README.md:63-66` documents a benchmark command with no `--isolation-dir`, although `benchmark.py:632` requires it. It says results go to `outputs/benchmark_results.json`, despite the current benchmark requiring result output inside the isolation root. It also says fixture namespaces are separate “per case and mode”; `_run_dataset` constructs each case fixture once and runs all selected modes against it. `evaluation/README.md:40-44` says `retrieve` does not accept an as-of timestamp and describes only oracle temporal cases, but `hybrid_retriever.py:12-13, 55-61` accepts and routes timestamps and the six-case `evaluation/temporal_end_to_end.jsonl` is explicitly end-to-end.

**Risk:** Following the README benchmark command fails at argument parsing; the evaluation README can lead readers to mislabel current end-to-end temporal results.

**Recommendation:** Update both documents with a fresh isolated-root command, separate the deterministic/mock, oracle-only, end-to-end, and live result descriptions, and state that benchmark fixtures may be seeded directly into SQLite.

#### H2 — Importing the database module initializes the configured application database

**Evidence:** `config.py:33-37` creates configured storage directories during import. `database.py` ends with a module-level `db=Database()`; `Database.__init__` creates its schema and indexes. Existing tests such as `tests/test_core.py` import `Database` at module scope. This means test collection can open/create the configured application database before individual tests instantiate their temporary databases. The benchmark explicitly avoids this by checking imports and applying isolation before loading storage modules.

**Risk:** Test operations use temporary databases, but import-time setup can still touch the configured default path. The audit did not compare production files before and after and therefore does not conclude that production data was changed.

**Recommendation:** For test runs, set `CAUSALMEM_DB`, `MEM0_PATH`, and `MEM0_DIR` to unique temporary paths in the process environment before pytest starts. Longer term, avoid database creation as a module-import side effect. Add a test that imports the app modules in a clean subprocess with isolated paths and verifies the default store is not opened.

### Medium

#### M1 — SQLite `julianday` comparisons collapse sub-millisecond instants

**Evidence:** `database.py:49-71` implements interval checks with SQLite `julianday`. A safe in-memory SQLite probe compared a timestamp ending in `.123456` with `.123455`, `.123400`, and `.123000`; each pair compared equal under `julianday`. Thus a request just before a validity boundary can be treated as exactly at the boundary, contrary to strict microsecond half-open semantics. No persistent database was used by the probe.

**Recommendation:** Compare parsed, aware Python datetimes at full stored precision after a user-scoped SQL read, or adopt a documented integer-epoch representation in a future migration. Add regression cases one microsecond and one millisecond before/after a boundary before changing implementation.

#### M2 — Mem0 and SQLite are not one atomic transaction

**Evidence:** `app.py:38-42` calls `add_memory` and `get_memories` before `ingest_message` writes SQLite. SQLite rollback protects state and transition changes but cannot undo a successful Mem0 write if extraction or SQLite persistence later fails.

**Risk:** Mem0 can contain a message without corresponding explicit state/transition evidence. This is an architectural consistency limit, not a failed SQLite transaction.

**Recommendation:** Document this partial-write behavior and consider an idempotent retry/reconciliation record in a later phase. Do not claim distributed atomicity.

#### M3 — Streamlit’s fallback memory-ID attribution can select an unrelated record

**Evidence:** `app.py:38-42` first takes IDs returned by `add_memory`; if none are returned, it uses `memories[-1]` from `get_memories`. The code does not prove that this last record corresponds to the submitted message. By contrast, `benchmark.py` attempts before/after ID difference and exact-text mapping, and declines ambiguous fixture attribution.

**Risk:** A state or transition may receive an incorrect source Mem0 ID if the add response lacks IDs and `get_memories` ordering is not a stable “newest first” contract.

**Recommendation:** Use a stable before/after ID diff or exact message-to-record match; if attribution remains ambiguous, preserve the memory without inventing a provenance link. Add a mocked test for an ambiguous get-all ordering.

#### M4 — Mem0 read isolation depends on the backend filter being honored

**Evidence:** `mem0_engine.py:28-33` passes the requested user filter. SQLite and graph reads are explicitly user-filtered. `hybrid_retriever.py:19` accepts the returned Mem0 results directly when no as-of timestamp is supplied; the explicit returned-record user check appears only in the as-of filtering branch.

**Risk:** No cross-user leak was demonstrated, and the current Mem0 adapter sends the user filter. However, there is no defense-in-depth post-filter if a backend or adapter returns a record with a conflicting `user_id`.

**Recommendation:** Filter records with an explicit nonmatching `user_id` before they enter the evidence bundle, and test a deliberately foreign record. Define fail-closed handling for records without a user field based on the Mem0 version’s verified response shape.

#### M5 — User IDs are namespaces, not authentication

**Evidence:** `app.py:19` exposes a freely editable user-ID field; database/Mem0 scoping uses that string. There is no login, identity verification, or authorization layer. README correctly limits the app to a local prototype and says it is not a hosted multi-tenant service.

**Recommendation:** Keep deployment local/trusted. Before exposing it to multiple users, add authenticated identity-to-namespace binding and ownership checks for memory update/delete operations. `mem0_engine.py:35-36` currently accepts only a memory ID for those helpers; no UI caller was found in the inspected app.

#### M6 — Benchmark failures do not preserve completed prediction rows

**Evidence:** `benchmark.py::_run_dataset` accumulates rows in memory and `main` calls `runtime.save` only after all dataset cases complete (`benchmark.py:467-658`). Progress records and stack frames help diagnose errors, but an error before the final save leaves no auditable partial result file.

**Recommendation:** Consider append-only per-case checkpoints or a separate failure manifest, with exclusive file creation and the same isolation-root checks. Keep failed cases distinct from valid predictions.

#### M7 — Runtime and dependency resolution are not fully reproducible

**Evidence:** `README.md:30` creates a Python 3.11 environment. The inspected environment is Python 3.12.10 with pytest 9.1.1, mem0ai 2.2.1, pydantic 2.13.5, openai 3.24.0, Streamlit 1.65.0, Ollama Python package 0.6.3, and NetworkX 3.7. `requirements.txt` uses lower bounds only; there is no lock file or declared tested Python range. The application uses the OpenAI-compatible client and Mem0 Ollama provider; no application import of the `ollama` Python package was found.

**Risk:** A fresh install may resolve materially different transitive dependencies. The documented Python version and the recently tested environment also differ.

**Recommendation:** Declare supported Python versions and add a reproducible lock or constraints file after deciding which runtime to support. Verify fresh installs in CI before publishing a release.

### Low

#### L1 — Temporary benchmark/test outputs are not covered by `.gitignore`

**Evidence:** `.gitignore` covers `.env`, `.venv`, `data/`, `outputs/`, Python bytecode, and `.pytest_cache`, but not `.tmp/` or `pytest-cache-files-*`. Temporary runs can contain SQLite/Qdrant stores and prediction outputs. No Git repository is present, so whether any file is tracked cannot be checked.

**Recommendation:** Ignore the project’s temporary run roots and continue using unique directories. Before a release, initialize/inspect Git metadata and confirm that `.env`, local stores, and run artifacts are not tracked.

#### L2 — Model and optional dependency requirements are underspecified

**Evidence:** `.env.example` includes a commented Ollama example, while `README.md:39` does not provide exact model pull commands. `mem0_engine.py:21-22` requires the configured LLM model and `nomic-embed-text` for the Ollama path. During the reported live run, spaCy and FastEmbed were absent and Mem0 warned that optional NLP and BM25 keyword search were disabled; the six-case run still completed. The Python `ollama` package is declared but not used by the inspected application modules.

**Recommendation:** Document exact tested Ollama model names and pull commands, distinguish required from optional retrieval extras, and remove or justify unused dependencies in a separately validated change.

## Evaluation labeling and claims

- `tests/test_core.py`, `tests/test_state_extractor.py`, `tests/test_hybrid_answers.py`, `tests/test_evaluator.py`, and `tests/test_ingestion_retrieval_integration.py` primarily use temporary SQLite and mocked model/Mem0 responses. Phase 6E’s answer assertions validate deterministic gate behavior and answer formatting; they do not measure live LLM answer quality.
- `evaluation/temporal_end_to_end.jsonl` contains six synthetic cases with `temporal_metric_scope=end_to_end`. `evaluator.py::_end_to_end_temporal_metrics` compares fixture keys on returned states and does not oracle-filter them. Oracle-only interval scores are separated in `_temporal_metrics`.
- Required/forbidden answer facts use substring proxies, not semantic entailment. Causal-claim detection is lexical and can miss implicit claims. Mem0-only causal support uses a weaker lexical proxy than the hybrid gate. P95 latency is omitted below 20 observations.
- The Phase 6D live run reported six completed synthetic cases and 100% temporal/evidence/answer proxies. Those results are small-sample observations, not evidence of general reliability. That run seeded temporal states directly into isolated SQLite and had no conversation ingestion for those cases; it therefore did not validate live Mem0 write/retrieval or live state extraction quality.
- The `evaluation/README.md` currently contradicts the implemented timestamp route and needs correction before the evaluation artifacts can be considered release-ready.

## Commands and audit validation

Read-only runtime inspection:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pytest --version
.\.venv\Scripts\python.exe -m pip show mem0ai pydantic openai streamlit ollama networkx pytest python-dotenv
```

Observed: Python 3.12.10, pytest 9.1.1; package versions are listed in finding M7. These commands did not install packages.

SQLite precision probe used only an in-memory database and exited 0:

```powershell
@'
import sqlite3
c = sqlite3.connect(':memory:')
boundary = '2026-10-23T10:00:00.123456+00:00'
for instant in ('2026-10-23T10:00:00.123455+00:00',
                '2026-10-23T10:00:00.123400+00:00',
                '2026-10-23T10:00:00.123000+00:00'):
    print(instant, c.execute(
        'select julianday(?) < julianday(?), julianday(?) = julianday(?)',
        (instant, boundary, instant, boundary)).fetchone())
'@ | .\.venv\Scripts\python.exe -
```

No pytest suite or benchmark was run during this audit. The latest full-suite result in the project/session context is 88 passed from Phase 6E; it was not rerun here because no implementation code changed. `Test-Path .git` returned false; `git status --short` exited 128 with “not a git repository.” Consequently, tracked-file history and production before/after state are not verifiable in this workspace.

Suggested post-fix validation command for Windows PowerShell, using a new base for each run:

```powershell
$testRoot = Join-Path (Get-Location) ('.tmp\pytest-run-' + [guid]::NewGuid().ToString('N'))
$testTemp = Join-Path $testRoot 'pytest'
$oldDb = $env:CAUSALMEM_DB; $oldMem0 = $env:MEM0_PATH; $oldMem0Dir = $env:MEM0_DIR
$env:CAUSALMEM_DB = Join-Path $testRoot 'storage\causalmem.sqlite3'
$env:MEM0_PATH = Join-Path $testRoot 'storage\mem0'
$env:MEM0_DIR = Join-Path $testRoot 'storage\mem0-config'
.\.venv\Scripts\python.exe -m pytest -q tests -p no:cacheprovider --basetemp="$testTemp"
$testExit = $LASTEXITCODE
if ($null -eq $oldDb) { Remove-Item Env:CAUSALMEM_DB -ErrorAction SilentlyContinue } else { $env:CAUSALMEM_DB = $oldDb }
if ($null -eq $oldMem0) { Remove-Item Env:MEM0_PATH -ErrorAction SilentlyContinue } else { $env:MEM0_PATH = $oldMem0 }
if ($null -eq $oldMem0Dir) { Remove-Item Env:MEM0_DIR -ErrorAction SilentlyContinue } else { $env:MEM0_DIR = $oldMem0Dir }
Write-Output "TEST_EXIT=$testExit"
```

For a benchmark, use a newly generated isolation root and outputs directory, for example:

```powershell
$runRoot = Join-Path (Get-Location) ('.tmp\benchmark-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe benchmark.py --isolation-dir "$runRoot" --output-dir "$runRoot\results" --dataset "evaluation\temporal_end_to_end.jsonl" --modes full
```

This command is a future run instruction, not an audit benchmark result.

## Release-readiness checklist

- [ ] Reconcile `README.md` and `evaluation/README.md` with the current isolation CLI and end-to-end timestamp metric.
- [ ] Decide and document a supported Python version range; lock or constrain dependencies.
- [ ] Make test collection safe from the configured application SQLite path; validate in a clean subprocess.
- [ ] Correct or explicitly bound sub-millisecond temporal interval precision and add edge tests.
- [ ] Add regression coverage for Mem0 response post-filtering and ambiguous source-ID attribution.
- [ ] Document the non-atomic Mem0/SQLite write boundary and add recovery guidance.
- [ ] Keep hosted/multi-user deployment disabled until authentication and ownership enforcement exist.
- [ ] Add isolated benchmark failure checkpoints if partial case auditability is required.
- [ ] Add `.tmp` and local benchmark artifacts to ignore rules, then inspect Git tracking once repository metadata exists.
- [ ] Re-run focused tests and the full suite after any implementation change; separately label deterministic, oracle-only, end-to-end, and live evaluation results.
- [ ] Perform and archive live Mem0/Qdrant persistence and Ollama extraction smoke tests in a fresh isolated run before making live integration claims.

**Audit conclusion:** Core local retrieval safeguards are substantial, but documentation and temporal precision need attention, and live Mem0 write/extraction behavior remains unvalidated. Treat this as a research prototype for trusted local use, not a release-ready multi-user service.
