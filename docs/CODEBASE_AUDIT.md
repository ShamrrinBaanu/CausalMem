# CausalMem Codebase Audit

> **Fresh audit recreated 2026-10-22.** The previously reported audit file was not present in this workspace and could not be recovered from a Git repository: the project root has no `.git` directory. This document is based on current source inspection and the commands/tests recorded below; it does not claim to reproduce the prior report.

## Scope and environment

Project root: `C:\Users\Shamrrin S\Downloads\Final`. Python is 3.12.10; pytest is 9.1.1. Installed core versions include mem0ai 2.2.1, Ollama Python client 0.6.3, OpenAI Python client 3.24.0, Streamlit 1.65.0, NetworkX 3.7, and Pydantic 2.13.5. Runtime configuration identifies Ollama and `llama3.2:3b`; `.env` values are deliberately not reproduced. `nomic-embed-text` is configured in `mem0_engine.py` as Mem0's embedding provider; dimension 768 is configured for its Qdrant collection. Live Ollama/Mem0 calls were not repeated in this audit.

## Architecture and module responsibilities

The Streamlit interface in `app.py` routes a submitted prompt through `query_router.classify_query`. Question prompts invoke `hybrid_retriever.retrieve` and `answer_generator.generate_answer`; statement prompts go to real Mem0 (`mem0_engine.add_memory`), list memories, then call `state_extractor.ingest_message`. Extracted states and edges are stored in SQLite. `causal_graph.py` reconstructs a NetworkX graph from SQLite, `temporal_engine.py` reads states and timeline, and `visualization.py` draws it using Plotly.

- `config.py`: loads dotenv-backed provider, model, paths, and Ollama/OpenAI-compatible endpoints; creates data directories.
- `llm_service.py`: structured extraction, query classification, answers, and user-facing error mapping via OpenAI-compatible client.
- `mem0_engine.py`: real Mem0 `Memory.from_config`; wraps add/search/get_all/update/delete. Ollama uses native Ollama model/embedder provider, local Qdrant, separate collection and 768 dimensions.
- `models.py`: Pydantic state, transition, query-analysis, and extraction records.
- `database.py`: SQLite schema and per-operation connection/commit/rollback/close behavior.
- `state_extractor.py`: state grounding, explicit-cause parsing, fallback state generation, and persistence.
- `query_router.py`: LLM classification plus deterministic question-start/punctuation fallback.
- `hybrid_retriever.py`: Mem0 search plus category-dependent state/timeline/graph evidence.
- `answer_generator.py`: abstains if both memory and state evidence are absent, otherwise asks LLM to answer from evidence.
- `temporal_engine.py`, `transition_engine.py`, `causal_graph.py`: queryable state history, edge retrieval, graph reconstruction/path lookup.
- `visualization.py`, `app.py`: current Streamlit UI and displays.
- `benchmark.py`, `dataset_generator.py`, `evaluator.py`: synthetic evaluation harness, fixture, and lexical metrics.
- `tests/`: config, error sanitization, evaluator, SQLite isolation/graph, and state extraction tests.

These backend modules are separable from Streamlit in principle, but several modules bind global `db`, `llm`, and cached Mem0 service objects at import time; a future frontend would need deliberate service boundaries.

## Confirmed ingestion behavior and bug

**High — arbitrary utterance fallback can persist as a personal state.** `query_router.classify_query` attempts deterministic recognition only for a question mark or a fixed set of leading words, then delegates other utterances to the LLM. In `app.py`, `analysis.is_question == False` enters the statement/ingestion branch. `state_extractor.ingest_message` calls extraction, grounds model states, and if none survive calls `_meaningful_fallback`. That helper accepts almost any utterance with at least two words. It creates `OTHER` whose `state_value` is the entire input, confidence 0.35. The persistence loop constructs a `State` and calls `db.save_state`; `causal_graph.build_graph` includes every stored state. Thus a question misrouted as a statement can become both a durable state and graph node. The route guard fixes the known prefix case but is not a complete deterministic question/request validator.

**Medium — not every rejected extraction can create edges, but independently returned valid state items can.** Edge construction is inside the loop over accepted `items`. If `items` is empty, no edge is created. But the generic fallback makes items nonempty for most multiword messages, and model transitions can then be considered. Transitions require a source state and a target phrase occurring in the message; explicit cause parsing has its own cause/target/source requirements. Graph nodes derive from stored states; graph edges from stored transitions.

**High — persistence is not atomic across one ingestion.** `database.connect()` commits or rolls back a single SQLite connection. Each `save_state` and `save_transition` opens its own connection and commits independently. `ingest_message` writes records sequentially without a surrounding transaction. If a later write fails, earlier writes remain committed. If LLM extraction throws before the persistence loop, no state/edge write in that call has occurred. A failure during a loop can leave partial state/transition data.

**Medium — repeated transition duplication is possible.** `state_extractor` can save a new `Transition` on each repeated extraction; there is no uniqueness constraint or deduplication by endpoints/evidence. Repeated state values are merged by active-state value, but edges are not similarly de-duplicated.

**Medium — causal inference is partially constrained.** Explicit `EXPLICIT_CAUSE` edges are produced by regex matching cause marker, change verb, target and a matching/sole active source. LLM claims labeled EXPLICIT_CAUSE are downgraded to UNKNOWN. Other model relation types, including TEMPORAL_SUCCESSION and INFERRED_RELATIONSHIP, may still be persisted if endpoints meet current textual checks. This is evidence filtering, not calibrated causal inference.

**Medium — transition evidence uses the current input's memory IDs.** The app passes IDs returned by the current Mem0 add operation (or falls back to the most recent listed memory). `state_extractor` attaches that list to state and transition records. There is no verified cross-check that fallback `memories[-1]` corresponds to the just-ingested prompt when Mem0 ordering or deduplication differs.

## Database schema and persistence

`database.py` creates `states` and `transitions` tables plus user/time indexes. State columns include identifier, user, type/value, active/inactive status, validity timestamps, confidence, source memory IDs JSON, and metadata JSON. Transition columns include identifiers/endpoints, transition and relation types, trigger/reason, timestamp, confidence, source memory IDs JSON, and metadata JSON. No foreign-key constraints, schema migration framework, or uniqueness constraint for duplicate transitions is defined. Reads are filtered by `user_id`; `clear_user` deletes transitions and states for that identifier. Durable SQLite persistence is implemented, but live restart survival was not separately exercised here. The existing production database `data/causalmem.sqlite3` was not opened for writes or modified.

## Mem0, Ollama, and retrieval

`mem0_engine.py` calls actual Mem0 OSS API, not a UI-only placeholder. It uses local Qdrant path storage; Ollama config sets the LLM provider to `ollama`, and embedder provider to `ollama` with model `nomic-embed-text`. LLM calls use the OpenAI-compatible client configured with Ollama `/v1` URL, while Mem0 receives the Ollama server root. The code sets 768 embedding dimensions and separate collection/path for Ollama versus OpenAI. That agrees with the expected nomic embedding dimension by configuration, but no fresh live embedding call was performed.

`hybrid_retriever.retrieve` always searches Mem0. The routed category controls whether timeline and graph evidence are included. For non-temporal categories it fetches active SQLite states; temporal categories include query-filtered timeline states; graph categories add transitions and a relevant graph path. `answer_generator` supplies evidence to the model and prompts it not to use outside facts, but correctness is not enforced beyond that prompt and the empty-evidence abstention.

## User isolation and security

SQLite state and transition queries filter by user ID, and tests cover state isolation. Mem0 calls pass user ID filters/namespaces for add/search/get_all. This is application-level identifier separation, not authentication or authorization: any user able to access the local UI can type another ID. The app is a local prototype, not a secured multi-tenant service. `.gitignore` contains `.env` and `data/`, but Git tracking/history could not be checked because `.git` is absent. No secret value is included in this report. Keep `.env` excluded and rotate any credential previously exposed outside the local workspace.

## Test and temp-directory diagnostic

Commands run:

- `\.venv\Scripts\python.exe --version` → Python 3.12.10.
- `\.venv\Scripts\python.exe -m pytest --version` → pytest 9.1.1.
- `\.venv\Scripts\python.exe -m pytest -q` → collection errors on four workspace `pytest-cache-files-*` directories with `PermissionError: [WinError 5] Access is denied`; pytest also warns it cannot create cache paths.
- `\.venv\Scripts\python.exe -m pytest -q tests` → 4 passed, 4 errors; the errors are `tmp_path` fixture setup failures attempting to access `%TEMP%\pytest-of-Shamrrin S`. Four existing inaccessible `pytest-cache-files-*` folders also explain the unscoped collection failures.
- `\.venv\Scripts\python.exe -m pytest -q tests/test_config.py tests/test_errors.py tests/test_evaluator.py -p no:cacheprovider` → **4 passed**.
- Import smoke check for config, Mem0 adapter, LLM, extractor, temporal/graph/retrieval/visualization modules → passed; printed effective provider/model only.

`TEMP` and `TMP` point to `C:\Users\SHAMRR~1\AppData\Local\Temp`; `PYTEST_ADDOPTS` and `PYTHONPATH` were not set. ACL inspection succeeded for workspace `.tmp`, showing modify permission for the current user and system administrators; reading ACLs on one `pytest-cache-files-*` directory was denied. An attempted `--basetemp` beneath workspace `.tmp` still failed with WinError 5 during pytest cleanup/access. This environment's sandbox/Windows directory ACL behavior is therefore implicated, but the exact owner/ACL of inaccessible temp/cache entries could not be verified. No temp directories were deleted and no permissions changed. A future safe workaround is to create a new uniquely named writable folder in an approved location and verify create/list/delete access before passing it to `--basetemp`; do not reuse or clean existing pytest folders automatically.

No LLM-backed or live Mem0/Ollama integration tests were run. The full suite has no clean pass result in this environment; report only the exact results above.

## Protection and change status

No `.git` directory is present at the current project root. Git status and tracked-file history are therefore unavailable; no claim about whether `.env` has ever been committed can be made. A SHA-256 hash of `data/causalmem.sqlite3` was read for verification during this recovery diagnostic; the database was not modified. No source, dependency, schema, database, or UI files were changed during this phase. This report is the only intended project modification.

## Prioritized remediation plan

1. **High:** add deterministic intent/utterance validation immediately before state persistence; exclude questions, commands/requests, and unsupported text while allowing evidenced factual statements. Ensure no fallback creates a state solely because the utterance has multiple words.
2. **High:** persist one ingestion's states and transitions in one SQLite transaction; add failure-side-effect tests.
3. **High:** add isolated tests for statement/question/ambiguous text, provider failure, user isolation including graph, and repeated transition idempotency.
4. **Medium:** tighten transition endpoint/evidence checks and deduplicate repeated edges; document confidence as uncalibrated.
5. **Medium:** make temp-dir diagnostics reproducible in this Windows environment and rerun full suite using a verified writable isolated basetemp.
6. **Low:** introduce service boundaries for eventual frontend replacement after backend correctness is covered.
