# CausalMem Repair Log

## Phase 0 - Reproduce and diagnose

### Repository state
- `git status` / `git diff --stat`: **not a git repository** (`fatal: not a git repository`). There is no Git history or diff available in this workspace. Changes are tracked below by file.
- Production data remains untouched. All repro work must use temporary paths.

### Live process / stale-code check
- Running backend: PID 13728, command `"C:\Users\Shamrrin S\Downloads\Final\.venv\Scripts\python.exe" api.py`, started **2026-10-25 09:46:37**.
- File modification times:
  - `state_extractor.py`: 2026-10-25 13:34:12
  - `chat_service.py`: 2026-10-25 13:34:42
  - `hybrid_retriever.py`: 2026-10-25 13:41:36
- **CONFIRMED:** the browser-facing backend process predates the latest source edits by ~4 hours. It cannot be serving the current extraction/retrieval code. It must be restarted for any live verification after fixes.

### Real API path
- Entry point: `api.py`, FastAPI app.
- Chat endpoint: `api.py:109` `POST /api/chat` -> `api.py:110` `chat()` -> `chat_service.process_chat()`.
- Ingestion path inside `process_chat`: `chat_service.py:191` deterministic check -> `add_memory()` (`mem0_engine.py:141`) -> `ingest_message()` (`state_extractor.py:875`).
- Retrieval path: `chat_service.py:171` -> `hybrid_retriever.retrieve()` -> `answer_generator.generate_answer_with_diagnostics()`.

### Repro results
Command: `.\.venv\Scripts\python.exe tests\repro_live_failures.py --scenario <A-E>`
Each run used a new `%TEMP%\causalmem-repro-*` root with its own SQLite, Qdrant, Mem0 config, and `test_user_<random>` namespace.

| Case | Observed behavior (current source, isolated stores) |
|---|---|
| A | With prior `I love running.`: switch ingest created `INTEREST running` (closed) + `ACTIVITY swimming` + `EXPLICIT_CAUSE` edge. `Why did I switch...` answered with the edge. `Did I switch because of a knee injury or is that unknown?` retrieved 2 states, **0 transitions**, model returned an uncited claim, answer abstained (`citation_ref_unknown`). |
| B | `I prefer Python for programming.` then `I prefer Java for programming.` created a replacement edge. `What programming language do I prefer now?` correctly answered Java. `what i prefer for programming` retrieved both `INACTIVE Python` and `ACTIVE Java` and answered **stale Python**. |
| C | `I prefer tea.` then `I prefer coffee.` left **two ACTIVE PREFERENCE states** and no transition. `What do I prefer?` safely abstained (`ambiguous_evidence`). `Do I prefer tea?` answered tea despite the conflict. |
| D | Move created `LOCATION Chennai` (closed) + `LOCATION Coimbatore` + `LOCATION_CHANGE`. Current location answered. Previous location retrieved Chennai + edge but validation rejected (`temporal_validity_failed`). Location-change question retrieved both endpoints + edge but rejected (`temporal_relation_or_support_rejected`). |
| E | Goal and completion were typed correctly on current source (`GOAL` + `LIFE_EVENT`, no Interest). Current goal and completion answered. Previous goal abstained because no previous goal existed (correct for this data). Live `OtherCurrent` behavior is explained by the stale server, not current extraction. |

Performance observed in isolated live repro: ingest calls took **21.5-105.9 s** each; query calls took **4.1-25.7 s**. The old browser server has been running since before the current extraction code was written.

### Root causes
- **A - CONFIRMED (retrieval):** `causal_graph.is_causal_question()` does not recognize yes/no causal paraphrases (`Did ... because ...`, `or is that unknown`), and `causal_graph.relevant_transitions()` only matches endpoint terms. Cause words in the question (`knee injury`) and the edge's `trigger`/`reason` are ignored. `hybrid_retriever.retrieve()` therefore returns 0 transitions for the paraphrase while the `why` form works.
- **A - CONFIRMED (source matching, secondary):** `state_extractor._explicit_change()` requires the stated origin to match an active state's exact value/substring. `I run regularly.` becomes `OTHER "I run regularly."`, so `running` does not match. A prior `I love running.` becomes `INTEREST running` and does match, which is why the isolated repro with that wording produced the edge.
- **B - CONFIRMED (routing/validation):** `what i prefer for programming` lacks a present-tense auxiliary and `now`, so `hybrid_retriever` does not set `current_intent`; it returns the full timeline including `INACTIVE Python`. `answer_generator` then accepts the stale historical state as a current preference. The `now` variant filters to ACTIVE and works, so the two routes disagree.
- **C - CONFIRMED (write semantics):** `state_extractor._preference_update()` only closes a predecessor when the statement has an explicit old value, a domain key, or a replacement marker. Plain `I prefer coffee.` after `I prefer tea.` has none, so both stay ACTIVE. The conflict check is not applied consistently to every route (`Do I prefer tea?` bypasses it).
- **D - CONFIRMED (answer validation):** retrieval returns the predecessor state and `LOCATION_CHANGE` edge, but `answer_generator` rejects the predecessor state with `temporal_validity_failed` and the change claim with `temporal_relation_or_support_rejected`. The typed data is present; validation support for `LOCATION_CHANGE` / closed-state evidence is missing.
- **E - CONFIRMED (stale server):** current `state_extractor` types goal/completion correctly. The live `OtherCurrent` result came from the backend process started at 09:46, before the 13:34 source changes. Previous-goal abstention is correct when no prior goal exists; goal replacement still needs explicit tests.
- **F - CONFIRMED (performance):** live isolated ingest is dominated by Mem0's extraction/embedding LLM calls (21-106 s), and causal queries that never call the answer model still spend seconds in Mem0 search plus validation. Instrumentation in Phase 5 must separate Mem0, embedding, LLM, and DB time.

## Phase 1 - Typed extraction and consistent writes

### Root cause
- **CONFIRMED:** current source already typed basic moves/goals/completions, but the deterministic parsers covered only narrow phrasings. `I am now trying/planning to ...` was not parsed as a goal statement, so it depended on the LLM. Completion cleanup only converted `LIFE_EVENT`/`OTHER` items; an extractor that labeled the completed object as `INTEREST` or `GOAL` left a duplicate typed state.
- **CONFIRMED:** `process_chat` had no response-level field for partial store writes. Mem0 and SQLite failures either raised or were invisible to the caller.

### Files changed
- `state_extractor.py`
  - Broadened `_GOAL_UPDATE_RE` / `_GOAL_STATEMENT_RE` to cover `I am (now/currently) trying|planning|working to ...` and `I now aim to ...`.
  - Broadened `_COMPLETION_RE` to cover `have/had/just/recently/already completed|finished|achieved|done with`.
  - Completion handling now removes any extractor item matching the completed object (including `INTEREST`/`GOAL`/`PREFERENCE`/`ACTIVITY`) and keeps exactly one `LIFE_EVENT`.
- `chat_service.py`
  - Added `storage_warnings`, `storage_errors`, and `memory_ids` to ingest responses.
  - Mem0 write failure returns `ingest_status="mem0_write_failed"`; structured-state write failure returns `ingest_status="state_write_failed"`. Both log only the exception type and never leak exception text.
  - A Mem0 success without a returned memory ID is surfaced as `storage_warnings=["mem0_no_memory_id"]`.
- `tests/test_comprehensive_matrix.py`
  - 4 location paraphrases, 4 goal paraphrases, 4 completion paraphrases, and store-failure response tests.

### Tests and results
- Targeted: `pytest -q tests/test_comprehensive_matrix.py -k "write_failures or location_move_paraphrases or goal_paraphrases or completion_paraphrases or process_chat"` -> **17 passed**.
- Full backend: `pytest -q` -> **407 passed, 13 warnings** (warnings are the pre-existing SQLite datetime adapter deprecation).

### Open issues
- Legacy `OtherCurrent` rows are not modified; Phase 4 audit script will report them.
- Mem0 writes happen before SQLite. A state-write failure now reports `state_write_failed`, but cannot roll back the already-written Mem0 record.


## Phase 2 - Query routing and temporal retrieval

### Root cause
- **CONFIRMED (previous location):** retrieval already returned the unique closed `LOCATION Chennai` state and its `TEMPORAL_SUCCESSION` edge. `answer_generator._temporal_state_claim_supported()` accepted the predecessor boundary, but lexical validation failed: the live model claimed `I lived in Chennai before`, and `lived` normalized to `liv` while the linked Mem0 record said `User lives in Chennai` (`live`) and the state label only had `LOCATION Chennai`. Synthetic tests had hidden this because their linked memory also used `lived`.
- **CONFIRMED (location change):** the model correctly cited the edge but transition lexical support only used raw endpoint values (`Chennai Coimbatore`). It omitted the state type (`LOCATION`) and treated the directional verb (`moved`) as a factual term, so a valid change claim was rejected.
- **CONFIRMED (predecessor routing):** `What goal did I have before?` was not recognized as a predecessor question because the pattern required `What did I ...` or a typed token followed by `previously`. `What did I finish?` was also not mapped to `LIFE_EVENT`, so it skipped completion focus.
- **CONFIRMED (model bundle):** predecessor questions exposed the transition to the answer model. That invites restating the whole change instead of the requested closed state, and can produce claims the state citation cannot support.

### Files changed
- `temporal_engine.py`
  - Added `lives`, `living`, `finish`, `finished`, `achieve`, and `achieved` to predecessor/type recognition.
  - Added predecessor phrasing for `What <typed> did I have before?` without making generic `before` questions predecessor queries.
  - Added `finish`/`achieve` to `LIFE_EVENT` terms.
- `hybrid_retriever.py`
  - Mapped `finish`, `finished`, `achieve`, and `achieved` to `LIFE_EVENT`.
- `answer_generator.py`
  - Transition endpoint text now includes resolved state types, so `LOCATION`/`GOAL`/`PREFERENCE` change claims are supported by the edge.
  - Directional and state-type verbs (`moved`, `live`, `lived`, `finish`, etc.) are treated as temporal framing, not extra factual terms.
  - Predecessor questions keep transitions in the validation bundle but remove them from the model-facing bundle, so the model answers from the closed state or edge from-state.
- `tests/repro_live_failures.py`
  - Added scenario `F` for goal replacement.
- `tests/test_comprehensive_matrix.py`
  - Added previous-location, previous-goal, completion, location-change, missing-predecessor, transition-citation, and inflected-`live` regression tests.

### Tests and results
- Targeted: `pytest -q tests/test_comprehensive_matrix.py -k "predecessor_state_claim or previous_location or previous_goal or completion_paraphrases or location_change or missing_predecessor"` -> **21 passed**.
- Full backend: `pytest -q` -> **422 passed, 13 warnings**.
- Live isolated scenario D (new temp DB/vector/user):
  - `Where do I live now?` -> Coimbatore, accepted 1, `validated_claims`.
  - `Where did I live before?` -> Chennai, accepted 1, `validated_claims`.
  - `What location change did I tell you about?` -> Chennai -> Coimbatore, accepted 1, `validated_claims`.
- Live isolated scenario F:
  - `What is my current goal?` -> `learn data structures`, accepted 1.
  - `What was my previous goal?` -> `learn system design`, accepted 1.
  - `What goal did I have before?` -> `learn system design`, accepted 1.
- Observed live timings in these runs: ingest 17.1-63.2 s; query 10.8-15.3 s.

### Open issues
- Final live verification must still restart the stale backend process before using the browser/API.
- Causal paraphrases, conflicting preferences, performance instrumentation, and the small UI diagnostics are Phase 3-6 work.

## Phase 3 - Causal retrieval and explicit-cause gate

### Root cause
- **CONFIRMED:** `causal_graph.is_causal_question()` recognized `why`, `what caused`, and `reason did`, but not yes/no paraphrases such as `Did I switch because ...`, `Was it because ...`, or `Did X cause Y`.
- **CONFIRMED:** causal retrieval only anchored on endpoint labels. A question naming the recorded trigger (`knee injury`) but not both endpoints returned no edge.
- **CONFIRMED:** the answer gate and causal model projection also required endpoint anchors, so a cause-anchored question could be rejected even when the edge contained the exact trigger.
- **CONFIRMED:** causal claim validation treated `switch` as a factual term rather than relation framing, so a correct claim like `A knee injury made me switch from running to swimming` failed lexical support.

### Files changed
- `causal_graph.py`
  - Added yes/no and alternative causal question forms (`because`, `Did X cause Y`, `what led/made`, `how come`).
  - Added causal framing words to generic terms so they cannot anchor retrieval.
  - Causal retrieval now matches query terms against the edge trigger/reason as well as endpoint labels.
- `answer_generator.py`
  - Added cause-anchored edge matching for the causal gate, destination resolution, and causal model projection.
  - Added causal framing words to relevance filtering.
  - Added `switch`/`replace` forms to causal claim framing so direction and cause are validated separately from relation wording.
- `tests/test_comprehensive_matrix.py`
  - Added 5 causal paraphrase retrieval tests, 2 unsupported-cause retrieval/abstention tests, an accepted causal citation test, and an unrecorded-cause rejection test.

### Tests and results
- Targeted: `pytest -q tests/test_comprehensive_matrix.py -k "causal_paraphrases or unsupported_cause or causal_claim or causal_transition"` -> **10 passed**.
- Full backend: `pytest -q` -> **431 passed, 20 warnings**.
- Live isolated scenario A:
  - `Why did I switch from running to swimming?` -> edge cited, gate passed, `validated_claims`.
  - `Did I switch because of a knee injury or is that unknown?` -> edge cited, gate passed, `validated_claims`.
  - `Did I switch because of a broken ankle?` -> 0 transitions, `missing_explicit_cause_transition`, gate false, model not called.
- Observed live timings in this run: ingest 82.0-86.7 s; causal queries 15.1-26.2 s.

### Open issues
- A switch from a loosely typed prior state (`I run regularly.` -> `OTHER`) is still a write/extraction concern, not a causal-gate concern. It is covered by the Phase 4 audit/consistency work.
- Performance remains dominated by Mem0; Phase 5 instrumentation is still required.

## Phase 4 - Preferences, conflicts, duplicates, and legacy audit

### Root cause
- **CONFIRMED:** plain `I prefer tea.` then `I prefer coffee.` left both `PREFERENCE` states ACTIVE with no transition. The replacement fallback only ran for statements containing `now`, `instead`, an old value, or a domain key.
- **CONFIRMED:** conflict ambiguity was only applied when `current_snapshot` was true. A preference question classified as `PREFERENCE`/`HISTORY` could still expose multiple active preferences and answer a stale one.
- **CONFIRMED:** `what i prefer for programming` lacked an auxiliary verb, so it was routed as history and included inactive Python even though a current replacement existed.
- **CONFIRMED:** there was no read-only audit for legacy `OTHER` active rows, duplicates, conflicts, dangling edges, or inconsistent replacement sources.

### Files changed
- `state_extractor.py`
  - A plain undomained preference update now closes exactly one prior undomained active preference and records `PREFERENCE_CHANGE`.
  - Domain-keyed preferences remain independent and are not auto-closed.
  - Multiple competing preferences remain a real conflict and are not silently collapsed.
- `hybrid_retriever.py`
  - Added present-tense no-auxiliary forms (`what i prefer ...`) and present yes/no forms (`Do I prefer ...?`) to current-intent routing.
  - Applied the multiple-active-candidate ambiguity rule to every current typed snapshot route, not only the `STATE` category.
- `scripts/audit_legacy.py` (new)
  - Opens SQLite with `mode=ro`; never writes, migrates, or repairs.
  - Reports active `OTHER`, duplicate active states, conflicting singular types, dangling/self/duplicate transitions, active replacement sources, explicit-cause edges without triggers, and invalid intervals.
  - Supports `--db`, `--output`, and `--limit`.
- `tests/test_comprehensive_matrix.py`
  - Updated the plain-preference test to require closure/transition.
  - Added programming-preference routing and legacy-conflict abstention tests.
- `tests/test_audit_legacy.py` (new)
  - Verifies findings and that the database bytes are unchanged after auditing.

### Tests and results
- Targeted preference tests -> **7 passed**.
- Audit tests -> **2 passed**.
- Full backend: `pytest -q` -> **435 passed, 20 warnings**.
- Live isolated scenario B:
  - `What programming language do I prefer now?` -> Java, `validated_claims`.
  - `what i prefer for programming` -> Java only, `validated_claims` (no inactive Python).
- Live isolated scenario C:
  - `I prefer tea.` then `I prefer coffee.` -> coffee ACTIVE, tea closed, `PREFERENCE_CHANGE`.
  - `What do I prefer?` -> coffee, `validated_claims`.
  - `Do I prefer tea?` -> safe abstention (`no_validated_claims`), no stale yes.
- Audit smoke test on an isolated scenario database -> all finding counts 0, `read_only: true`.
- Observed live timings in these runs: ingest 62.6-68.6 s; queries 8.7-26.3 s.

### Open issues
- Legacy production conflicts are reported, not repaired. Manual review is required.
- `Do I prefer tea?` after a replacement safely abstains rather than answering "no"; this is conservative and does not claim unsupported negation.

## Phase 5 - Latency instrumentation and measured optimization

### Root cause
- **CONFIRMED:** the response had no per-turn request identifier and no phase breakdown. `total_ms` mixed intent classification, retrieval, and answer generation.
- **CONFIRMED:** Mem0's own fact-extraction LLM dominated deterministic ingestion: live scenario D measured `llm_memory_extraction` at 47.2 s and 60.1 s inside 58.2-64.8 s Mem0 writes.
- **CONFIRMED:** deterministic statements were also paying for an unnecessary query-classification LLM call (4.5-5.5 s), even though the local extractor already proved the message was a statement.
- **CONFIRMED:** query latency was split between classification (1.8-5.0 s), Mem0 search (0.06-2.3 s), and answer generation (6.1-7.3 s in isolated runs). The answer model is now the largest query cost.

### Files changed
- `chat_service.py`
  - Added `request_id` (`req_<uuid hex>`) to every answer, ingest, ignored, and partial-write response.
  - Added `classification_ms`; wired retrieval and generation sub-timings into the response.
  - Deterministic statements skip the classifier and use a local statement analysis.
  - Passes `infer=not deterministic` to Mem0: deterministic messages store raw provenance without a second Mem0 fact-extraction LLM pass; free-form messages keep normal Mem0 inference.
  - Ingestion timings now expose `mem0_infer`.
- `mem0_engine.py`
  - `add_memory(..., infer=True)` forwards the Mem0 inference choice without changing response normalization.
- `hybrid_retriever.py`
  - `retrieve(..., timings=...)` records `mem0_search_ms`, `timeline_db_ms`, `graph_ms`, and `total_ms`.
- `answer_generator.py`
  - Diagnostics now record `model_ms`, `validation_ms`, `render_ms`, and `total_ms`.
- `llm_service.py`
  - Ollama requests send `keep_alive: 30m` when the OpenAI-compatible endpoint supports it.
- `tests/repro_live_failures.py`
  - Prints request IDs and timing JSON for each live turn.
- `tests/test_comprehensive_matrix.py`
  - Added request ID, timing breakdown, deterministic-classifier skip, and Mem0 inference-mode tests.
- `tests/test_mem0_attribution.py`
  - Added a test that `infer=False` is forwarded to Mem0.

### Tests and results
- Targeted process-chat tests -> **7 passed**.
- Full backend: `pytest -q` -> **438 passed, 20 warnings**.
- Live isolated scenario D after optimization:
  - Ingest 1: **62.8 s -> 6.7 s**; ingest 2: **70.5 s -> 0.1 s**.
  - Classification on deterministic ingests: **4.5-5.5 s -> 0.1-0.6 ms**.
  - Queries stayed correct: current Coimbatore, previous Chennai, Chennai -> Coimbatore transition.
  - Query totals: 10.6-13.4 s; answer generation 6.6-7.2 s remained dominant.
- Live isolated scenario B after optimization:
  - Ingest 1: 4.2 s; ingest 2: 0.1 s.
  - Java remained current; both preference queries returned `validated_claims`.
- Isolated API verification on port 8001 with temp DB/Mem0:
  - Ingest returned `request_id`, `classification_ms`, `mem0_infer=false`.
  - Query returned `request_id`, `classification_ms`, `retrieval_ms`, `generation_ms`, and Mem0 search timing.
  - Built frontend root and JS asset served HTTP 200.
- Production backend restarted: stopped PIDs 13728/21960, started PID 16108.
  - `/api/health` -> ok; production answer included `request_id` and retrieval/generation breakdown.
  - Broad production query abstained with `no_validated_claims`; total 32.4 s, answer generation 24.6 s.

### Open issues
- Query latency is still dominated by the local answer model, especially on a broad production evidence bundle; no semantic shortcut was added.
- Mem0 raw writes for deterministic messages trade Mem0's duplicate/update inference for speed. SQLite typed states remain authoritative and the raw memory is retained for provenance.
- Production legacy records were not modified.

## Phase 6 - Response diagnostics UI

### Root cause
- **CONFIRMED:** the UI showed raw diagnostic codes, no request ID, and no intent/retrieval breakdown. Legacy/untyped records were visually indistinguishable from fully typed records.

### Files changed
- `web/src/types.ts`
  - Added `request_id`, `classification_ms`, retrieval and generation timing types, and `mem0_infer`.
- `web/src/components/Timing.tsx`
  - Turn latency now includes intent routing.
  - Added a retrieval breakdown for Mem0 search, timeline SQLite, and causal graph.
- `web/src/components/ui.tsx`
  - Added human-readable diagnostic labels and legacy/review helpers.
- `web/src/pages/ChatPage.tsx`
  - Shows the validated answer reason and request ID.
- `web/src/pages/DiagnosticsPage.tsx`
  - Shows request ID, intent routing, retrieval breakdown, and human-readable rejection labels.
- `web/src/pages/EvidencePage.tsx`
  - Shows request ID, intent timing, human-readable reasons, and legacy/review badges.
- `web/src/pages/MemoryPage.tsx` and `web/src/pages/TimelinePage.tsx`
  - Added legacy/review badges for `OTHER`/`UNKNOWN` or explicitly marked records.
- `web/src/styles.css`
  - Added request-ID wrapping/truncation.

### Tests and results
- Frontend production build: `npm run build` -> **success**.
- Built asset served by the isolated API -> HTTP 200.
- Production `/api/health` -> ok after restart.

### Open issues
- Browser-level visual interaction was not automated; the build and served asset were verified.
- Legacy badges use explicit metadata plus the `OTHER`/`UNKNOWN` heuristic, so manual review is still required for production legacy rows.


## Phase 7 - Visible answer cleanup and live regression repair

### Initial inspection
- The workspace still has no `.git` repository metadata. `git status --short` and `git diff` both reported “not a git repository”; no Git baseline can be used to identify prior edits.
- The six phases above were present in the checked-out source before this turn. They were inspected rather than reapplied.
- Production data was not opened for edits or migration. The end-to-end runner uses a fresh temporary SQLite/Mem0 store and generated test namespaces.

### Files changed
- `chat_service.py`
  - The API response now extracts validated source IDs into `citations` before removing `[Sources: ...]` markers from the visible `answer`. Evidence and IDs remain available to the Evidence Inspector.
- `answer_generator.py`
  - Causal fallback wording now adds a punctuated, natural follow-up (“I then switched to …”) while keeping its transition citation.
  - A validated causal claim using the generic “The recorded reason for the change was …” template is rendered as “The change happened because of …”; this post-validation wording normalization keeps the accepted cause and transition citation unchanged.
  - Historical answer wording such as “I used to live in Chennai” now passes the existing predecessor check after the unique transition verifies the time boundary. The inflected `used` token is treated as temporal framing; the location/value must still match owned evidence, the state/transition citation must resolve, and the predecessor boundary check remains mandatory.
- `llm_service.py`
  - The answer prompt now asks for concise, natural, direct answers and the shortest wording supported by the cited evidence, without unsupported geographic qualifiers or extra details.
- `tests/repro_live_failures.py`
  - Each scenario now uses a separate user namespace inside its temporary database/vector store. Its output includes state intervals, provenance IDs, and transition timestamps for debugging.
- `tests/test_api.py`, `tests/test_comprehensive_matrix.py`, `tests/test_hybrid_answers.py`
  - Added checks for API answer cleanup with retained citation IDs, the historical “used to” phrasing, and updated punctuation/fallback expectations.

### Verification performed
- Targeted: `python -m pytest -q tests/test_api.py tests/test_comprehensive_matrix.py -k "visible_answer or previous_location or previous_goal or goal_completion or location_move_is_typed or location_change or causal_paraphrases or unsupported_cause or preference or timing or process_chat"` -> **55 passed**.
- Final focused regression: `python -m pytest -q tests/test_comprehensive_matrix.py tests/test_hybrid_answers.py tests/test_api.py -k "predecessor or temporal or visible_answer or causal_answer_adds_recorded_destination or relevant_explicit_cause_passes_gate"` -> **40 passed**.
- Final full backend suite: `.\\.venv\\Scripts\\python.exe -m pytest -q --tb=short` -> **440 passed, 20 warnings**. Warnings are the existing Python 3.12 SQLite datetime-adapter deprecation.
- Frontend: `cd web; npm run build` -> **success** (TypeScript and Vite production build; 40 modules).
- Temporary-store live scenarios:
  - A, causal wording: the “why” and knee-injury questions cited the explicit cause transition and passed the gate. Both claims validated. The user was assigned a unique test namespace.
  - After the final wording change, replay of the knee-injury yes/no question returned “The change happened because of a knee injury. I then switched to swimming.”, with the explicit transition citation, gate `true`, and one accepted claim.
  - B, Python then Java: Java was returned for both current-preference phrasings; retrieval included only the active Java preference.
  - C, tea then coffee: coffee was returned as current; “Do I prefer tea?” safely abstained.
  - D, Chennai then Coimbatore: current location and direct change answered correctly. The initial old-location answer exposed the “used to” lexical rejection. Replaying that same temporary database after the fix returned “I used to live in Chennai”, with one accepted claim and the Chennai state citation; the answer string contained no raw source annotation.
  - E, goal plus completed machine-learning project: current goal and completion answered from typed GOAL/LIFE_EVENT states. The previous-goal question abstained because the scenario had no predecessor goal.
  - F, system-design goal replaced by data-structures goal: current and previous goal queries both returned the supported typed states.
- Isolated timing observations (milliseconds unless marked seconds):
  - Clean predecessor replay after the fix: **11.46 s total**, classification **3.73 s**, retrieval **3.80 s** (Mem0 search **3.79 s**), answer generation **3.93 s**, validation **0.2 ms**.
  - Final causal wording replay: **14.39 s total**, classification **6.25 s**, retrieval **3.76 s**, answer generation **4.38 s** (model **4.38 s**, validation **1.9 ms**, render **0.2 ms**).
  - Typical preference/location/goal answer model time was **5.3–7.2 s**; classification ranged **1.6–6.3 s**; Mem0 query search ranged **55 ms–2.25 s** (the predecessor replay above was a slower **3.79 s** sample).
  - Deterministic follow-up ingests in the isolated runs took **0.1–0.13 s**. First vector writes took **2.1–4.8 s**.
  - Free-form causal scenario ingests took **76–84 s**; Mem0's measured LLM extraction stage alone took **45.95–47.80 s**. Causal answers took **17.3–24.3 s**, with answer generation at **7.0–16.6 s** and retrieval at about **2.2 s**.

### Remaining issues
- Query latency remains dominated by the intent-classification and answer-generation model calls. No evidence-validation shortcut was introduced.
- Free-form ingestion remains slow when Mem0 performs its own LLM extraction; deterministic, locally typed messages avoid that pass after the initial embedding/vector-store cost.
- The live environment reported optional spaCy and fastembed packages missing. The checks still completed using the configured local vector retrieval path; BM25 keyword search was unavailable in that environment.
- The backend suite still emits the existing SQLite datetime-adapter deprecation warnings. Production legacy records remain unchanged and the read-only audit remains the mechanism for identifying them.


## Phase 8 - Retrieval completion, conflict reporting, and routing latency

### Root causes and changes
- `hybrid_retriever.py`
  - Added typed intent aliases for relocation endpoints and sport/exercise predecessors, plus goal-purpose phrasing and completed-project disambiguation.
  - Stored-preference inventory queries now carry a conflict summary only when at least two active, same-domain preference states have distinct values and belong to the requesting user. This special inventory route does not suppress conflict/ambiguity behavior for a question asking which preference is current.
  - The previous-place, current-goal, finished-project, and prior-activity retrieval routes remain backed by the SQLite state/transition records and linked memory IDs.
- `causal_graph.py` and `temporal_engine.py`
  - Added generic relocation/place and sport/exercise vocabulary to state-type recognition and predecessor/transition intent. Existing verified relation and boundary checks remain required; unknown relations and temporal-only evidence do not become causes.
- `query_router.py`
  - Clear personal-history questions now use deterministic routing for causal, predecessor, direct-transition, stored-preference inventory, completed-item, and explicit current-state/goal-purpose forms. Unclear questions continue to use the existing LLM classifier.
- `llm_service.py`
  - When the validated evidence bundle contains a verified stored-preference conflict, the model prompt asks it to name each alternative and state that they conflict, citing every listed preference state.
- `answer_generator.py`
  - Added a narrow group-claim validation path for a stored-preference conflict summary. It requires all and only the active, same-user preference state IDs in the conflict bundle, requires distinct recorded values and explicit conflict wording, requires each value to appear in the claim, and requires every state citation to be present in the model bundle. Ordinary preference claims still use the existing lexical validator.
  - Causal answers that omit one or both endpoints can be rendered into a concise sentence from the already validated explicit-cause transition and its same-user endpoints. Fully stated validated causal claims keep their wording and citation. Cause validation and citation acceptance happen before this rendering.
- `tests/test_comprehensive_matrix.py`, `tests/test_hybrid_answers.py`
  - Added regressions for relocation endpoints, place before relocation, current goal wording, completion wording, prior sport/exercise, stored programming-preference conflicts, and deterministic routing. Updated causal rendering expectations.

### Verification
- Focused retrieval/routing/conflict tests: **12 passed**.
- Focused causal rendering/prompt regressions: **6 passed**, followed by a rerun of the three issues found in the first full run; those focused reruns passed.
- Full backend suite: `.\\.venv\\Scripts\\python.exe -m pytest -q --tb=short` -> **453 passed, 20 warnings** (the existing Python 3.12 SQLite datetime-adapter deprecation).
- Isolated live location scenario: `.\\.venv\\Scripts\\python.exe tests\\repro_live_failures.py --scenario D` -> completed against a disposable SQLite database, Mem0/Qdrant directory, and generated test user. Current Coimbatore, prior Chennai, and the Chennai-to-Coimbatore statement each had one accepted claim; the move statement retained its temporal transition citation. The first run attempt was blocked by sandbox temp-folder permissions; the approved retry completed.
- Isolated live goal/project scenario: `.\\.venv\\Scripts\\python.exe tests\\repro_live_failures.py --scenario E` -> completed against a separate disposable SQLite/Mem0 store and generated test user. The current system-design goal and completed machine-learning project each had one accepted claim; the unsupported previous-goal question abstained before answer generation.
- No frontend files changed, so `npm run build` was not applicable to this backend retrieval repair.
- Production database and vector namespace were not selected by the isolated live script or the temporary-database test fixtures. No migration or production record rewrite was performed.

### Measured latency
- Direct classifier baseline before deterministic routing (same three queries): `Which city did I move from?` **4.913 s**, `What am I trying to learn now?` **1.632 s**, `What are my stored programming preferences?` **1.721 s**.
- After routing, warm medians across 25 runs were **0.007 ms**, **0.010 ms**, and **0.009 ms**, respectively. The initial cold import for the relocation classifier sample was **116 ms**; subsequent calls are reflected in the warm medians.
- The isolated goal/project scenario showed `What did I complete?` taking **1.704 s** for classification before its deterministic route was added. Its warmed direct classification median afterward was **0.007 ms**.
- The isolated live location scenario reported classification at **0.044-0.068 ms**. End-to-end query times were **8.21 s** for current location, **8.49 s** for previous location, and **8.67 s** for the location-change question. Answer generation took **6.31-8.14 s**; the previous-location Mem0 search took **2.17 s** (current-location Mem0 search: **66 ms**). These observations show that model generation and some Mem0 searches remain the principal latency costs; no claim-validation shortcut was introduced.
- Before the completed-item route was added, the isolated goal/project run measured **8.27 s** for the current-goal query (**8.17 s** generation), **9.70 s** for project completion (**5.83 s** generation, **2.15 s** Mem0 search), and **2.17 s** for the unsupported previous-goal abstention. Afterward, only the classifier path was remeasured directly; a live end-to-end replay of the same completion query was not performed.

### Remaining limitations
- Answer generation and occasional Mem0 retrieval still take seconds in the local live run. This repair reduced the measured intent-routing delay but did not optimize the answer model call.
- The live environment reports optional spaCy and fastembed packages missing, so BM25 keyword search is unavailable. The live location checks still completed using the configured vector retrieval path.


## Phase 9 - Live default-namespace regression investigation

### Live process and first-stage failures
- Port `127.0.0.1:8000` was serving the CausalMem API with `default_user=default`, Ollama `llama3.2:3b`, and the project `.env` paths (`data/causalmem.sqlite3`, `data/mem0`). The active process was started with `api.py` and did not reload edited Python modules. Its classifier traces and old `HISTORY`/`FACTUAL` routes differed from the current `query_router.py`. I stopped only the verified listener PID and relaunched the API using this project's `.venv` from the project directory. The final live replay used the restarted listener.
- Live read-only inspection showed that the underlying records exist but include legacy typing mismatches: the move statement and completed-project event are `OTHER`; the new goal is `OTHER` and has no linked state memory IDs; running is `INTEREST`; the Java programming preference is represented by a legacy memory/`INTEREST`, while Python is `PREFERENCE`.
- Before restart, origin/destination questions routed as `HISTORY`, did not retrieve the recorded transition, and claims citing a location state failed temporal validation. The previous-place query returned an unscoped history bundle without the location edge; its selected state claim failed temporal validation. The current-goal query retrieved the matching memory but the old answer bundle included many unrelated states; the claim was rejected for lexical support. The finished-project query retrieved an `OTHER` row and memories, then failed temporal validation. The activity question routed as `FACTUAL`, did not load the transition, and its claim failed temporal validation. Stored programming preferences routed as `PREFERENCE` and were blocked by ambiguity before answer generation; the evidence did not represent the Java and Python sources as one same-domain conflict. In each of those seven cases, the visible API answer was the generator's abstention. Both causal safety queries retained their intended behavior after restart.
- The shared repair keeps matching owned memory evidence in a narrow typed fallback when legacy extraction has no state of the requested type; the answer layer exposes only that fallback evidence to the model and validates any fallback statement through the standard ownership/citation/lexical checks. Direct from/to fallbacks require one known relation, one selected transition, a typed endpoint request, and the original transition citation validator. Stored preference conflicts can combine explicit domain-matched legacy source memories and typed preference states; the answer must cite every source and state every value. Ordinary current-preference ambiguity remains fail-closed.

### Final live API verification
- Ran all nine exact questions through `POST /api/chat` at `http://127.0.0.1:8000` with `user_id=default` on the restarted API. All seven retrieval checks returned one accepted claim and a citation:
  - `Which city did I move from?` -> Chennai; cited the `TEMPORAL_SUCCESSION` relocation transition.
  - `Which city did I move to?` -> Coimbatore; cited the same transition.
  - `Which place was I living in before my relocation?` -> Chennai; cited the historical Chennai state.
  - `What am I trying to learn now?` -> system design; cited the owned goal memory.
  - `Which project have I finished?` -> machine learning project; cited the owned completion memory.
  - `What sport or exercise was I doing before swimming?` -> running; cited the historical running state.
  - `What are my stored programming preferences?` -> Java and Python conflict; cited both exact programming-preference memories and did not choose a definitive current preference.
- `Why did I move from Chennai to Coimbatore?` -> abstained with `missing_explicit_cause_transition`, no citations.
- `Why did I switch from running to swimming?` -> answered with the knee injury and cited the owned `EXPLICIT_CAUSE` transition.
- The live verification asserted that every citation kind/ID appeared in that response's evidence bundle, the accepted/rejected claim counts matched the expected behavior, and unsupported relocation cause produced no citation. The Chat page retains the API response and `EvidencePage` renders that response's citations, evidence IDs, and citation-validation checks. `npm run build` succeeded and generated the same JS/CSS hashes currently served by the API (`index-CVlK02HG.js`, `index-DJ3xz6v7.css`).
- Default-namespace counts were identical before and after the nine read-only API calls: **20 memories, 17 states, 14 active states, 3 transitions, 1 explicit-cause transition, 1 temporal-succession transition**. `data/causalmem.sqlite3` retained its prior last-write timestamp; no memory/state/transition writes or migrations were performed by these verification requests.

### Final checks and timings
- Final focused regressions for legacy `OTHER`-state memory fallbacks, relocation fallback, conflict reporting, and retrieved-transition paraphrases: **10 passed**.
- Exact non-causal multi-hop safety regression after tightening the fallback: **3 passed**.
- Final full backend suite: `.\\.venv\\Scripts\\python.exe -m pytest -q --tb=short` -> **456 passed, 20 existing SQLite datetime-adapter deprecation warnings**.
- Frontend production build: `(cd web; npm run build)` -> **success**, TypeScript build and Vite production bundle (41 modules).
- Final live API timings (seconds; classification/retrieval/generation in milliseconds):
  - Move from: **9.83 s** total; **0.06 / 2,189 / 7,640 ms**.
  - Move to: **8.02 s**; **0.07 / 2,138 / 5,886 ms**.
  - Previous place: **8.42 s**; **0.08 / 2,173 / 6,244 ms**.
  - Current goal: **7.15 s**; **0.05 / 2,158 / 4,988 ms**.
  - Finished project: **7.26 s**; **0.08 / 2,132 / 5,124 ms**.
  - Previous activity: **8.51 s**; **0.08 / 2,131 / 6,375 ms**.
  - Stored programming conflict: **12.98 s**; **0.09 / 2,135 / 10,845 ms**.
  - Unsupported move cause: **2.13 s**; **0.05 / 2,130 / 0.1 ms** (answer generation skipped).
  - Explicit swimming cause: **7.43 s**; **0.04 / 79 / 7,347 ms**.
- The slow path remains LLM answer generation (about **5.0-10.8 s** for the supported answers); intent routing is now deterministic for these clear query forms (**0.04-0.09 ms**). Retrieval of the preference/legacy records spends about **2.1 s** in Mem0 search. Optional spaCy and fastembed are absent, and BM25 remains unavailable.

