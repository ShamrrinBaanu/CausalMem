# CausalMem: An Evidence-Linked Temporal Memory Layer for Conversational AI

**Research report draft**  
**Project evidence reviewed:** source code, tests, `README.md`, `evaluation/README.md`, versioned evaluation datasets, and the archived six-case and single-case diagnostic artifacts.  
**Evidence status:** historical run results below are reported from the saved artifacts and project documentation; no tests or evaluations were rerun for this draft.

## 1. Abstract

Long-running conversational assistants need to retain useful user information, revise it when the user reports a change, and answer questions about what was true at an earlier time. CausalMem is a local research prototype that extends the open-source Mem0 conversational-memory framework with a separate SQLite representation of typed user states and transitions. Its retrieval path combines user-scoped Mem0 records with current or historically valid SQLite states and relevant transition evidence. The answer path validates model-proposed claims against evidence references, ownership, temporal validity, and relation semantics, and can abstain when support is insufficient.

This report describes the implementation and evaluates the available evidence without treating the prototype as a validated causal inference system. Deterministic tests exercise policy and data-flow behavior with fixtures and mocks. A recorded six-case live evaluation using synthetic cases completed with all six cases passing the configured answer and proxy checks. Its exact evidence-key, substring-fact, citation-group, transition-provenance, and causal-support scores are proxies, not semantic entailment measures. A separate single-case latency diagnostic took 215.384 seconds. Earlier logs also record an anomalous multi-hour ingestion duration whose cause remains unknown. The results motivate further replicated, independently reviewed evaluation; they do not establish general reliability or superiority over Mem0 or other systems.

## 2. Introduction and Problem Statement

Language-model assistants do not inherently maintain a reliable, inspectable record of user information across separate conversations. Persistent memory systems address this limitation by extracting and retrieving selected information, while long-term conversational-memory research has also emphasized evaluation across extended interactions [1, 3, 4]. Yet a memory system that stores current facts alone may not answer questions about replaced goals or preferences. Chronological order also does not, by itself, establish why a change happened.

CausalMem investigates a layered design for these needs. Mem0 OSS remains responsible for persistent conversational memories and semantic retrieval; a separate SQLite store records typed states, their validity intervals, and links between state changes. The system aims to make changes inspectable and to distinguish a user's recorded explanation from mere temporal succession. It is a prototype implementation, not a formal causal discovery method and not an authenticated multi-user service.

## 3. Research Questions and Objectives

The project documentation frames the following questions as open research questions, not established findings:

1. Can Mem0 be extended with explicit temporal state transitions while preserving earlier states for historical retrieval?
2. Can explicitly recorded reasons for changes be distinguished from chronological succession without treating sequence as causation?
3. How reliably do the current retrieval and validation policies support as-of queries and abstention on the available synthetic cases?
4. What performance costs occur in this local Ollama configuration, and which pipeline stages account for them?

The immediate objectives are to preserve persistent conversational memory, record state changes with source attribution, retrieve states for a requested recorded time, and prevent unsupported claims from being presented as evidence-backed answers. The available experiments only provide preliminary evidence for these objectives.

## 4. Related Work

Persistent conversational memory is motivated by the gap between bounded interaction context and conversations that continue over multiple sessions. Mem0 describes a memory architecture that extracts, consolidates, and retrieves salient information, and reports evaluations on the LoCoMo benchmark [1]. Mem0 OSS documentation describes a self-hostable library whose LLM, embedder, vector store, and other components can be configured [2]. MemoryBank is another approach to long-term memory for language models, using retrieval and memory updates [4]. LoCoMo provides a benchmark and dataset for very long-term conversational memory, including question answering and temporal reasoning tasks [3]. These works supply context for CausalMem's motivation; their results are not results for CausalMem.

Retrieval-augmented generation (RAG) combines generated language with retrieved, non-parametric information [8]. Work on factual consistency and faithfulness shows why generated text should not automatically be treated as faithful to source material [9, 10]. Research on unanswerable questions likewise frames abstention as a necessary behavior when the provided context does not support an answer [11]. RAGAs presents a framework for evaluating retrieval-augmented generation along retrieval and generation dimensions [12]. CausalMem uses its own deterministic validation policy and proxy metrics; it does not implement or claim the RAGAs metric suite.

For temporal data, time-oriented database literature distinguishes valid-time and transaction-time concepts and studies state tables and interval queries [5]. System-versioned temporal database documentation provides a concrete example of reconstructing data as of a specified instant [6]. CausalMem's SQLite model is narrower: its state intervals use recorded timestamps, and it does not provide a general bitemporal database model.

Formal causal inference makes causal claims conditional on assumptions and a causal model [7]. CausalMem's `EXPLICIT_CAUSE` relation instead records an explanation explicitly provided in user text and checks that the answer cites that recorded evidence. This captures a textual attribution; it does not estimate causal effects or prove that the stated cause is scientifically correct. A `TEMPORAL_SUCCESSION` edge records order and is not treated as evidence of causation.

## 5. CausalMem Architecture and Implementation

The implementation separates conversational memory from explicit state history. The major components and responsibilities are summarized below.

| Component | Implementation role | Evidence source |
|---|---|---|
| Mem0 OSS | Persistent conversational-memory writes and semantic search through the installed Mem0 package | `mem0_engine.py`, `README.md`; Mem0 OSS docs [2] |
| Local model and embeddings | The documented local configuration uses Ollama with `llama3.2:3b` for the chat model and `nomic-embed-text` for embeddings | `config.py`, `mem0_engine.py`, `evaluation/README.md` |
| SQLite | Stores typed `State` and `Transition` records, validity timestamps, status, and source memory IDs | `models.py`, `database.py` |
| Ingestion/extraction | Extracts and grounds candidate states, recognizes supported changes, and writes state/transition updates | `state_extractor.py`, `transition_engine.py` |
| Retrieval | Routes the question, retrieves user-scoped Mem0 and SQLite evidence, and adds relevant transition endpoints where appropriate | `query_router.py`, `hybrid_retriever.py`, `causal_graph.py`, `temporal_engine.py` |
| Answer validation | Validates structured model claims and references against retrieved evidence, ownership, temporal rules, and causal relation semantics | `llm_service.py`, `answer_generator.py` |
| Interface | Provides the local Streamlit interaction and memory/state/graph views | `app.py`, `visualization.py` |
| Evaluation | Provides deterministic policy evaluation and isolated live-pipeline evaluation | `research_evaluator.py`, `research_live_evaluator.py`, `evaluation/` |

The conversation flow is sequential across persistence systems. The app writes a message to Mem0, then sends it through state extraction and SQLite ingestion. The SQLite state, transition, and source-state-closure writes for one logical ingestion operation share a SQLite transaction and roll back together on an exception. Mem0 and SQLite do **not** share a distributed or cross-store transaction; a successful Mem0 write can remain even if later extraction or SQLite persistence fails.

User IDs act as storage namespaces. They are passed through retrieval and database operations, but neither the React console nor the HTTP API authenticates users or verifies that a person is entitled to choose a namespace. They therefore do not constitute authentication or authorization. The project is intended for trusted local use, not hosted multi-tenant deployment.

## 6. Temporal State Representation and As-of Retrieval

`models.py` defines state records with a user ID, type, value, status, `valid_from`, `valid_until`, confidence, source memory IDs, and metadata. Transition records identify a source and destination state, relation kind, trigger/reason, timestamp, confidence, and source memory IDs. `database.py::states_at` first filters rows by `user_id`, parses timestamps as timezone-aware values, normalizes them to UTC, and compares them at microsecond precision in Python rather than relying on SQLite `julianday` rounding.

Validity uses the half-open interval `valid_from <= t < valid_until`.

The start is inclusive and the end is exclusive; a null end represents an open interval. Thus, at a replacement boundary, the predecessor is no longer valid and the successor is valid, assuming matching boundary timestamps. Null or malformed stored start times are excluded, and malformed non-null end times are excluded. Requested timestamps must be valid, timezone-aware ISO timestamps; naive or malformed requests are rejected rather than silently interpreted as current-state queries. `temporal_engine.py` supports a conservative set of natural UTC date forms and predecessor queries, with ambiguity handled by returning no unique predecessor rather than choosing arbitrarily.

These timestamps represent when CausalMem recorded a state/transition and its validity in the application's recorded history. They are not automatically parsed real-world event times. The code and tests cover interval boundaries, user scoping, state replacement, predecessor retrieval, and future-transition filtering (`tests/test_core.py`, `tests/test_ingestion_retrieval_integration.py`, and `tests/test_hybrid_answers.py`). The semantics should therefore be described as recorded-time history, not as a complete event-time or bitemporal model [5, 6].

## 7. Explicit-Cause Evidence, Grounding, and Abstention

`state_extractor.py::ingest_message` rejects clear questions before persistence, grounds extracted state candidates in the message, and distinguishes supported replacements from additive interests. A user statement that gives a reason may create an `EXPLICIT_CAUSE` transition with recorded trigger/reason evidence. A replacement without an explicitly stated reason can create `TEMPORAL_SUCCESSION`, which expresses ordering only. Historical states remain available through their validity interval.

The answer path asks the model for structured claims with references to supplied memory, state, or transition IDs. `answer_generator.py` then validates claim structure, reference resolution, same-user ownership, lexical or linked-source support, temporal validity, ambiguity, and causal relation requirements. A causal question requires a relevant `EXPLICIT_CAUSE` transition with recorded cause evidence and valid same-user endpoints; temporal succession, unknown, and inferred relations do not satisfy this causal gate. At an as-of time, future transitions cannot justify a past causal answer. Unsupported or malformed claims can be rejected and the system can abstain. Prompt instructions guide the model but are not the safety mechanism.

These safeguards validate whether the recorded evidence meets the system's rules. They do not prove that a user's stated reason is true, that the text captures every nuance, or that generated prose is semantically entailed by the records. Causal inference literature emphasizes assumptions in causal claims [7]; factual-consistency research likewise motivates verification beyond generation prompts [9, 10]. Abstention is a recognized response to context that does not support an answer [11]. CausalMem's own lexical and identifier checks are narrower than a general semantic entailment verifier.

## 8. Experimental Setup and Evaluation Protocol

The project has distinct forms of evidence. They should not be combined into a single score.

| Evaluation evidence | Data and execution | What it tests | What it does not establish |
|---|---|---|---|
| Deterministic policy tests | Pytest fixtures, temporary SQLite databases, and mocked Mem0/model responses | Persistence, ownership, transition rules, timestamp boundaries, retrieval/answer validation, failure handling, and selected ingestion flows | Live model quality, production behavior, broad retrieval quality, or user-level usefulness |
| Research policy evaluator | Versioned `evaluation/research_v1.jsonl`, fixed evidence bundles and fixed structured model responses | Deterministic claim policy, answerability labels, evidence-ID agreement, and related metric calculations | Live LLM outputs, live Mem0 ranking, or end-user semantic correctness |
| Six-case live-pipeline evaluation | `evaluation/live_research_v1.jsonl`; isolated Mem0/Qdrant/SQLite paths; live Ollama, retrieval, and answer generation | End-to-end behavior on six synthetic cases, including three as-of cases | Generalization, statistical reliability, or a paired live Mem0-only comparison |
| Single-case latency diagnostic | One isolated `explicit-cause` run with content-free stage timing instrumentation | Observed stage and overall elapsed times on one local configuration | A quality result, stable performance estimate, or causal diagnosis of the latency |

The live evaluator uses fresh isolation paths for SQLite, Mem0/Qdrant, Mem0 configuration, progress, and result output. The archived six-case artifact records Ollama and the hybrid retrieval/answer path as live. Dataset expectations are attached to the saved result after prediction; they are not included in model evidence. The progress JSONL is allowlisted and redacted, but the final result intentionally contains predictions and evidence and is sensitive. Results and stores should remain isolated and should not be shared without inspection.

The six-case dataset is synthetic and includes direct fact, explicit cause, historical predecessor, and before/at/after replacement cases. Evaluation metrics include exact fixture-key evidence agreement, substring-based required/forbidden fact checks, exact citation-group mapping, temporal state fixture-key matches, and causal evidence proxies. Each measures a limited observable. None establishes semantic entailment.

| Six-case live-run configuration | Recorded value |
|---|---|
| Dataset | `evaluation/live_research_v1.jsonl`, dataset version 1.0 |
| Evaluator and result schema | Evaluator version 1.0; schema version 1.0 |
| Case IDs (recorded order) | `direct-fact`, `historical-predecessor`, `asof-before-replacement`, `asof-at-replacement`, `asof-after-replacement`, `explicit-cause` |
| Run date | Console timestamps: 2026-10-23 to 2026-10-24 UTC; the result JSON has no single run-date field |
| Random seed | 0 |
| Provider and models | Ollama; `llama3.2:3b`; embedding model `nomic-embed-text` |
| Recorded package versions | Python 3.12.10; `mem0ai` 2.2.1; `qdrant-client` 1.19.1; `openai` 3.24.0; `streamlit` 1.65.0 |
| Component status | SQLite isolated; Qdrant/Mem0, Ollama LLM, hybrid retrieval, and answer generation live |
| Hardware | Not recorded in the six-case result artifact |
| Reproduction | Follow `evaluation/README.md` under `One-case isolated live evaluation` with a fresh isolation root; use `--dataset .\evaluation\live_research_v1.jsonl` and `--case-ids direct-fact historical-predecessor asof-before-replacement asof-at-replacement asof-after-replacement explicit-cause`, and give `--output` a new file inside that root |

The historical record in `evaluation/README.md` reports the full pytest suite as 194 passed, 0 failed, 0 skipped, with eight existing SQLite datetime-adapter deprecation warnings. This report audit did not rerun the suite; no raw pytest transcript is claimed here.

## 9. Results and Discussion

The archived final run at `.tmp/phase5222-quality-final-6ae7c0522e5d458d920ac5587c31d218/` completed six of six selected synthetic cases, with zero failed or skipped cases. The result artifact records live Ollama, Mem0/Qdrant, retrieval, and answer generation. Its principal metrics are summarized below.

| Measure | Recorded result | Interpretation |
|---|---:|---|
| Cases | 6 attempted; 6 completed; 0 failed; 0 skipped | Run completed; completion is not semantic validation |
| Evidence retrieval | Precision 1.00, recall 1.00, F1 1.00; 8 true-positive fixture keys, 0 false positives, 0 false negatives | Exact fixture-key mapping proxy |
| Answer/abstain behavior | 6/6 (100%) | Exact expected behavior label agreement on this subset; all six expected answers were answered |
| Required facts | 7/7 found; forbidden facts 0/4 found | Case-insensitive substring proxy, not entailment |
| Citation groups | 6/6 agreement | Exact mapping of citations to fixture-key source groups, not semantic support |
| As-of state retrieval | 3/3 exact state-key matches | End-to-end state keys returned by the actual retriever; three cases only |
| Transition provenance | 1/1 | Exact fixture-key provenance proxy for one annotated case |
| Causal evidence | 1/1; unsupported causal assertion proxy 0/1 | Single annotated explicit-cause case; lexical/gate-based proxy |
| Latency | Mean 26.90 s total query pipeline; 5.38 s retrieval; 21.52 s answer generation | Excludes fixture ingestion; p95 unavailable because the evaluator requires at least 20 observations |

The final run was preceded by a six-case run in which `asof-after-replacement` abstained after temporal lexical support failed. The later run passed the configured checks, but the earlier intermittent failure remains relevant: six successful cases do not establish that the issue cannot recur. The results support the narrower statement that this recorded run passed these proxy checks on these synthetic cases. They do not support a conclusion that CausalMem is better than Mem0 or another system.

The archived JSON reports abstention precision and recall as undefined because this six-case set contained no expected abstentions; those metrics should not be converted to zero or treated as measured. Its overall accuracy and evidence-support fields are also not applicable in that report format. Three as-of retrieval cases are reported separately from the oracle-only interval tests; oracle-only checks are not counted as end-to-end temporal successes.

## 10. Performance Analysis

The archived progress JSONL records a 211.415-second case duration for the isolated single-case `explicit-cause` diagnostic. The console's structured events span 214.412 seconds from first to last event; these are different timing boundaries, not conflicting measurements of one quantity. The available archived files do not substantiate the 215.384-second figure that `evaluation/README.md` describes as the one-case completion/total time; the README does not define that timing boundary, so this report does not treat it as a verified measurement. The recorded Mem0 LLM memory-extraction stages took 50.981 seconds and 50.978 seconds; answer generation took 45.703 seconds. Embedding generation, vector search, and vector insertion were reported as substantially shorter, and Ollama reported CPU execution. This is one observation on one local configuration, not a replicated performance estimate. It does not establish that CPU execution, context size, or a particular model operation caused the latency.

The evaluation documentation also reports an explicit-cause fixture-ingestion delay of approximately 4.28 hours on the progress clock during the six-case run, despite normal query latency. That anomaly is distinct from the single-case diagnostic durations above. Its root cause was not established; the available observations do not justify attributing it to one provider call, a timestamp-accounting error, or a specific model stage. Neither result is a quality metric. No paired hardware/model/context experiment or repeated performance run is available.

## 11. Limitations and Threats to Validity

1. **Small synthetic sample.** Six live cases are too few and too controlled to establish general reliability, robustness, or performance across users and conversation styles.
2. **Proxy metrics.** Fixture-key overlap and citation-group agreement measure exact attribution mappings; required/forbidden substring checks measure lexical occurrence. These do not establish that a claim is semantically correct or entailed.
3. **No paired live baseline.** The recorded six-case run does not provide a same-query, same-condition live Mem0-only baseline, so comparative superiority cannot be assessed.
4. **Temporal scope.** Stored timestamps represent recorded state/transition history rather than necessarily the event's real-world time. Natural-date interpretation is intentionally limited. The data model is not a general bitemporal system.
5. **Intermittent support failure.** An earlier as-of case failed lexical support before a later successful six-case run. The subsequent result does not prove permanent resolution.
6. **Causal interpretation.** `EXPLICIT_CAUSE` captures recorded user attribution under application rules. It does not perform statistical causal inference or independently validate the truth of the attribution.
7. **Cross-store consistency.** SQLite changes are transactional, but Mem0 writes occur separately and sequentially. A Mem0 record can persist without a corresponding SQLite state or transition if later processing fails.
8. **Identity and privacy.** User IDs are namespaces, not authenticated identities. Final result JSON contains predictions/evidence and must be treated as sensitive even when generated from synthetic cases. The redacted progress log does not make every console or UI surface content-free.
9. **Performance evidence.** The archived single-case diagnostic records a 211.415-second case duration and a 214.412-second console-event span; process wall time is not established. A separate multi-hour ingestion anomaly remains unexplained. There are no replicated performance measurements; p95 was unavailable with six queries.
10. **Reproducibility.** The repository does not pin all dependencies or specify a tested Python compatibility matrix. The documented environment snapshot is not a lockfile or compatibility guarantee.
11. **Test scope.** The reported 194-test suite is primarily deterministic and mock-based. Passing policy tests do not establish live extraction quality, real-world answer correctness, or safe hosted multi-user operation.

## 12. Conclusion and Future Work

CausalMem implements a local prototype that combines Mem0 conversational memory with a user-scoped SQLite history of typed states and transitions. Its half-open validity intervals support as-of retrieval; its transition types distinguish recorded explicit cause from temporal succession; and its answer validator checks claim references and evidence conditions before presenting model-generated claims. These are implementation properties supported by code and tests, not evidence that the system has solved long-term memory or causal reasoning.

The archived six-case live run is encouraging only within its narrow synthetic and proxy-based scope. The earlier temporal lexical-support failure, single-run CPU latency observations, and unexplained multi-hour ingestion anomaly argue for further testing. Future work should include repeated isolated runs, a paired live Mem0-only baseline, larger and independently reviewed datasets, human assessment of semantic support and abstention, robustness testing on diverse phrasing, controlled latency experiments, improved cross-store reconciliation, and authenticated identity if multi-user deployment is intended. These are proposals, not completed experiments.

## 13. References

[1] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. “Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory.” In *ECAI 2025: 28th European Conference on Artificial Intelligence*, vol. 413, *Frontiers in Artificial Intelligence and Applications*, pp. 2993–3000. IOS Press, 2025. DOI: [10.3233/FAIA251160](https://doi.org/10.3233/FAIA251160). Published conference paper; preprint: [arXiv:2504.19413](https://arxiv.org/abs/2504.19413).

[2] Mem0. “Open Source Overview” and “Configure the OSS Stack.” Official project documentation, n.d. [Overview](https://docs.mem0.ai/open-source/overview); [configuration](https://docs.mem0.ai/open-source/configuration).

[3] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. “Evaluating Very Long-Term Conversational Memory of LLM Agents.” In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 13851–13870, 2024. DOI: [10.18653/v1/2024.acl-long.747](https://doi.org/10.18653/v1/2024.acl-long.747). Peer-reviewed conference paper.

[4] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. “MemoryBank: Enhancing Large Language Models with Long-Term Memory.” In *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(17), pp. 19724–19731, 2024. DOI: [10.1609/aaai.v38i17.29946](https://doi.org/10.1609/aaai.v38i17.29946). Peer-reviewed conference paper.

[5] Richard T. Snodgrass. *Developing Time-Oriented Database Applications in SQL*. 1st ed. Morgan Kaufmann, 1999. ISBN 978-0-08-050422-3. Scholarly technical book. [Publisher record](https://shop.elsevier.com/books/developing-time-oriented-database-applications-in-sql/snodgrass/978-0-08-050422-3).

[6] Microsoft. “Query Data in a System-Versioned Temporal Table.” *Microsoft Learn*, n.d. Official technical documentation. [URL](https://learn.microsoft.com/en-us/sql/relational-databases/tables/temporal/query-data?view=sql-server-ver17).

[7] Judea Pearl. “Causal Inference in Statistics: An Overview.” *Statistics Surveys* 3 (2009): 96–146. DOI: [10.1214/09-SS057](https://doi.org/10.1214/09-SS057). Peer-reviewed survey article.

[8] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.” In *Advances in Neural Information Processing Systems 33*, pp. 9459–9474, 2020. Peer-reviewed conference paper. [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2020/hash/6b493230-Abstract.html).

[9] Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. “On Faithfulness and Factuality in Abstractive Summarization.” In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pp. 1906–1919, 2020. DOI: [10.18653/v1/2020.acl-main.173](https://doi.org/10.18653/v1/2020.acl-main.173). Peer-reviewed conference paper.

[10] Wojciech Kryscinski, Bryan McCann, Caiming Xiong, and Richard Socher. “Evaluating the Factual Consistency of Abstractive Text Summarization.” In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing*, pp. 9332–9346, 2020. DOI: [10.18653/v1/2020.emnlp-main.750](https://doi.org/10.18653/v1/2020.emnlp-main.750). Peer-reviewed conference paper.

[11] Pranav Rajpurkar, Robin Jia, and Percy Liang. “Know What You Don’t Know: Unanswerable Questions for SQuAD.” In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pp. 784–789, 2018. DOI: [10.18653/v1/P18-2124](https://doi.org/10.18653/v1/P18-2124). Peer-reviewed conference paper.

[12] Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. “RAGAs: Automated Evaluation of Retrieval Augmented Generation.” In *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations*, pp. 150–158, 2024. DOI: [10.18653/v1/2024.eacl-demo.16](https://doi.org/10.18653/v1/2024.eacl-demo.16). Peer-reviewed conference system-demonstration paper.

### Final note: unverified claims and evidence needed

- The six-case result does not establish broad semantic answer quality, generalization, or a permanent fix for temporal lexical-support failures. Larger, independently annotated and human-reviewed evaluations are needed.
- No paired live Mem0-only comparison is recorded; a controlled same-query comparison is required before making comparative effectiveness claims.
- The cause of the approximately 4.28-hour ingestion anomaly is unknown. Repeated instrumented runs and controlled stage/model/context measurements are needed to diagnose it.
- The single-case diagnostic has not been replicated; its archived case duration is 211.415 seconds and its console-event span is 214.412 seconds, while process wall time is not established. More runs and controlled hardware/model conditions are needed for stable latency estimates.
- The reported 194-test result is documented historical evidence and was not rerun while drafting this report.
- CausalMem has no scholarly publication in the supplied project materials. The citations above provide background, not external validation of CausalMem.
