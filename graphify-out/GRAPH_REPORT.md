# Graph Report - .  (2026-05-31)

## Corpus Check
- Corpus is ~29,732 words - fits in a single context window. You may not need a graph.

## Summary
- 852 nodes · 2212 edges · 65 communities (50 shown, 15 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 188 edges (avg confidence: 0.56)
- Token cost: 0 input · 238,439 output

## Community Hubs (Navigation)
- [[_COMMUNITY_AutoGen & LangGraph Adapters|AutoGen & LangGraph Adapters]]
- [[_COMMUNITY_Storage Backends & Serialization|Storage Backends & Serialization]]
- [[_COMMUNITY_OpenInferenceOTel Adapters & Registry|OpenInference/OTel Adapters & Registry]]
- [[_COMMUNITY_Agno Adapter & Tracing|Agno Adapter & Tracing]]
- [[_COMMUNITY_CLI Convert Pipeline|CLI Convert Pipeline]]
- [[_COMMUNITY_BaseAdapter Buffering Core|BaseAdapter Buffering Core]]
- [[_COMMUNITY_CrewAI Adapter|CrewAI Adapter]]
- [[_COMMUNITY_Adapter Test Suite & Canonical Model|Adapter Test Suite & Canonical Model]]
- [[_COMMUNITY_Atomic Agents Adapter|Atomic Agents Adapter]]
- [[_COMMUNITY_LangGraph Adapter|LangGraph Adapter]]
- [[_COMMUNITY_MCP Adapter|MCP Adapter]]
- [[_COMMUNITY_AgentOps Adapter|AgentOps Adapter]]
- [[_COMMUNITY_MLflow Adapter|MLflow Adapter]]
- [[_COMMUNITY_CLI Entrypoint & Integration Tests|CLI Entrypoint & Integration Tests]]
- [[_COMMUNITY_Streaming Event Normalization|Streaming Event Normalization]]
- [[_COMMUNITY_Canonical Serialization|Canonical Serialization]]
- [[_COMMUNITY_OpenInferenceOTel Package Exports|OpenInference/OTel Package Exports]]
- [[_COMMUNITY_CrewAI Converters & Tests|CrewAI Converters & Tests]]
- [[_COMMUNITY_MCP JSON-RPC Converters|MCP JSON-RPC Converters]]
- [[_COMMUNITY_AutoGenCrewAI Package Init|AutoGen/CrewAI Package Init]]
- [[_COMMUNITY_Adapter Convention Tests|Adapter Convention Tests]]
- [[_COMMUNITY_MLflow Converters & Tests|MLflow Converters & Tests]]
- [[_COMMUNITY_LangGraph Package|LangGraph Package]]
- [[_COMMUNITY_MCP Package|MCP Package]]
- [[_COMMUNITY_MLflow Package|MLflow Package]]
- [[_COMMUNITY_Optional Adapter Packages|Optional Adapter Packages]]
- [[_COMMUNITY_Record Normalization|Record Normalization]]
- [[_COMMUNITY_Canonical Data Model|Canonical Data Model]]
- [[_COMMUNITY_Multi-Format Export|Multi-Format Export]]
- [[_COMMUNITY_Multi-Backend Storage|Multi-Backend Storage]]
- [[_COMMUNITY_Convert CLI Command|Convert CLI Command]]
- [[_COMMUNITY_Agno from_trace|Agno from_trace]]
- [[_COMMUNITY_Agno Run Output Parser|Agno Run Output Parser]]
- [[_COMMUNITY_Atomic from_agent_run|Atomic from_agent_run]]
- [[_COMMUNITY_Utils Package Init Tests|Utils Package Init Tests]]
- [[_COMMUNITY_Metadata Helper Tests|Metadata Helper Tests]]
- [[_COMMUNITY_Integration Convention Tests|Integration Convention Tests]]

## God Nodes (most connected - your core abstractions)
1. `CanonicalInteraction` - 118 edges
2. `get_value()` - 74 edges
3. `object_to_dict()` - 43 edges
4. `compact_dict()` - 38 edges
5. `CanonicalMessage` - 37 edges
6. `message_to_canonical()` - 35 edges
7. `from_run_output()` - 34 edges
8. `BaseAdapter` - 32 edges
9. `str` - 31 edges
10. `Formatter` - 30 edges

## Surprising Connections (you probably didn't know these)
- `AgnoAdapter` --references--> `AgentScribe Project Overview`  [INFERRED]
  agentscribe/adapters/agno/agno.py → README.md
- `_looks_gzipped()` --calls--> `Path`  [INFERRED]
  agentscribe/cli.py → tests/integrations/conftest.py
- `_looks_like_jsonl()` --calls--> `Path`  [INFERRED]
  agentscribe/cli.py → tests/integrations/conftest.py
- `Payload` --uses--> `CanonicalInteraction`  [INFERRED]
  tests/adapters/utils/test_normalization.py → agentscribe/core/canonical.py
- `MCP JSON-RPC Pairing by id` --conceptually_related_to--> `from_jsonrpc_messages()`  [INFERRED]
  agentscribe/adapters/mcp/mcp.md → agentscribe/adapters/mcp/mcp.py

## Hyperedges (group relationships)
- **Framework Adapters Normalize to Canonical Interaction** — agno_agno_from_run_output, crewai_crewai_from_llm_call_context, autogen_autogen_from_task_result, atomic_agents_atomic_agents_from_agent_response, agentops_agentops_from_trace [INFERRED 0.85]
- **BaseAdapter Buffering and Flush Flow** — base_baseadapter__finalise_and_flush, base_baseadapter__flush_buffer, base_baseadapter_flush [EXTRACTED 1.00]
- **Agno Capture Mechanisms** — agno_agno_mlflow_autolog, agno_agno_agentos_tracing, agno_agno_post_hooks, agno_agno_tool_hooks [INFERRED 0.75]
- **OpenTelemetry-based shared span parsing across adapters** — opentelemetry_opentelemetry_from_spans, mlflow_mlflow_from_trace, openinference_openinference_from_spans [INFERRED 0.85]
- **Normalization helpers building canonical messages** — utils_normalization_message_to_canonical, utils_normalization_interaction_from_messages, utils_normalization_append_unique_message, utils_normalization_infer_message_role [INFERRED 0.75]
- **Registry dispatching CLI records to per-framework adapters** — utils_registry_adapter_record_to_interactions, utils_registry_adapter_record_loaders, langgraph_langgraph_from_state, mcp_mcp_from_jsonrpc_messages, mlflow_mlflow_from_trace [INFERRED 0.85]
- **Canonical message format dispatch** — cli__format_messages, cli__to_openai_message, cli__to_sharegpt, cli__to_alpaca, cli__to_prompt_completion, cli__to_preference [EXTRACTED 1.00]
- **Storage backend registry resolution** — storage_register_backend, storage_get_backend, storage_localstoragebackend, storage_fsspecstoragebackend, storage_postgresstoragebackend [EXTRACTED 1.00]
- **CLI convert pipeline load-format-write** — cli_convert, cli__load_records, cli__format_records, storage_write_jsonl [EXTRACTED 1.00]
- **Adapter Package __all__ Re-export Tests** — autogen_test_init, crewai_test_init, langgraph_test_init, mcp_test_init, mlflow_test_init, openinference_test_init, opentelemetry_test_init [INFERRED 0.85]
- **Streaming Event Normalization Test Pattern** — autogen_test_autogen, crewai_test_crewai, langgraph_test_langgraph [INFERRED 0.75]
- **Trace/Span Based Adapter Test Pattern** — mlflow_test_mlflow, openinference_test_openinference, opentelemetry_test_init [INFERRED 0.75]
- **CLI Convert to Local Storage Flow** — integrations_test_cli_convert, cli_main, registry_adapter_record_to_interactions, storage_write_jsonl [INFERRED 0.85]
- **Adapter Test Convention Enforcement** — adapters_test_adapter_conventions, adapters_test_init, integrations_test_integration_conventions [INFERRED 0.75]
- **Agno Capture-Format-Persist Pipeline** — agno_integration_agnoadapter_local_save, agno_from_run_output, formatter_formatter, storage_write_jsonl [INFERRED 0.85]

## Communities (65 total, 15 thin omitted)

### Community 0 - "AutoGen & LangGraph Adapters"
Cohesion: 0.07
Nodes (81): Any, bool, CanonicalInteraction, CanonicalMessage, str, Any, bool, CanonicalInteraction (+73 more)

### Community 1 - "Storage Backends & Serialization"
Cohesion: 0.07
Nodes (64): ABC, test_every_adapter_python_file_has_mirrored_unit_test(), _batched(), FSSpecStorageBackend, get_backend(), _import_fsspec(), _is_replace_mode(), _is_write_mode() (+56 more)

### Community 2 - "OpenInference/OTel Adapters & Registry"
Cohesion: 0.06
Nodes (63): AdapterRecord, Any, CanonicalInteraction, str, Any, CanonicalInteraction, CanonicalMessage, str (+55 more)

### Community 3 - "Agno Adapter & Tracing"
Cohesion: 0.06
Nodes (58): Any, CanonicalInteraction, str, agno._run_metadata, agno._tool_execution_messages, AgentOS Tracing, AgnoAdapter, AgnoTraceCollector (+50 more)

### Community 4 - "CLI Convert Pipeline"
Cohesion: 0.07
Nodes (67): _adapter_record_to_interactions(), _adapter_record_to_message_batches(), _alpaca_to_messages(), _coerce_text(), convert(), _ensure_mapping_records(), _ensure_supported_source(), _expect_list() (+59 more)

### Community 5 - "BaseAdapter Buffering Core"
Cohesion: 0.06
Nodes (46): BaseAdapter, Write all buffered interactions to storage immediately.          Returns, Enter a ``with`` block — no special setup needed., Exit the ``with`` block — automatically flushes all data.          Example, Common logic for capturing agent interactions and saving them as datasets., Move a finished interaction from pending to the buffer.          If the buffer r, Format and write all buffered interactions to storage.          Returns, _interaction() (+38 more)

### Community 6 - "CrewAI Adapter"
Cohesion: 0.08
Nodes (46): Any, CanonicalInteraction, str, BaseAdapter._finalise_and_flush, BaseAdapter._flush_buffer, BaseAdapter.flush, _agent_metadata(), _context_provenance() (+38 more)

### Community 7 - "Adapter Test Suite & Canonical Model"
Cohesion: 0.05
Nodes (53): BaseAdapter Tests, AgnoTraceCollector, agno from_run_output, Agno Adapter Local Save Example, BaseAdapter, CanonicalInteraction, CanonicalMessage, _adapter_record_to_interactions (+45 more)

### Community 8 - "Atomic Agents Adapter"
Cohesion: 0.08
Nodes (45): Any, CanonicalInteraction, str, Any, str, _agent_metadata(), AtomicAgentsTraceCollector, from_agent_response() (+37 more)

### Community 9 - "LangGraph Adapter"
Cohesion: 0.09
Nodes (36): Any, bool, CanonicalInteraction, InteractionCollector, str, Duck-typed LangGraph Integration, LangGraph adapter package., _extract_messages (langgraph) (+28 more)

### Community 10 - "MCP Adapter"
Cohesion: 0.15
Nodes (33): Any, CanonicalInteraction, str, Model Context Protocol adapter package., MCP JSON-RPC Pairing by id, _content_from_tool_result (mcp), _session_id (mcp), _content_from_tool_result() (+25 more)

### Community 11 - "AgentOps Adapter"
Cohesion: 0.18
Nodes (15): from_events(), from_trace(), AgentOps trace/export adapter., Normalize an AgentOps trace/export payload., Normalize AgentOps operation/task/tool event exports., agentops from_events, agentops from_trace, AgentOps adapter package. (+7 more)

### Community 12 - "MLflow Adapter"
Cohesion: 0.19
Nodes (14): Any, CanonicalInteraction, str, MLflow adapter package., from_trace(), from_trace_dict(), AgentScribe + MLflow Capture Guide, MLflow tracing adapter. (+6 more)

### Community 13 - "CLI Entrypoint & Integration Tests"
Cohesion: 0.25
Nodes (11): main(), Capture, inspect, and convert AgentScribe fine-tuning datasets., read_jsonl_file(), write_jsonl_file(), test_cli_convert_adapter_record_uses_registry_formatter_and_local_storage(), test_cli_convert_jsonl_openai_chat_to_sharegpt_local_file(), test_cli_convert_skip_invalid_continues_writing_valid_records(), test_framework_collector_converts_records_and_flushes_through_shared_collector() (+3 more)

### Community 14 - "Streaming Event Normalization"
Cohesion: 0.17
Nodes (12): from_chat_history, autogen.from_stream_events, from_task_result, messages_from_autogen_item, Autogen Adapter Tests, from_state, langgraph.from_stream_events, LangGraphRecorder (+4 more)

### Community 15 - "Canonical Serialization"
Cohesion: 0.27
Nodes (6): Any, str, Reconstruct an interaction from a dictionary., Convert to a plain dictionary for serialization., Create a CanonicalMessage from a dictionary., Convert the whole interaction to a dictionary.

### Community 16 - "OpenInference/OTel Package Exports"
Cohesion: 0.22
Nodes (9): OpenInference Adapter Package, OpenTelemetry Adapter Package, openinference.from_spans, openinference.from_trace, openinference.messages_from_span, openinference.span_attributes, OpenInference Package Init Test, OpenInference Adapter Tests (+1 more)

### Community 17 - "CrewAI Converters & Tests"
Cohesion: 0.33
Nodes (6): CrewAIAdapter, crewai.from_event, from_kickoff_output, from_llm_call_context, from_tool_call_context, CrewAI Adapter Tests

### Community 18 - "MCP JSON-RPC Converters"
Cohesion: 0.40
Nodes (5): from_jsonrpc_messages, from_jsonrpc_pair, mcp.from_tool_call, from_tools_list, MCP Adapter Tests

### Community 19 - "AutoGen/CrewAI Package Init"
Cohesion: 0.50
Nodes (4): Autogen Adapter Package, CrewAI Adapter Package, Autogen Package Init Test, CrewAI Package Init Test

### Community 20 - "Adapter Convention Tests"
Cohesion: 0.67
Nodes (3): Adapter Convention Tests, Adapter Package Init Tests, OpenTelemetry Adapter Tests

### Community 21 - "MLflow Converters & Tests"
Cohesion: 0.67
Nodes (3): mlflow.from_trace, from_trace_dict, MLflow Adapter Tests

## Knowledge Gaps
- **90 isolated node(s):** `str`, `bytes`, `BackendFactory`, `AgentScribe Project Overview`, `Canonical Data Model` (+85 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CanonicalInteraction` connect `BaseAdapter Buffering Core` to `AutoGen & LangGraph Adapters`, `OpenInference/OTel Adapters & Registry`, `Agno Adapter & Tracing`, `CLI Convert Pipeline`, `CrewAI Adapter`, `Atomic Agents Adapter`, `LangGraph Adapter`, `MCP Adapter`, `AgentOps Adapter`, `MLflow Adapter`, `CLI Entrypoint & Integration Tests`, `Canonical Serialization`?**
  _High betweenness centrality (0.381) - this node is a cross-community bridge._
- **Why does `BaseAdapter` connect `Adapter Test Suite & Canonical Model` to `Agno Adapter & Tracing`, `CrewAI Adapter`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Are the 77 inferred relationships involving `CanonicalInteraction` (e.g. with `AdapterRecord` and `BaseAdapter`) actually correct?**
  _`CanonicalInteraction` has 77 INFERRED edges - model-reasoned connections that need verification._
- **Are the 23 inferred relationships involving `CanonicalMessage` (e.g. with `Any` and `bool`) actually correct?**
  _`CanonicalMessage` has 23 INFERRED edges - model-reasoned connections that need verification._
- **What connects `str`, `bytes`, `BackendFactory` to the rest of the system?**
  _274 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `AutoGen & LangGraph Adapters` be split into smaller, more focused modules?**
  _Cohesion score 0.07328907048008172 - nodes in this community are weakly interconnected._
- **Should `Storage Backends & Serialization` be split into smaller, more focused modules?**
  _Cohesion score 0.06683878370625358 - nodes in this community are weakly interconnected._