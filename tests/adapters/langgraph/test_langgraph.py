from __future__ import annotations

from types import SimpleNamespace

from agentscribe.adapters.langgraph import LangGraphRecorder, from_state, from_stream_events, normalize_stream_event, wrap_graph
from agentscribe.adapters.utils import InteractionCollector


def test_from_state_extracts_direct_messages_thread_id_and_extra_state() -> None:
    graph = SimpleNamespace(name="graph-a")
    interaction = from_state(
        {"messages": [{"role": "user", "content": "hi"}], "score": 1},
        config={"configurable": {"thread_id": "thread-1"}},
        graph=graph,
        metadata={"case": "state"},
    )

    assert interaction.session_id == "thread-1"
    assert interaction.thread_id == "thread-1"
    assert interaction.metadata["graph_name"] == "graph-a"
    assert interaction.metadata["case"] == "state"
    assert interaction.extra["state"] == {"score": 1}
    assert interaction.messages[0].content == "hi"


def test_from_state_extracts_nested_messages_and_session_override() -> None:
    interaction = from_state({"node": {"messages": [{"role": "assistant", "content": "nested"}]}}, session_id="manual")

    assert interaction.session_id == "manual"
    assert interaction.messages[0].content == "nested"


def test_normalize_stream_event_handles_mapping_tuple_shapes_and_unknowns() -> None:
    assert normalize_stream_event({"stream_mode": "updates", "namespace": ("n",), "data": {"x": 1}}) == {
        "mode": "updates",
        "namespace": ["n"],
        "data": {"x": 1},
    }
    assert normalize_stream_event(("values", {"messages": []})) == {"mode": "values", "data": {"messages": []}}
    assert normalize_stream_event(("messages", ("chunk", {"meta": True}))) == {"mode": "messages", "data": ["chunk", {"meta": True}]}
    assert normalize_stream_event(("mode", "ns", {"x": 1})) == {"mode": "mode", "namespace": "ns", "data": {"x": 1}}
    assert normalize_stream_event(5) == {"mode": "unknown", "data": 5}


def test_from_stream_events_records_spans_and_omits_chunks_by_default() -> None:
    events = [
        ("updates", {"messages": [{"role": "user", "content": "Q"}]}),
        ("messages", ({"role": "assistant", "content": "partial"}, {})),
    ]

    interaction = from_stream_events(events, config={"configurable": {"thread_id": "thread"}})

    assert interaction.session_id == "thread"
    assert [message.content for message in interaction.messages] == ["Q"]
    assert len(interaction.spans) == 2


def test_from_stream_events_can_include_message_chunks() -> None:
    interaction = from_stream_events([("messages", ({"role": "assistant", "content": "partial"}, {}))], include_message_chunks=True)

    assert interaction.messages[0].content == "partial"
    assert interaction.messages[0].metadata["stream_chunk"] is True


def test_langgraph_recorder_invokes_streams_and_records_results() -> None:
    class Graph:
        name = "g"

        def invoke(self, *_args, **_kwargs):
            return {"messages": [{"role": "assistant", "content": "done"}]}

        def stream(self, *_args, **_kwargs):
            yield ("values", {"messages": [{"role": "assistant", "content": "streamed"}]})

    collector = InteractionCollector(source_framework="langgraph")
    recorder = LangGraphRecorder(Graph(), collector=collector)

    assert recorder.invoke({"messages": []})["messages"][0]["content"] == "done"
    assert recorder.stream({}) == [("values", {"messages": [{"role": "assistant", "content": "streamed"}]})]
    assert [item.messages[0].content for item in collector.interactions] == ["done", "streamed"]
    assert wrap_graph(recorder.graph, collector=collector).graph is recorder.graph
