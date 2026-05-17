from __future__ import annotations

from agentscribe.adapters.mlflow import from_trace, from_trace_dict


def test_from_trace_uses_data_spans_trace_id_and_request_metadata() -> None:
    trace = {
        "info": {"trace_id": "trace-1", "request_metadata": {"user": "u1"}},
        "data": {"spans": [{"input.value": "Q", "output.value": "A"}]},
    }

    interaction = from_trace(trace, metadata={"case": "trace"})

    assert interaction.source_framework == "mlflow"
    assert interaction.trace_id == "trace-1"
    assert interaction.metadata["source_shape"] == "trace"
    assert interaction.metadata["user"] == "u1"
    assert interaction.metadata["case"] == "trace"
    assert [message.content for message in interaction.messages] == ["Q", "A"]


def test_from_trace_falls_back_to_top_level_operations() -> None:
    interaction = from_trace({"request_id": "request-1", "operations": [{"output.value": "A"}]})

    assert interaction.trace_id == "request-1"
    assert interaction.messages[0].content == "A"


def test_from_trace_dict_delegates_to_from_trace() -> None:
    interaction = from_trace_dict({"spans": [{"input.value": "Q"}]})

    assert interaction.source_framework == "mlflow"
    assert interaction.messages[0].content == "Q"
