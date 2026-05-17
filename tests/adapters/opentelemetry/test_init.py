from __future__ import annotations

import agentscribe.adapters.opentelemetry as opentelemetry


def test_opentelemetry_package_reexports_public_api() -> None:
    assert set(opentelemetry.__all__) == {"from_spans", "from_trace", "messages_from_span", "span_attributes"}
