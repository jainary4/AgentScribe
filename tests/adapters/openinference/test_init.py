from __future__ import annotations

import agentscribe.adapters.openinference as openinference


def test_openinference_package_reexports_public_api() -> None:
    assert set(openinference.__all__) == {"from_spans", "from_trace", "messages_from_span", "span_attributes"}
