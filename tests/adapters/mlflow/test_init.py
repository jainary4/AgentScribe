from __future__ import annotations

import agentscribe.adapters.mlflow as mlflow


def test_mlflow_package_reexports_public_api() -> None:
    assert set(mlflow.__all__) == {"from_trace", "from_trace_dict"}
