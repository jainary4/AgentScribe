"""Optional-dependency adapter packages for AgentScribe.

Concrete packages expose duck-typed converters and collectors. They
intentionally avoid importing their target frameworks at package import time.
"""

__all__ = [
	"agentops",
	"agno",
	"atomic_agents",
	"autogen",
	"base",
	"crewai",
	"langgraph",
	"mcp",
	"mlflow",
	"opentelemetry",
	"openinference",
	"utils",
]
