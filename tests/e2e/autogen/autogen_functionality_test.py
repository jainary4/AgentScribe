"""
AgentScribe x AutoGen / AG2 — every capture surface, every output format.

ONE file, two modes:
  * `python autogen_functionality_test.py`
        Runs live, writes out_autogen/<method>_<format>.jsonl for inspection,
        and prints per-format validation. (Loads .env itself so the key is found.)
  * `pytest -m live tests/e2e/autogen/autogen_functionality_test.py`
        Runs live, asserts every record conforms to its format via
        agentscribe.core.validation, and writes NO artifacts. Deselected by
        default (see addopts `-m "not live"`), so a plain `pytest tests/` never
        bills; opt in with `-m live`.

The four methods mirror the adapter's four exported surfaces:
  1. from_task_result           — the ChatResult object from initiate_chat
  2. from_chat_history          — the raw transcript list (here: a tool exchange)
  3. from_stream_events         — the live event stream from agent.run().events
  4. messages_from_autogen_item — the lowest-level, one-item-at-a-time API

What conformance can and can't prove: validate_record checks each record's
SHAPE. It cannot tell whether `user`/`assistant` were swapped — an inverted
transcript is still a structurally valid record. AG2's role inversion is
therefore pinned by deterministic unit tests in
tests/adapters/autogen/test_autogen.py, not here.

Needs: ag2 with the openai SDK (`pip install 'ag2[openai]'`) and OPENROUTER_API_KEY
for live runs. (Plain `ag2` can't reach OpenAI/OpenRouter — openai is an optional
ag2 extra.)
"""

import os
from pathlib import Path

import pytest

# Keep `pytest tests/` collectable even without ag2 installed (no-op when running
# as a script with autogen present, which the script needs anyway).
pytest.importorskip("autogen")

from autogen import ConversableAgent, UserProxyAgent

from agentscribe.core.canonical import CanonicalInteraction
from agentscribe.core.formatter import Formatter
from agentscribe.core.validation import validate_record
from agentscribe.storage import write_jsonl
from agentscribe.adapters.autogen import (
    from_task_result, from_chat_history, from_stream_events, messages_from_autogen_item,
)

SCRIPT_DIR = Path(__file__).parent.resolve()
OUT = SCRIPT_DIR / "out_autogen"


def _load_dotenv() -> Path | None:
    """Make `python this_file.py` behave like pytest: load the repo .env so
    OPENROUTER_API_KEY is available. (pytest auto-loads it via pytest-env; a
    plain script run does not.) Dependency-free; never overrides an existing var.
    """
    for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        env_path = parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            return env_path
    return None


_load_dotenv()

MODEL = "google/gemini-3.1-flash-lite"
FORMATS = Formatter.SUPPORTED_FORMATS          # every shipped format, not just one
USER = "user"                                  # the proxy name we drive runs with

CONCISE = "You are a concise assistant. Answer in one short sentence."
EXTRACT = "Extract all programming languages mentioned and return them as a comma-separated list."


def _has_key() -> bool:
    return os.environ.get("OPENROUTER_API_KEY", "mock-key") != "mock-key"


def llm_config():
    return {"config_list": [{
        "api_type": "openai", "model": MODEL,
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY", "mock-key"),
    }]}


def make_agent(name="assistant", system_message="You are a helpful assistant."):
    return ConversableAgent(name=name, system_message=system_message, llm_config=llm_config())


def run_once(agent, message):
    """Drive exactly one assistant turn via a silent user proxy."""
    user = UserProxyAgent(name=USER, human_input_mode="NEVER",
                          max_consecutive_auto_reply=0,
                          code_execution_config={"use_docker": False})
    return user.initiate_chat(agent, message=message, clear_history=True, max_turns=1)


def with_system(agent, chat_history):
    """Prepend the agent's real system prompt to a captured transcript."""
    return [{"role": "system", "content": agent.system_message}] + list(chat_history)


def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the product."""
    return a * b


# ---------------------------------------------------------------- capture surfaces
# 1. from_task_result — pass the whole ChatResult object (plain agent).
def m1_task_result() -> CanonicalInteraction:
    agent = make_agent(name="plain")
    result = run_once(agent, "Translate 'good morning' to French.")
    return from_task_result(result, agent=agent)


# 2. from_chat_history — a real tool exchange, so every format is exercised
#    meaningfully (tool turns survive openai_chat/sharegpt, degrade in alpaca/etc).
def m2_chat_history() -> CanonicalInteraction:
    agent = make_agent(name="math", system_message="Use the multiply tool for any products.")
    agent.register_for_llm(name="multiply", description="Multiply two integers")(multiply)
    user = UserProxyAgent(name=USER, human_input_mode="NEVER",
                          max_consecutive_auto_reply=1,
                          code_execution_config={"use_docker": False})
    user.register_for_execution(name="multiply")(multiply)
    result = user.initiate_chat(agent, message="Use your tool to compute 6 times 7.", clear_history=True)
    return from_chat_history(with_system(agent, result.chat_history))


# 3. from_stream_events — drive a real streamed run and consume its events.
def m3_stream_events() -> CanonicalInteraction:
    agent = make_agent(name="concise", system_message=CONCISE)
    response = agent.run(message="Give me one fact about the moon.", max_turns=1, user_input=False)
    return from_stream_events(list(response.events))


# 4. messages_from_autogen_item — assemble an interaction one item at a time.
def m4_single_items() -> CanonicalInteraction:
    agent = make_agent(name="extractor", system_message=EXTRACT)
    result = run_once(agent, "We shipped Python and Go services, with a little Ruby.")
    interaction = CanonicalInteraction(source_framework="autogen")
    for item in with_system(agent, result.chat_history):
        for message in messages_from_autogen_item(item, human_name=USER):
            interaction.messages.append(message)
    return interaction


SURFACES = [
    ("m1_task_result",   m1_task_result),
    ("m2_chat_history",  m2_chat_history),
    ("m3_stream_events", m3_stream_events),
    ("m4_single_items",  m4_single_items),
]


# ---------------------------------------------------------------- pytest mode
# Live: real run -> assert every record conforms to its format. No artifacts.
@pytest.mark.live
@pytest.mark.parametrize("tag,capture", SURFACES, ids=[tag for tag, _ in SURFACES])
def test_surface_conforms_to_every_format(tag, capture):
    if not _has_key():
        pytest.skip("OPENROUTER_API_KEY not set")
    interaction = capture()
    for fmt in FORMATS:
        record = Formatter(fmt).format_single(interaction)
        issues = validate_record(record, fmt)
        assert issues == [], f"{tag} ({fmt}) is non-conformant: {issues}"


# ---------------------------------------------------------------- script mode
def _run_as_script():
    OUT.mkdir(exist_ok=True)
    if not _has_key():
        print("WARNING: OPENROUTER_API_KEY not found (no .env loaded?). Live calls will 401 "
              "and out_autogen/ will stay empty.\n")

    for tag, capture in SURFACES:
        try:
            interaction = capture()
            print(f"[ok]   {tag}")
        except Exception as exc:
            print(f"[FAIL] {tag}: {type(exc).__name__}: {exc}")
            continue
        for fmt in FORMATS:
            record = Formatter(fmt).format_single(interaction)
            write_jsonl(OUT / f"{tag}_{fmt}.jsonl", [record], mode="a")
            issues = validate_record(record, fmt)
            tagged = "OK" if not issues else "ISSUES"
            print(f"    [{tagged}] {tag} ({fmt})" + (f": {'; '.join(issues)}" if issues else ""))

    print("\nLocal outputs:")
    for p in sorted(OUT.glob("*.jsonl")):
        print(f"  {p.name}: {sum(1 for _ in p.open())} record(s)")


if __name__ == "__main__":
    _run_as_script()
