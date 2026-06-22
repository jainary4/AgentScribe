"""Regression test for the AgnoAdapter buffered-write durability bug."""

import json
import subprocess
import sys
import textwrap


def test_atexit_flushes_buffer_on_process_exit(tmp_path):
    """Bug under test:
        AgnoAdapter buffers captured interactions and, with the default
        flush_interval=10, writes NOTHING to storage until the buffer fills or
        the user calls flush()/uses a `with` block. A process that captures
        fewer than flush_interval interactions and simply exits therefore lost
        all of them silently.

    Expected behaviour (atexit durability net):
        On a normal process exit, the buffered interactions are flushed to
        storage at interpreter shutdown — so the output file contains every
        captured interaction even though flush() was never called and no `with`
        block was used.

    (A subprocess is required because atexit handlers only run at real
    interpreter shutdown, not when an object is dropped inside the test.)
    """
    out = tmp_path / "data.jsonl"
    script = textwrap.dedent("""
        import sys
        from agentscribe.adapters.agno import AgnoAdapter

        adapter = AgnoAdapter(output=sys.argv[1])     # default flush_interval=10
        for i in range(3):
            adapter.post_hook(
                {"messages": [{"role": "user", "content": f"q{i}"},
                              {"role": "assistant", "content": f"a{i}"}]},
                {"name": "agent"},
            )
        # deliberately NO flush() and NO `with` — rely on the atexit handler
    """)
    subprocess.run([sys.executable, "-c", script, str(out)], check=True)

    written = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(written) == 3