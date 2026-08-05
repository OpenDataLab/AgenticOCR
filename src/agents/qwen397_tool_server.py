"""qwen397_tool wrapper: OpenAI-compat passthrough that injects the
AgenticOCR system prompt before forwarding to upstream Qwen3.5-397B.

The existing AgenticOCR Python client drives this wrapper via multi-turn
tool calls — same flow as the original AgenticOCR endpoint, only the LLM
backbone changes.

Run via scripts/start_qwen397_tool.sh.
"""

import os
import logging
import uvicorn

from src.agents.AgenticOCR import SYSTEM_PROMPT
from src.agents._qwen_proxy_common import make_app

app = make_app(
    title="Qwen3.5-397B AgenticOCR-Tool Wrapper",
    system_prompt=SYSTEM_PROMPT,
)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    port = int(os.environ.get("QWEN397_TOOL_PORT", 8006))
    uvicorn.run(app, host="0.0.0.0", port=port)
