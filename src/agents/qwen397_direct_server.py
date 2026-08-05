"""qwen397_direct wrapper: OpenAI-compat passthrough that injects a
single-shot JSON-extraction system prompt before forwarding to upstream
Qwen3.5-397B.

The wrapped LLM responds with a single ```json ... ``` block on the first
turn, so the existing AgenticOCR Python client terminates after one round.

Run via scripts/start_qwen397_direct.sh.
"""

import os
import logging
import uvicorn

from src.agents._qwen_proxy_common import make_app


SYSTEM_PROMPT = """You are a Visual Document Analysis Agent. Given a page image and a user query, identify every region on the page that serves as evidence for answering the query.

### Input
1. One page image.
2. The user's query.

### Task
1. **Relevance check**: decide whether the page contains any evidence for the query. If not, return an empty list.
2. **Evidence extraction**: if relevant, list every evidence region — text blocks, tables, charts, images, equations. Each region should be self-contained (understandable without page context).

### Output Format
Respond with a single JSON array inside a ```json fenced block. Coordinates are normalized to 0–1000 relative to the original page (x1,y1,x2,y2).

```json
[
  {
    "evidence": "<self-contained content>",
    "bbox": [xmin, ymin, xmax, ymax]
  }
]
```

If the page is irrelevant, return:
```json
[]
```

Do not emit tool calls. Do not include any commentary outside the JSON block."""


app = make_app(
    title="Qwen3.5-397B Direct Wrapper",
    system_prompt=SYSTEM_PROMPT,
)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    port = int(os.environ.get("QWEN397_DIRECT_PORT", 8007))
    uvicorn.run(app, host="0.0.0.0", port=port)
