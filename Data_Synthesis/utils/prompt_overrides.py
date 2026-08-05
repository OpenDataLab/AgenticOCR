import json
import os
from functools import lru_cache
from typing import Dict


@lru_cache(maxsize=1)
def _load_override_payload() -> Dict[str, Dict[str, str]]:
    """Load prompt override payload from env-defined JSON file.

    Expected format:
    {
      "step_1_generate_outline": {"PROMPT": "..."},
      "step_5_verify_qa": {"PROMPT_STEP1": "...", "PROMPT_STEP2": "..."}
    }
    """
    path = os.getenv("OMNIDOC_PROMPT_OVERRIDES")
    if not path:
        return {}

    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}

    return payload


def apply_prompt_overrides(step_name: str, prompts: Dict[str, str], logger=None) -> Dict[str, str]:
    """Apply prompt overrides for one step.

    `prompts` is a dict of current prompt variables; returns updated dict.
    """
    all_overrides = _load_override_payload()
    step_overrides = all_overrides.get(step_name, {})
    if not isinstance(step_overrides, dict):
        return prompts

    updated = dict(prompts)
    for key, value in step_overrides.items():
        if key in updated and isinstance(value, str) and value.strip():
            updated[key] = value
            if logger:
                logger.info("Prompt override applied: %s.%s", step_name, key)
    return updated
