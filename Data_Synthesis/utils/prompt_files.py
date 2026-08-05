from pathlib import Path
from typing import Dict, Iterable, Union
import os


ROOT = Path(__file__).resolve().parent.parent


def get_prompt_root() -> Path:
    env_dir = os.getenv("OMNIDOC_PROMPT_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (ROOT / "prompts").resolve()


def get_prompt_file_path(step_name: str, prompt_key: str) -> Path:
    safe_step = step_name.strip()
    safe_key = prompt_key.strip()
    return get_prompt_root() / f"{safe_step}__{safe_key}.txt"


def load_step_prompts(
    step_name: str,
    prompts: Union[Dict[str, str], Iterable[str]],
    logger=None,
) -> Dict[str, str]:
    keys = prompts.keys() if isinstance(prompts, dict) else prompts
    resolved: Dict[str, str] = {}
    for key in keys:
        path = get_prompt_file_path(step_name, key)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        try:
            resolved[key] = path.read_text(encoding="utf-8")
        except Exception as e:
            if logger:
                logger.error("Failed to read prompt file %s: %s", path, e)
            raise
    return resolved
