import os


def get_env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_first_env(names: list[str], default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        value = value.strip()
        if value:
            return value
    return default


def _normalize_genai_base_url(url: str) -> str:
    """Normalize base URL for google genai client.

    GenAI SDK expects gateway base URL, not OpenAI-style /v1.
    """
    if not url:
        return url
    normalized = url.strip()
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    elif normalized.endswith("/v1/"):
        normalized = normalized[:-4]
    if normalized and not normalized.endswith("/"):
        normalized += "/"
    return normalized


def get_genai_base_url() -> str:
    raw = get_first_env(
        [
            "OMNIDOC_GENAI_BASE_URL",
            "GEMINI_BASE_URL",
            "OPENAI_API_BASE",
            "API_BASE_URL",
            "API_BASE",
            "BASE_URL",
        ],
        "",
    )
    return _normalize_genai_base_url(raw)


def get_genai_api_key() -> str:
    return get_first_env(
        [
            "OMNIDOC_GENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "API_KEY",
        ],
        "no-key-required",
    )


def get_genai_model() -> str:
    return get_first_env(
        [
            "OMNIDOC_GENAI_MODEL",
            "OPENAI_MODEL_NAME",
            "MODEL_NAME",
        ],
        "gemini-3-flash-preview",
    )


def get_genai_timeout_ms() -> int:
    return get_env_int("OMNIDOC_GENAI_TIMEOUT_MS", 1200000)


def create_genai_client(genai_module, types_module):
    return genai_module.Client(
        http_options=types_module.HttpOptions(
            base_url=get_genai_base_url(),
            timeout=get_genai_timeout_ms(),
        ),
        api_key=get_genai_api_key(),
    )


def create_genai_client_with(genai_module, types_module, base_url: str, api_key: str):
    return genai_module.Client(
        http_options=types_module.HttpOptions(
            base_url=_normalize_genai_base_url(base_url),
            timeout=get_genai_timeout_ms(),
        ),
        api_key=(api_key or "").strip() or "no-key-required",
    )


def get_step7_check_base_url() -> str:
    raw = get_first_env(
        [
            "OMNIDOC_STEP7_CHECK_BASE_URL",
            "OMNIDOC_QWEN_BASE_URL",
            "OMNIDOC_GENAI_BASE_URL",
            "GEMINI_BASE_URL",
            "OPENAI_API_BASE",
            "API_BASE_URL",
            "API_BASE",
            "BASE_URL",
        ],
        "",
    )
    return _normalize_genai_base_url(raw)


def get_step7_check_api_key() -> str:
    return get_first_env(
        [
            "OMNIDOC_STEP7_CHECK_API_KEY",
            "OMNIDOC_QWEN_API_KEY",
            "OMNIDOC_GENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "API_KEY",
        ],
        "no-key-required",
    )


def get_step7_check_model() -> str:
    return get_first_env(
        [
            "OMNIDOC_STEP7_CHECK_MODEL",
            "OMNIDOC_QWEN_MODEL",
            "OMNIDOC_GENAI_MODEL",
            "OPENAI_MODEL_NAME",
            "MODEL_NAME",
        ],
        "gemini-3-flash-preview",
    )


def get_step7_judge_base_url() -> str:
    raw = get_first_env(
        [
            "OMNIDOC_STEP7_JUDGE_BASE_URL",
            "OMNIDOC_GENAI_BASE_URL",
            "GEMINI_BASE_URL",
            "OPENAI_API_BASE",
            "API_BASE_URL",
            "API_BASE",
            "BASE_URL",
        ],
        "",
    )
    return _normalize_genai_base_url(raw)


def get_step7_judge_api_key() -> str:
    return get_first_env(
        [
            "OMNIDOC_STEP7_JUDGE_API_KEY",
            "OMNIDOC_GENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "API_KEY",
        ],
        "no-key-required",
    )


def get_step7_judge_model() -> str:
    return get_first_env(
        [
            "OMNIDOC_STEP7_JUDGE_MODEL",
            "OMNIDOC_GENAI_MODEL",
            "OPENAI_MODEL_NAME",
            "MODEL_NAME",
        ],
        "gemini-3-flash-preview",
    )


def get_mineru_server_url() -> str:
    return get_env_str("OMNIDOC_MINERU_SERVER_URL", "")


def get_qwen_base_url() -> str:
    return get_first_env(
        [
            "OMNIDOC_QWEN_BASE_URL",
            "OPENAI_API_BASE",
        ],
        "",
    )


def get_qwen_api_key() -> str:
    return get_first_env(
        [
            "OMNIDOC_QWEN_API_KEY",
            "OPENAI_API_KEY",
            "API_KEY",
        ],
        "no-key-required",
    )


def get_qwen_model() -> str:
    return get_first_env(
        [
            "OMNIDOC_QWEN_MODEL",
            "OPENAI_MODEL_NAME",
            "MODEL_NAME",
        ],
        "",
    )
