import base64
from typing import Final

TRUSTED_BLOCK_START: Final[str] = "[TRUSTED_SYSTEM_BLOCK]"
TRUSTED_BLOCK_END: Final[str] = "[/TRUSTED_SYSTEM_BLOCK]"
UNTRUSTED_BLOCK_START: Final[str] = "[UNTRUSTED_ENCODED_BLOCK]"
UNTRUSTED_BLOCK_END: Final[str] = "[/UNTRUSTED_ENCODED_BLOCK]"

DATAMARK_CHAR: Final[str] = "\u2581"

SPOTLIGHT_MODES: Final[tuple[str, ...]] = ("base64", "delimiter", "datamarking")

def spotlight_wrap_trusted(text: str) -> str:

    return f"{TRUSTED_BLOCK_START}\n{text}\n{TRUSTED_BLOCK_END}"

def _datamark(text: str) -> str:

    return DATAMARK_CHAR.join(text.split())

def spotlight_encode_untrusted(text: str, mode: str = "base64") -> str:

    if mode not in SPOTLIGHT_MODES:
        raise ValueError(f"Unknown spotlight mode: {mode!r}")

    if mode == "base64":
        body = base64.b64encode(text.encode("utf-8")).decode("ascii")
    elif mode == "datamarking":
        body = _datamark(text)
    else:
        body = text

    return f"{UNTRUSTED_BLOCK_START}\n{body}\n{UNTRUSTED_BLOCK_END}"

def spotlight_decode(text: str) -> str:

    stripped = text.strip()
    if not stripped.startswith(UNTRUSTED_BLOCK_START):
        return text
    try:
        encoded = stripped.splitlines()[1]
        return base64.b64decode(encoded.encode("ascii")).decode("utf-8")
    except Exception:
        return text
