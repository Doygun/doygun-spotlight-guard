import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass
from typing import Final, Optional

DEFAULT_SECRET_ENV: Final[str] = "PROMPT_INTEGRITY_KEY"
DEFAULT_SECRET: Final[str] = "prompt_integrity_key_2025"
DEFAULT_DIGITS: Final[int] = 16
DEFAULT_FRESHNESS_WINDOW_S: Final[float] = 300.0

def _resolve_secret(secret: Optional[str]) -> str:

    if secret is not None:
        return secret
    return os.environ.get(DEFAULT_SECRET_ENV, DEFAULT_SECRET)

def make_signature(
    message: str, secret: Optional[str] = None, n: int = DEFAULT_DIGITS
) -> str:

    key = _resolve_secret(secret)
    digest = hmac.new(key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:n]

def verify_signature(
    message: str, signature: str, secret: Optional[str] = None
) -> bool:
    expected = make_signature(message, secret=secret, n=len(signature))
    return hmac.compare_digest(expected, signature)

def generate_nonce(num_bytes: int = 16) -> str:

    return secrets.token_hex(num_bytes)

@dataclass
class SignedPayload:

    message: str
    nonce: str
    timestamp: float
    signature: str

    def canonical(self) -> str:
        return _canonical(self.message, self.nonce, self.timestamp)

def _canonical(message: str, nonce: str, timestamp: float) -> str:

    return f"{message}|{nonce}|{timestamp:.3f}"

def sign_payload(
    message: str,
    secret: Optional[str] = None,
    n: int = DEFAULT_DIGITS,
    nonce: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> SignedPayload:

    nonce = nonce or generate_nonce()
    timestamp = time.time() if timestamp is None else timestamp
    signature = make_signature(
        _canonical(message, nonce, timestamp), secret=secret, n=n
    )
    return SignedPayload(message, nonce, timestamp, signature)

class NonceCache:

    def __init__(self, window_s: float = DEFAULT_FRESHNESS_WINDOW_S) -> None:
        self._window_s = window_s
        self._seen: dict[str, float] = {}

    def _evict(self, now: float) -> None:
        cutoff = now - self._window_s
        stale = [n for n, ts in self._seen.items() if ts < cutoff]
        for nonce in stale:
            del self._seen[nonce]

    def check_and_register(self, nonce: str, timestamp: float) -> bool:

        self._evict(timestamp)
        if nonce in self._seen:
            return False
        self._seen[nonce] = timestamp
        return True

def verify_payload(
    payload: SignedPayload,
    secret: Optional[str] = None,
    window_s: float = DEFAULT_FRESHNESS_WINDOW_S,
    nonce_cache: Optional[NonceCache] = None,
    now: Optional[float] = None,
) -> tuple[bool, str]:

    now = time.time() if now is None else now

    if not verify_signature(payload.canonical(), payload.signature, secret=secret):
        return False, "invalid_mac"

    age = now - payload.timestamp
    if age > window_s:
        return False, "stale_timestamp"
    if age < -window_s:
        return False, "future_timestamp"

    if nonce_cache is not None and not nonce_cache.check_and_register(
        payload.nonce, payload.timestamp
    ):
        return False, "replayed_nonce"

    return True, "ok"
