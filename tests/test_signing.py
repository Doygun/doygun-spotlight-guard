import pytest

from src.utils.signing import (
    NonceCache,
    make_signature,
    sign_payload,
    verify_payload,
    verify_signature,
)

def test_signature_roundtrip():
    message = "test message"
    signature = make_signature(message)
    assert verify_signature(message, signature)
    assert not verify_signature(message + "!", signature)

@pytest.mark.parametrize(
    "length",
    [8, 16, 32],
)
def test_signature_length(length: int):
    message = "another message"
    signature = make_signature(message, n=length)
    assert len(signature) == length

def test_signed_payload_roundtrip():
    payload = sign_payload("enroll certificate", timestamp=1000.0)
    ok, reason = verify_payload(payload, now=1000.0)
    assert ok and reason == "ok"

def test_tampered_payload_rejected():
    payload = sign_payload("enroll certificate", timestamp=1000.0)
    payload.message = "transfer all funds"
    ok, reason = verify_payload(payload, now=1000.0)
    assert not ok and reason == "invalid_mac"

def test_stale_timestamp_rejected():
    payload = sign_payload("enroll certificate", timestamp=1000.0)
    ok, reason = verify_payload(payload, now=1000.0 + 10_000, window_s=300.0)
    assert not ok and reason == "stale_timestamp"

def test_replayed_nonce_rejected():

    cache = NonceCache(window_s=300.0)
    payload = sign_payload("enroll certificate", timestamp=1000.0)

    first_ok, _ = verify_payload(payload, now=1000.0, nonce_cache=cache)
    second_ok, reason = verify_payload(payload, now=1001.0, nonce_cache=cache)

    assert first_ok
    assert not second_ok and reason == "replayed_nonce"

def test_distinct_nonces_allowed():
    cache = NonceCache(window_s=300.0)
    p1 = sign_payload("read calendar", timestamp=1000.0)
    p2 = sign_payload("read calendar", timestamp=1000.0)
    ok1, _ = verify_payload(p1, now=1000.0, nonce_cache=cache)
    ok2, _ = verify_payload(p2, now=1000.0, nonce_cache=cache)
    assert ok1 and ok2 and p1.nonce != p2.nonce
