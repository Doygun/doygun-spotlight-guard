ATTACK_KEYWORDS = {
    "ignore previous",
    "system prompt",
    "send to",
    "email to",
    "forward to",
    "transfer",
    "secret",
    "leak",
    "confidential",
    "policy",
    "payment",
    "bank",
    "password",
    "credentials",
    "access",
    "grant",
    "disable",
    "override",
}

def has_attack_markers(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in ATTACK_KEYWORDS)
