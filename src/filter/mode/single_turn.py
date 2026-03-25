"""Utilities for formatting single-turn conversations."""


def format_single_turn(row: dict, max_chars: int = None) -> str | None:
    """Format the first user+assistant exchange as a plain-text string.

    Returns None if the conversation has fewer than 2 turns or the first
    user message is empty.

    Args:
        row: A dataset row with a "conversation" key containing a list of
             {"role": str, "content": str} dicts.
        max_chars: If set, truncate the combined text to this many characters.

    Returns:
        "USER: <msg>\\nASSISTANT: <msg>" or None.
    """
    conversation = row.get("conversation", [])
    if len(conversation) < 2:
        return None
    user_msg = conversation[0].get("content", "").strip()
    asst_msg = conversation[1].get("content", "").strip()
    if not user_msg:
        return None
    text = f"USER: {user_msg}\nASSISTANT: {asst_msg}"
    if max_chars is not None:
        text = text[:max_chars]
    return text
