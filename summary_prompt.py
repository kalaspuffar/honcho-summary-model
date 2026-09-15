#!/usr/bin/env python3
"""summary_prompt.py — VERBATIM copy of Honcho's summariser prompts (parity file — never hand-edit).

Source: plastic-labs/honcho `src/utils/summarizer.py`, commit 8e386180bd87b852e7934cfef53cba3d6bee1bb4
(main, 2026-09-11), copied 2026-09-12. If Honcho changes these, re-copy this file, bump the commit
above, regenerate all data and retrain.

How Honcho calls the model (src/utils/summarizer.py + src/llm/api.py, same commit):
  - ONE user message whose content is the prompt below; no system prompt; no tools.
  - previous_summary_text = stored summary, or NO_PREVIOUS_SUMMARY.
  - formatted_messages = _format_messages(messages)  ("peer_name: content" per line).
  - short: output_words = int(min(input_tokens, SUMMARY_MAX_TOKENS_SHORT) * 0.75), max_tokens = SUMMARY_MAX_TOKENS_SHORT (default 1000)
    long : output_words = int(SUMMARY_MAX_TOKENS_LONG * 0.75),                     max_tokens = SUMMARY_MAX_TOKENS_LONG  (default 4000)
  - cadence: short every MESSAGES_PER_SHORT_SUMMARY (20) messages, long every MESSAGES_PER_LONG_SUMMARY (60).
  - transport: OpenAI-compatible /v1/chat/completions; the answer is `content` only.
"""
from inspect import cleandoc as c

NO_PREVIOUS_SUMMARY = "There is no previous summary -- the messages are the beginning of the conversation."
MAX_TOKENS_SHORT_DEFAULT = 1000
MAX_TOKENS_LONG_DEFAULT = 4000
MESSAGES_PER_SHORT_SUMMARY = 20
MESSAGES_PER_LONG_SUMMARY = 60


def output_words_short(input_tokens: int, max_tokens_short: int = MAX_TOKENS_SHORT_DEFAULT) -> int:
    return int(min(input_tokens, max_tokens_short) * 0.75)


def output_words_long(max_tokens_long: int = MAX_TOKENS_LONG_DEFAULT) -> int:
    return int(max_tokens_long * 0.75)


def short_summary_prompt(
    formatted_messages: str,
    output_words: int,
    previous_summary_text: str,
) -> str:
    """Generate the short summary prompt."""
    return c(f"""
You are a system that summarizes parts of a conversation to create a concise and accurate summary. Focus on capturing:

1. Key facts and information shared (**Capture as many explicit facts as possible**)
2. User preferences, opinions, and questions
3. Important context and requests
4. Core topics discussed

If there is a previous summary, ALWAYS make your new summary inclusive of both it and the new messages, therefore capturing the ENTIRE conversation. Prioritize key facts across the entire conversation.

Provide a concise, factual summary that captures the essence of the conversation. Your summary should be detailed enough to serve as context for future messages, but brief enough to be helpful. Prefer a thorough chronological narrative over a list of bullet points.

Return only the summary without any explanation or meta-commentary.

<previous_summary>
{previous_summary_text}
</previous_summary>

<conversation>
{formatted_messages}
</conversation>

Hard limit: {output_words} words maximum. If needed, drop lower-priority detail to stay within the limit.
""")




def long_summary_prompt(
    formatted_messages: str,
    output_words: int,
    previous_summary_text: str,
) -> str:
    """Generate the long summary prompt."""
    return c(f"""
You are a system that creates thorough, comprehensive summaries of conversations. Focus on capturing:

1. Key facts and information shared (**Capture as many explicit facts as possible**)
2. User preferences, opinions, and questions
3. Important context and requests
4. Core topics discussed in detail
5. User's apparent emotional state and personality traits
6. Important themes and patterns across the conversation

If there is a previous summary, ALWAYS make your new summary inclusive of both it and the new messages, therefore capturing the ENTIRE conversation. Prioritize key facts across the entire conversation.

Provide a thorough and detailed summary that captures the essence of the conversation. Your summary should serve as a comprehensive record of the important information in this conversation. Prefer an exhaustive chronological narrative over a list of bullet points.

Return only the summary without any explanation or meta-commentary.

<previous_summary>
{previous_summary_text}
</previous_summary>

<conversation>
{formatted_messages}
</conversation>

Hard limit: {output_words} words maximum. If needed, drop lower-priority detail to stay within the limit.
""")



def _format_messages(messages: list) -> str:
    """
    Format a list of messages into a string by concatenating their content and
    prefixing each with the peer name.
    """
    if len(messages) == 0:
        return ""
    return "\n".join([f"{msg.peer_name}: {msg.content}" for msg in messages])


def format_messages(messages: list) -> str:
    """Honcho's _format_messages over dicts {"peer_name", "content"} (or objects with those attributes)."""
    class _M:
        def __init__(self, d): self.peer_name = d["peer_name"]; self.content = d["content"]
    return _format_messages([_M(m) if isinstance(m, dict) else m for m in messages])


def build_messages(kind: str, messages: list, previous_summary, output_words: int) -> list:
    """The exact chat payload Honcho sends: one user turn. kind = "short" | "long"."""
    prev = previous_summary or NO_PREVIOUS_SUMMARY
    fn = short_summary_prompt if kind == "short" else long_summary_prompt
    return [{"role": "user", "content": fn(format_messages(messages), output_words, prev)}]
