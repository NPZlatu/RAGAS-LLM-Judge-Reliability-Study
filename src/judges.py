"""
LLM Judge initialization for RAGAS evaluation.
Phase 1: Gemini 2.5 Flash (Google) and Llama 3.1 8B (Groq).

Both judges use AsyncOpenAI pointed at provider-specific OpenAI-compatible
endpoints, because RAGAS 0.4.3 collections metrics require async LLM clients
(batch_score → abatch_score → ascore → agenerate).
"""

import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory

load_dotenv()


def init_gemini_judge():
    """Judge 1: Gemini 2.5 Flash via Google AI OpenAI-compatible endpoint."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set in .env")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    # Gemini 2.5 Flash is a "thinking" model — thinking tokens count against
    # max_tokens, so we need a large budget (default 1024 gets exhausted by ~980
    # thinking tokens, truncating the actual output).
    return llm_factory(
        "gemini-2.5-flash", provider="openai", client=client, max_tokens=8192,
    )


def init_groq_judge():
    """Judge 2: Llama 3.1 8B Instant via Groq OpenAI-compatible endpoint."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in .env")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    return llm_factory("llama-3.1-8b-instant", provider="openai", client=client)


def get_all_judges():
    """Initialize all Phase 1 judges. Returns dict of {name: wrapped_llm}."""
    judges = {}

    try:
        judges["gemini-2.5-flash"] = init_gemini_judge()
        print("[OK] Gemini 2.5 Flash initialized")
    except Exception as e:
        print(f"[FAIL] Gemini: {e}")

    try:
        judges["llama-3.1-8b-groq"] = init_groq_judge()
        print("[OK] Llama 3.1 8B (Groq) initialized")
    except Exception as e:
        print(f"[FAIL] Groq: {e}")

    return judges


if __name__ == "__main__":
    print("Testing judge initialization...\n")
    judges = get_all_judges()
    print(f"\nSuccessfully initialized {len(judges)}/2 judges: {list(judges.keys())}")
