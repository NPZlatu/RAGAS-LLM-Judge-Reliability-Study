"""
LLM Judge initialization for RAGAS evaluation.
Phase 1: Gemini 2.5 Flash (Google) and Llama 3.1 8B (Groq).
"""

import os
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper

load_dotenv()


def init_gemini_judge():
    """Judge 1: Gemini 2.5 Flash via Google AI free tier."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set in .env")

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return LangchainLLMWrapper(llm)


def init_groq_judge():
    """Judge 2: Llama 3.1 8B Instant via Groq free tier."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in .env")

    from langchain_groq import ChatGroq

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    return LangchainLLMWrapper(llm)


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
