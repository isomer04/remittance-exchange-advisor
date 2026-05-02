"""
Streamlit Chat UI for the Remittance & Exchange Advisor Agent

This version uses the google-genai Python SDK directly (no ADK).
Gemini's automatic function calling handles the tool loop for us —
the SDK converts Python functions to tool schemas, calls them when
Gemini requests it, and feeds results back to the model automatically.

Run locally:
    streamlit run app.py

Requires:
    - GOOGLE_API_KEY set in .env or as an environment variable
    - google-genai and streamlit installed (see requirements.txt)
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Load environment variables (for local dev — Replit uses Secrets instead)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL = "gemini-2.5-flash"
CSV_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "rate_sheet.csv"

SYSTEM_INSTRUCTION = (
    "You are a helpful remittance and currency exchange advisor assistant.\n\n"
    "RULES:\n"
    "- ALWAYS use the get_exchange_rate tool to look up exchange rates and fees. "
    "Never guess or invent an exchange rate.\n"
    "- ALWAYS use the calculate_remittance_amount tool to compute how much the recipient will receive. "
    "Never do the math yourself.\n"
    "- When a user asks about a remittance, first look up the exchange rate and fees, "
    "then pass those to the remittance calculator.\n"
    "- Present results in a clear, friendly way. Include the exchange rate, "
    "transfer fees, and the final amount the recipient will receive.\n"
    "- If the user hasn't specified a detail (sender country, recipient country, or amount), "
    "ask them before proceeding.\n"
    "- Always clarify the sender's currency and recipient's currency to find the right corridor.\n\n"
    "FORMATTING:\n"
    "- Respond in plain conversational sentences.\n"
    "- Do NOT use markdown headers (#, ##, ###).\n"
    "- Do NOT use bold or italic formatting.\n"
    "- Use line breaks to separate sections for readability.\n"
)


# ---------------------------------------------------------------------------
# Tool 1: Exchange Rate & Fee Lookup
# ---------------------------------------------------------------------------

def get_exchange_rate(
    from_country: str,
    to_country: str,
) -> dict:
    """Looks up the current exchange rate and transfer fees for a remittance corridor.

    This tool queries a governed, locally maintained CSV file — the single
    source of truth for today's rates and fees. The LLM should NEVER guess a rate.

    Args:
        from_country: The sender's country (e.g. 'United States', 'Canada').
        to_country: The recipient's country (e.g. 'India', 'Philippines').

    Returns:
        A dictionary with 'status' and either a 'result' containing
        the exchange rate, currencies, and transfer fees, or an 'error_message'.
    """
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        return {
            "status": "error",
            "error_message": f"Exchange rate sheet not found at {CSV_PATH}.",
        }

    match = df[
        (df["from_country"].str.lower() == from_country.strip().lower())
        & (df["to_country"].str.lower() == to_country.strip().lower())
    ]

    if match.empty:
        return {
            "status": "error",
            "error_message": (
                f"No exchange rate found for sending from '{from_country}' to '{to_country}'. "
                f"This corridor may not be supported yet. "
                f"Try asking about corridors like: United States to India, United States to Philippines, "
                f"Canada to India, Europe to India, United Kingdom to India, or Australia to Philippines."
            ),
        }

    row = match.iloc[0]
    return {
        "status": "success",
        "result": {
            "from_country": row["from_country"],
            "from_currency": row["from_currency"],
            "to_country": row["to_country"],
            "to_currency": row["to_currency"],
            "exchange_rate": float(row["exchange_rate"]),
            "transfer_fee_pct": float(row["transfer_fee_pct"]),
            "transfer_fee_min": float(row["transfer_fee_min"]),
        },
    }


# ---------------------------------------------------------------------------
# Tool 2: Remittance Amount Calculator
# ---------------------------------------------------------------------------

def calculate_remittance_amount(
    sender_amount: float,
    exchange_rate: float,
    transfer_fee_pct: float,
    transfer_fee_min: float,
    from_currency: str,
    to_currency: str,
) -> dict:
    """Calculates how much the recipient will receive after fees and exchange.

    This tool applies the exchange rate and transfer fees to determine
    the final amount the recipient receives. The LLM must never attempt
    to compute this on its own. Even small rounding errors are unacceptable
    in a remittance context.

    Args:
        sender_amount: The amount being sent by the sender (in sender's currency).
        exchange_rate: The exchange rate (how much recipient currency per 1 sender currency).
            This should come from the get_exchange_rate tool.
        transfer_fee_pct: The transfer fee as a percentage (e.g. 2.5 means 2.5%).
        transfer_fee_min: The minimum transfer fee (flat fee in sender's currency).
        from_currency: The sender's currency code (e.g. 'USD').
        to_currency: The recipient's currency code (e.g. 'INR').

    Returns:
        A dictionary with 'status' and either a 'result' containing
        the recipient amount and fee breakdown, or an 'error_message'.
    """
    if sender_amount <= 0:
        return {"status": "error", "error_message": "sender_amount must be a positive number."}
    if exchange_rate <= 0:
        return {"status": "error", "error_message": "exchange_rate must be a positive number."}
    if transfer_fee_pct < 0 or transfer_fee_pct > 50:
        return {"status": "error", "error_message": "transfer_fee_pct must be between 0 and 50."}
    if transfer_fee_min < 0:
        return {"status": "error", "error_message": "transfer_fee_min must be a non-negative number."}

    # Calculate the actual transfer fee (percentage or minimum, whichever is greater)
    fee_by_percentage = sender_amount * (transfer_fee_pct / 100)
    actual_fee = max(fee_by_percentage, transfer_fee_min)

    # Amount after fee deduction (in sender's currency)
    amount_after_fee = sender_amount - actual_fee

    # Convert to recipient's currency using exchange rate
    recipient_amount = amount_after_fee * exchange_rate

    return {
        "status": "success",
        "result": {
            "sender_sends": round(sender_amount, 2),
            "from_currency": from_currency,
            "transfer_fee": round(actual_fee, 2),
            "fee_percentage_used": transfer_fee_pct,
            "amount_after_fee": round(amount_after_fee, 2),
            "exchange_rate": exchange_rate,
            "recipient_receives": round(recipient_amount, 2),
            "to_currency": to_currency,
        },
    }


# ---------------------------------------------------------------------------
# Gemini client & chat session (cached across Streamlit reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_client():
    """Create the genai client once."""
    return genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))


def get_chat():
    """Create a chat session once per Streamlit browser session."""
    if "chat" not in st.session_state:
        client = get_client()
        st.session_state.chat = client.chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[get_exchange_rate, calculate_remittance_amount],
                # Automatic function calling: the SDK calls our Python
                # functions when Gemini requests them, and feeds results
                # back to the model — no manual loop needed.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    maximum_remote_calls=10,
                ),
            ),
        )
    return st.session_state.chat


def send_message(user_text: str) -> str:
    """Send a message to the Gemini chat and return the final text response."""
    chat = get_chat()
    response = chat.send_message(user_text)
    return response.text if response.text else "I wasn't able to generate a response. Please try again."


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Remittance & Exchange Advisor",
        page_icon="💱",
        layout="centered",
    )

    st.title("💱 Remittance & Exchange Advisor")
    st.caption(
        "Powered by Gemini  •  Exchange rates from a governed rate sheet  •  "
        "Fees calculated with precision"
    )

    # -- Sidebar with quick-start info ------------------------------------
    with st.sidebar:
        st.header("ℹ️ How it works")
        st.markdown(
            "This agent has **two tools**:\n\n"
            "1. **Exchange Rate Lookup** — reads today's exchange rates and fees from a CSV data sheet "
            "(the platform's source of truth).\n"
            "2. **Remittance Calculator** — calculates the exact recipient amount after fees and exchange. "
            "The LLM never does this math itself.\n\n"
            "The LLM decides *what* to look up and *when* to calculate. "
            "The tools ensure *how* it's done is governed and exact."
        )
        st.divider()
        st.subheader("💬 Try these prompts")
        st.code("What's the exchange rate from USD to INR?", language=None)
        st.code("If I send $500 to India, how much will the recipient get?", language=None)
        st.code("Compare sending $1000 to Philippines vs India from the US.", language=None)
        st.code("What are the available remittance corridors?", language=None)

        st.divider()
        st.subheader("🌍 Supported corridors")
        st.markdown(
            "**Senders:** United States (USD), Canada (CAD), Europe (EUR), "
            "United Kingdom (GBP), Australia (AUD)\n\n"
            "**Recipients:** India (INR), Philippines (PHP), Mexico (MXN), "
            "Pakistan (PKR), Nigeria (NGN), Brazil (BRL)"
        )

    # -- Chat history in session state ------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome! I can help you send money internationally and understand "
                    "exchange rates. Tell me where you're sending from and where you're "
                    "sending to, along with the amount, and I'll look up today's exchange "
                    "rate and fees, then calculate exactly how much the recipient will receive."
                ),
            }
        ]

    # -- Display existing messages ----------------------------------------
    # Streamlit's markdown renderer treats $...$ as LaTeX math, which
    # garbles any response containing dollar amounts. Replacing with the
    # HTML entity &#36; prevents LaTeX interpretation entirely.
    # We also strip markdown headers (##) that Gemini sometimes adds,
    # which render in a larger font and break visual consistency.
    def display(text: str):
        import re
        cleaned = text.replace("$", "&#36;")
        cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
        st.markdown(cleaned, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            display(msg["content"])

    # -- Handle new input -------------------------------------------------
    if prompt := st.chat_input("Ask about exchange rates or remittances..."):
        # Show the user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            display(prompt)

        # Get the agent's response
        with st.chat_message("assistant"):
            with st.spinner("Looking that up..."):
                response = send_message(prompt)
            display(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
