# Remittance & Exchange Advisor

A self-built, production-oriented conversational remittance advisor that demonstrates how to keep pricing data and regulatory calculations deterministic while leveraging an LLM for the conversation layer. The toolset reads governed data and computes exact transfer outcomes without relying on model-generated numbers.

Built for fintech and remittance professionals (product managers, engineers, compliance teams) who want to understand the agent pattern: the LLM handles the conversation; the tools perform data access and math.

## The Core Idea

In regulated domains, the most sensitive data and calculations should be governed by deterministic tools rather than the model. This project demonstrates delegating data access and arithmetic to explicit tools, while the language model focuses on user interaction and workflow orchestration.

| Tool                          | What It Does                                                               | Why It's a Tool                                                                                                                              |
| ----------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_exchange_rate`           | Looks up the current exchange rate and transfer fees from a CSV data sheet | Exchange rates and fees are governed data — they change and must come from the platform's source of truth, not the model's training data |
| `calculate_remittance_amount` | Calculates the exact amount the recipient receives after fees and exchange | Remittance math must be deterministic and exact — even small rounding errors are unacceptable in payments contexts                          |

The `google-genai` SDK's automatic function calling handles orchestration: you pass Python functions as tools, and the SDK auto-generates the schemas from your docstrings, calls your functions when Gemini requests them, and feeds the results back to the model.

## Setup and Local Development

### Quick Start (Local Development)

Setup Instructions

```bash
# 1. Clone or download this project

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
#    a. Copy the template file
cp .env.example .env

#    b. Get your API key from: https://aistudio.google.com/apikey
#
#    c. Edit .env and replace 'your_api_key_here' with your actual API key
#       GOOGLE_API_KEY=<your_actual_api_key>
#
#    ⚠️  IMPORTANT: Never commit .env to version control!
#        .env is in .gitignore and will be excluded automatically.

# 5. Run the app locally
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Demo

<img width="1201" height="1202" alt="exchange agent demo_1" src="https://github.com/user-attachments/assets/980ccb79-2464-45fd-bce7-909c46f46a6f" />

### Environment Variables

The app uses the following environment variable (managed by `python-dotenv`):

| Variable         | Purpose                             | Example                                      |
| ---------------- | ----------------------------------- | -------------------------------------------- |
| `GOOGLE_API_KEY` | Gemini API key for LLM interactions | `abc123xyz...` (from your Google AI account) |

**Security Notes:**

- The `.env` file is automatically excluded from version control (see `.gitignore`)
- Each developer should create their own `.env` file locally with their own API key
- Never share or commit your `.env` file

## Project Structure

```
remittance_exchange_advisor/
├── app.py                          # Streamlit chat UI (google-genai SDK)
├── requirements.txt                # Python dependencies
├── .env                            # API key (local dev only — create from .env.example)
├── .env.example                    # Template for .env (safe to commit)
├── .gitignore                      # Excludes .env and other sensitive files
├── rate_sheet.csv                  # Governed exchange rate & fee data (15+ corridors)
└── README.md                       # This file
```

## Example Prompts

**Single tool call — exchange rate lookup:**

> What's the exchange rate from USD to INR?

**Tool chaining — rate lookup then amount calculation:**

> If I send $500 to India, how much will the recipient get?

**Multiple chained calls with comparison:**

> Compare sending $1000 to Philippines vs India from the United States.

**Error handling — unsupported corridor:**

> What about sending from South Africa to Brazil?

## How It Works

The entire integration is ~15 lines of code:

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=...)

chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[get_exchange_rate, calculate_remittance_amount],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=10,
        ),
    ),
)

response = chat.send_message("If I send $100 USD to India, how much will they receive?")
print(response.text)
```

When a user asks about a remittance, Gemini reasons through the request, calls `get_exchange_rate` to look up the exchange rate and fees, receives the result, then calls `calculate_remittance_amount` with those values, and finally summarizes everything in a natural language response. The SDK handles the full loop automatically.

## The Exchange Rate Sheet

The exchange rate sheet is a simple CSV with remittance corridors, exchange rates, and transfer fees:

| from_country  | from_currency | to_country  | to_currency | exchange_rate | transfer_fee_pct | transfer_fee_min |
| ------------- | ------------- | ----------- | ----------- | ------------- | ---------------- | ---------------- |
| United States | USD           | India       | INR         | 83.50         | 2.5              | 3.0              |
| United States | USD           | Philippines | PHP         | 57.80         | 2.0              | 2.5              |
| Canada        | CAD           | India       | INR         | 61.80         | 2.8              | 3.5              |
| Europe        | EUR           | Philippines | PHP         | 63.00         | 2.0              | 2.5              |
| ...           | ...           | ...         | ...         | ...           | ...              | ...              |

In production, `get_exchange_rate` might call an external currency API (like Wise, Remitly, or a central bank), a Snowflake query, or an internal database. The tool interface stays the same — you just swap the implementation.

## Docstrings Matter

The `google-genai` SDK reads your function's docstring and type hints to auto-generate the tool schema that Gemini sees. This means:

- The **description** tells Gemini when to call the tool
- The **Args** section becomes the parameter schema (names, types, valid values)
- The **Returns** section tells Gemini what to expect back

Better docstrings = more reliable tool calls. It's prompt engineering applied to function signatures.

## Deploy on Replit (Optional)

If you prefer to run the app on Replit instead of locally:

1. Create a new Replit — choose "Import from GitHub" or "Upload folder" and upload all the files.
2. In the Replit sidebar, go to **Secrets** (the lock icon) and add:
   - Key: `GOOGLE_API_KEY`
   - Value: your Google AI Studio API key
3. Hit **Run**. Streamlit will launch automatically.
4. Click **Deploy** to get a public URL.

**Note:** For local development, use the **Quick Start** section above instead.

## Extending This

- **Add more corridors** — expand the exchange rate sheet with new sender/recipient countries and currency pairs.
- **Add promotional rates** — add a tool that checks if the sender/recipient qualifies for special rates based on volume or loyalty.
- **Swap the data source** — replace the CSV with a Snowflake query (`pd.read_sql()`), REST API call (Wise, xe.com), or database lookup. The tool interface stays the same.
- **Add compliance checks** — add a tool that checks AML/KYC rules for high-risk corridors or large amounts.
- **Add session memory** — maintain conversation context and user preferences (preferred corridors, saved recipients).
- **Add multi-currency support** — extend the tools to handle more complex scenarios like converting between three currencies (e.g., GBP → USD → INR).