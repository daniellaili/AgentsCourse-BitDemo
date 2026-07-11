# AgentsCourse-BitDemo

## Assignment Implementation

This project implements the Bit-like payment agents assignment in Python.

It includes:
- `RouterAgent` and `OrchestratorAgent`
- `PaymentSystem` with users, wallets, transactions, payment requests, and audit log
- short-term business memory
- `FraudDetectionAgent`, `SecurityAgent`, `ExplanationAgent`, `CriticAgent`, and `FallbackAgent`
- two advanced elements: `PolicyAgent` and JSON save/load
- a demo with the required assignment test cases
- OpenAI Agents SDK usage for natural-language inference, tools, and handoffs
- short-term memory context for follow-up questions such as "its balance" or "the last transaction"

## Setup

Use Python 3.10 or newer. Free-speech mode requires the OpenAI Agents SDK and an OpenAI API key.

Create a `.env` file and set your OpenAI key:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
```

Optional environment settings:

```env
# Enable only if you intentionally need a proxy for OpenAI requests.
OPENAI_USE_PROXY=true

# Use a custom CA bundle if your network requires one.
OPENAI_CA_BUNDLE=C:\path\to\your-ca-bundle.pem

# Tracing is disabled by default to avoid separate tracing upload/certificate noise.
OPENAI_AGENTS_TRACING=true
```

Install the SDKs:

```bash
pip install -r requirements.txt
```

Free-speech requests are handled by OpenAI SDK agents that use tools and handoffs. If the API key or Agents SDK package is unavailable, free-speech mode raises a clear error instead of silently falling back.

On Windows, the project uses `truststore` so Python can use the OS certificate store. It also ignores ambient `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` values by default because some dev environments set them to a dead local proxy. Set `OPENAI_USE_PROXY=true` only when you explicitly want those proxy variables honored.

## Run

```bash
python run_assignment_demo.py
```

You can also run the main implementation directly:

```bash
python agentic_payment_project.py
```

The demo prints the required test results and writes `payment_state.json` to demonstrate JSON persistence.

## Free-Speech Interactive Mode

Run the system as an interactive natural-language chatbot:

```bash
python interactive_payment_chat.py
```

Example messages:

```text
Create a user named Alice with phone 050-1111111 and balance 8000
Add Bob, phone 050-2222222, with 500 shekels
Send 150 from U001 to U002
Explain the last transaction
```

In free-speech mode, `OrchestratorAgent` delegates to OpenAI SDK agents with handoffs between policy, payment, fraud, security, explanation, and fallback agents. If the SDK runtime is unavailable, the command exits with a clear runtime error.

The chat keeps short-term business memory inside the active `OrchestratorAgent` instance. For example, after creating Alice as `U001`, a follow-up like "What is its balance?" resolves to `U001` and still calls the balance tool for the current value.

## Gradio Notebook

The notebook [`agentic_payment_chat_gradio.ipynb`](agentic_payment_chat_gradio.ipynb) provides a Gradio chat UI. Run the notebook cells from the top, or restart the kernel and rerun the final UI cell after code changes so it reloads `agentic_payment_project.py`.

The final UI cell catches exceptions and returns a JSON object with `error_type` and a short traceback tail, which is easier to debug than Gradio's generic `error` message.
