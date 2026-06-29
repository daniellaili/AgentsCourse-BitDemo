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
- deterministic local fallback logic when no API key or Agents SDK package is available

## Setup

Use Python 3.10 or newer for the OpenAI Agents SDK path. Older Python versions can still run the deterministic fallback.

Create `.env` from `.env.example` and set your OpenAI key:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
```

Install the SDKs:

```bash
pip install -r requirements.txt
```

With `OPENAI_API_KEY` and `openai-agents` installed, free-speech requests are handled by OpenAI SDK agents that use tools and handoffs. The deterministic parser/router is kept only as a fallback when the API key or Agents SDK package is unavailable.

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

In normal API-backed mode, `OrchestratorAgent` delegates to OpenAI SDK agents with handoffs between policy, payment, fraud, security, explanation, and fallback agents. If the API-backed agent runtime is unavailable, `FreeSpeechParserAgent` uses the local fallback parser and the CLI asks only for missing fields.
