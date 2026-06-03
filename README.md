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
- optional OpenAI SDK usage for chatbot-style routing/explanations

## Setup

Create `.env` from `.env.example` and set your OpenAI key:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
```

Install the SDK:

```bash
pip install openai
```

The project still runs without an API key or without the SDK installed. In that case it uses deterministic local fallback logic.

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

`FreeSpeechParserAgent` parses the message into an intent and parameter schema, then `OrchestratorAgent` routes it to the matching agent. If the parser is missing a required field, the CLI asks only for that field.
