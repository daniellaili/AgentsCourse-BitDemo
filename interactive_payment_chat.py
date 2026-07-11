from agentic_payment_project import OrchestratorAgent, compact
import json


HELP_TEXT = """
Free-speech Bit-like payment agent

Write naturally, for example:
  Create a user named Alice with phone 050-1111111 and balance 8000
  Add Bob, phone 050-2222222, with 500 shekels
  Send 150 from U001 to U002
  What is the balance of U001?
  Ask U002 to pay U001 75
  Approve request R0001
  Show the transaction history for U001
  Was the last transaction suspicious?
  Explain the last transaction
  Run a security review

Type help to show this text, or quit to exit.
""".strip()


def coerce_answer(field_name: str, answer: str):
    if field_name in {"amount", "initial_balance"}:
        return float(answer)
    return answer


def fill_missing_fields(parsed_output: dict) -> dict:
    parameters = dict(parsed_output["parsed_parameters"])
    for field_name in parsed_output["missing_fields"]:
        while True:
            answer = input(f"{field_name}: ").strip()
            if not answer:
                print("This field is required.")
                continue
            try:
                parameters[field_name] = coerce_answer(field_name, answer)
                break
            except ValueError:
                print("Please enter a valid number.")
    return parameters


def main() -> None:
    orch = OrchestratorAgent()
    print(HELP_TEXT)

    while True:
        user_input = input("\nYou> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break
        if user_input.lower() in {"help", "?"}:
            print(HELP_TEXT)
            continue

        result = orch.run_free_speech(user_input)
        if (
            result.agent_name == "FreeSpeechParserAgent"
            and isinstance(result.output, dict)
            and result.output.get("missing_fields")
        ):
            parameters = fill_missing_fields(result.output)
            result = orch.run(result.output["intent"], **parameters)

        print("\nAgent>")
        print(json.dumps(compact(result), indent=2, ensure_ascii=False))
        orch.payment_system.save_to_json("payment_state.json")


if __name__ == "__main__":
    main()
