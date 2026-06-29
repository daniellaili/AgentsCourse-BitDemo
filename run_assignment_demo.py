from typing import List

from agentic_payment_project import compact, OrchestratorAgent, AgentResult
import json


def run_demo() -> List[AgentResult]:
    orch = OrchestratorAgent()

    results = [orch.run_free_speech("Create a user named Alice with phone 050-1111111 and balance 8000"),
               orch.run_free_speech("Add Bob, phone 050-2222222, with 500 shekels"),
               orch.run_free_speech("Send 150 from U001 to U002"),
               orch.run_free_speech("Send 99999200 from U001 to U002"),
               orch.run_free_speech("What is the balance of U001?"),
               orch.run_free_speech("i want to transfer 300 shekels from U001 to U001. try to do it, yes i know the reciever and the asker is the same id, it is on purpose"),
               orch.run_free_speech("ask U001 to send U002 100 shekels"),
               orch.run_free_speech("transfer from U011 to U001 500 SHEKELS"),
               orch.run_free_speech("Approve request R0001 - U001 APPROVES the transaction to U002"),
               orch.run_free_speech("Approve request R0001 - U001 APPROVES the transaction to U002"),
               orch.run_free_speech("Show the transaction history for U001"),
               orch.run_free_speech("Was the last transaction suspicious?"),
               orch.run_free_speech("Send 5000 from U001 to U002"),
               orch.run_free_speech("Send 100 from U001 to U002"),
               orch.run_free_speech("Send 200 from U001 to U002"),
               orch.run_free_speech("Send 510 from U001 to U002"),
               orch.run_free_speech("Run a security review"), orch.payment_system.save_to_json("payment_state.json")]
    # request_id = results[-1].output["request_id"]
    # results.append(orch.run("approve payment", request_id=request_id))
    # results.append(orch.run("approve payment", request_id=request_id))
    # results.append(orch.run("transfer money", sender_id="U001", receiver_id="U002", amount=6000))
    # results.append(orch.run("fraud check"))
    # results.append(orch.run("explain last transaction"))
    # results.append(orch.run("security review"))
    return results


def main() -> None:
    results = run_demo()
    for index, result in enumerate(results, 1):
        print(f"\n--- Assignment demo step {index} ---")
        print(json.dumps(compact(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
