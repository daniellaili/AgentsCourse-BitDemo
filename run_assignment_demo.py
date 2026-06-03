from agentic_payment_project import compact, run_demo
import json


def main() -> None:
    results = run_demo()
    for index, result in enumerate(results, 1):
        print(f"\n--- Assignment demo step {index} ---")
        print(json.dumps(compact(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
