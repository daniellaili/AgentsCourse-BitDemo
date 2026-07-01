from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os
import re
import sys
from openai import OpenAI

try:
    from agents import Agent as OpenAIAgent
    from agents import Runner as OpenAIAgentRunner
    from agents import function_tool
    from agents import RunHooks as OpenAIRunHooks
except Exception:
    OpenAIAgent = None
    OpenAIAgentRunner = None
    OpenAIRunHooks = None

    def function_tool(func):
        return func

DEFAULT_MODEL = "gpt-4.1-mini"


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class OpenAIChatbot:
    """Optional OpenAI SDK wrapper with local deterministic fallback."""

    def __init__(self, model: Optional[str] = None):
        load_env_file()
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.enabled = False
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY", "").strip()

        if not api_key:
            return

        try:
            self.client = OpenAI(api_key=api_key)
            self.enabled = True
        except Exception as e:
            self.client = None
            self.enabled = False

    def ask(self, system_prompt: str, user_prompt: str, fallback: str) -> str:
        if not self.enabled or self.client is None:
            return fallback

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = getattr(response, "output_text", "")
            return text.strip() or fallback
        except Exception:
            return fallback


@dataclass
class ParsedUserRequest:
    intent: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class AgentResult:
    agent_name: str
    output: Any
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class User:
    user_id: str
    name: str
    phone_number: str


@dataclass
class Wallet:
    user_id: str
    balance: float


@dataclass
class Transaction:
    transaction_id: str
    sender_id: str
    receiver_id: str
    amount: float
    timestamp: str
    status: str
    risk_score: float = 0.0
    reason: str = ""


@dataclass
class PaymentRequest:
    request_id: str
    requester_id: str
    payer_id: str
    amount: float
    status: str
    created_at: str


@dataclass
class ShortTermMemory:
    limit: int = 10
    last_action: Optional[str] = None
    last_user: Optional[str] = None
    last_transaction: Optional[Transaction] = None
    last_payment_request: Optional[PaymentRequest] = None
    last_result: Optional[AgentResult] = None
    recent_actions: List[Dict[str, Any]] = field(default_factory=list)

    def update(
        self,
        action: str,
        result: AgentResult,
        user_id: Optional[str] = None,
        transaction: Optional[Transaction] = None,
        payment_request: Optional[PaymentRequest] = None,
    ) -> None:
        self.last_action = action
        self.last_result = result
        if user_id:
            self.last_user = user_id
        if transaction:
            self.last_transaction = transaction
        if payment_request:
            self.last_payment_request = payment_request
        self.recent_actions.append(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "action": action,
                "result": result.output,
            }
        )
        self.recent_actions = self.recent_actions[-self.limit:]


class PaymentSystem:
    def __init__(self, policy_agent: Optional[Any] = None):
        self.users: Dict[str, User] = {}
        self.wallets: Dict[str, Wallet] = {}
        self.transactions: List[Transaction] = []
        self.payment_requests: Dict[str, PaymentRequest] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self.policy_agent = policy_agent
        self._user_counter = 1
        self._transaction_counter = 1
        self._request_counter = 1

    def _log(self, action: str, details: Dict[str, Any]) -> None:
        self.audit_log.append(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "action": action,
                "details": details,
            }
        )

    def create_user(self, name: str, phone_number: str, initial_balance: float) -> AgentResult:
        if initial_balance < 0:
            result = AgentResult("PaymentSystem", "Cannot create a user with a negative initial balance.", 0.95)
            self._log("create_user_failed", {"name": name, "reason": result.output})
            return result

        user_id = f"U{self._user_counter:03d}"
        self._user_counter += 1
        user = User(user_id, name, phone_number)
        self.users[user_id] = user
        self.wallets[user_id] = Wallet(user_id, float(initial_balance))
        output = {"user": asdict(user), "balance": initial_balance}
        self._log("create_user", output)
        return AgentResult("PaymentSystem", output)

    def get_balance(self, user_id: str) -> AgentResult:
        if user_id not in self.wallets:
            return AgentResult("PaymentSystem", f"User {user_id} does not exist.", 0.9)
        balance = self.wallets[user_id].balance
        self._log("check_balance", {"user_id": user_id, "balance": balance})
        return AgentResult("PaymentSystem", {"user_id": user_id, "balance": balance})

    def transfer_money(self, sender_id: str, receiver_id: str, amount: float) -> AgentResult:
        validation_error = self._validate_transfer(sender_id, receiver_id, amount)
        if validation_error:
            transaction = self._record_transaction(sender_id, receiver_id, amount, "rejected", validation_error)
            return AgentResult("PaymentSystem", validation_error, 0.95, {"transaction": asdict(transaction)})

        policy_result = self._check_transfer_policy(sender_id, amount)
        if policy_result:
            return policy_result

        self.wallets[sender_id].balance -= amount
        self.wallets[receiver_id].balance += amount
        transaction = self._record_transaction(sender_id, receiver_id, amount, "approved", "Transfer completed.")
        output = {
            "message": "Transfer approved.",
            "transaction": asdict(transaction),
            "sender_balance": self.wallets[sender_id].balance,
            "receiver_balance": self.wallets[receiver_id].balance,
        }
        return AgentResult("PaymentSystem", output, 1.0, {"transaction": asdict(transaction)})

    def _validate_transfer(self, sender_id: str, receiver_id: str, amount: float) -> Optional[str]:
        if amount <= 0:
            return "Cannot transfer a zero or negative amount."
        if sender_id not in self.users:
            return f"Sender {sender_id} does not exist."
        if receiver_id not in self.users:
            return f"Receiver {receiver_id} does not exist."
        if sender_id == receiver_id:
            return "Cannot transfer money to yourself."
        if self.wallets[sender_id].balance < amount:
            return "Insufficient balance."
        return None

    def _check_transfer_policy(self, sender_id: str, amount: float) -> Optional[AgentResult]:
        if self.policy_agent is None:
            return None
        result = self.policy_agent.check_transfer(sender_id, float(amount), self)
        if self.policy_agent.is_approved(result):
            return None
        return result

    def _record_transaction(
        self, sender_id: str, receiver_id: str, amount: float, status: str, reason: str
    ) -> Transaction:
        transaction = Transaction(
            transaction_id=f"T{self._transaction_counter:04d}",
            sender_id=sender_id,
            receiver_id=receiver_id,
            amount=float(amount),
            timestamp=datetime.now().isoformat(timespec="seconds"),
            status=status,
            reason=reason,
        )
        self._transaction_counter += 1
        self.transactions.append(transaction)
        self._log("transfer_money", asdict(transaction))
        return transaction

    def get_transactions(self, user_id: str) -> AgentResult:
        if user_id not in self.users:
            return AgentResult("PaymentSystem", f"User {user_id} does not exist.", 0.9)
        rows = [
            asdict(tx)
            for tx in self.transactions
            if tx.sender_id == user_id or tx.receiver_id == user_id
        ]
        self._log("show_transactions", {"user_id": user_id, "count": len(rows)})
        return AgentResult("PaymentSystem", rows)

    def request_payment(self, requester_id: str, payer_id: str, amount: float) -> AgentResult:
        if amount <= 0:
            return AgentResult("PaymentSystem", "Cannot request a zero or negative amount.", 0.95)
        if requester_id not in self.users or payer_id not in self.users:
            return AgentResult("PaymentSystem", "Requester or payer does not exist.", 0.95)
        if requester_id == payer_id:
            return AgentResult("PaymentSystem", "Cannot request payment from yourself.", 0.95)

        request = PaymentRequest(
            request_id=f"R{self._request_counter:04d}",
            requester_id=requester_id,
            payer_id=payer_id,
            amount=float(amount),
            status="pending",
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        self._request_counter += 1
        self.payment_requests[request.request_id] = request
        self._log("request_payment", asdict(request))
        return AgentResult("PaymentSystem", asdict(request), metadata={"payment_request": asdict(request)})

    def approve_payment_request(self, request_id: str) -> AgentResult:
        request = self.payment_requests.get(request_id)
        if request is None:
            return AgentResult("PaymentSystem", f"Payment request {request_id} does not exist.", 0.9)
        if request.status != "pending":
            return AgentResult("PaymentSystem", f"Payment request {request_id} is already {request.status}.", 0.95)

        policy_result = self._check_transfer_policy(request.payer_id, request.amount)
        if policy_result:
            self._log("approve_payment_request_failed", {"request_id": request_id, "reason": policy_result.output})
            return policy_result

        result = self.transfer_money(request.payer_id, request.requester_id, request.amount)
        if isinstance(result.output, dict) and result.output.get("transaction"):
            request.status = "approved"
        else:
            request.status = "rejected"
        self._log("approve_payment_request", asdict(request))
        return AgentResult(
            "PaymentSystem",
            {"payment_request": asdict(request), "transfer_result": result.output},
            result.confidence,
            {"payment_request": asdict(request), **(result.metadata or {})},
        )

    def reject_payment_request(self, request_id: str) -> AgentResult:
        request = self.payment_requests.get(request_id)
        if request is None:
            return AgentResult("PaymentSystem", f"Payment request {request_id} does not exist.", 0.9)
        if request.status != "pending":
            return AgentResult("PaymentSystem", f"Payment request {request_id} is already {request.status}.", 0.95)
        request.status = "rejected"
        self._log("reject_payment_request", asdict(request))
        return AgentResult("PaymentSystem", asdict(request), metadata={"payment_request": asdict(request)})

    def save_to_json(self, path: str) -> AgentResult:
        data = {
            "users": {k: asdict(v) for k, v in self.users.items()},
            "wallets": {k: asdict(v) for k, v in self.wallets.items()},
            "transactions": [asdict(tx) for tx in self.transactions],
            "payment_requests": {k: asdict(v) for k, v in self.payment_requests.items()},
            "audit_log": self.audit_log,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        return AgentResult("PaymentSystem", f"Saved system state to {path}.")

    def load_from_json(self, path: str) -> AgentResult:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.users = {k: User(**v) for k, v in data.get("users", {}).items()}
        self.wallets = {k: Wallet(**v) for k, v in data.get("wallets", {}).items()}
        self.transactions = [Transaction(**v) for v in data.get("transactions", [])]
        self.payment_requests = {
            k: PaymentRequest(**v) for k, v in data.get("payment_requests", {}).items()
        }
        self.audit_log = data.get("audit_log", [])
        self._user_counter = self._next_counter(self.users, "U")
        self._transaction_counter = self._next_counter({tx.transaction_id: tx for tx in self.transactions}, "T")
        self._request_counter = self._next_counter(self.payment_requests, "R")
        return AgentResult("PaymentSystem", f"Loaded system state from {path}.")

    @staticmethod
    def _next_counter(items: Dict[str, Any], prefix: str) -> int:
        numbers = [int(k.removeprefix(prefix)) for k in items if k.startswith(prefix) and k.removeprefix(prefix).isdigit()]
        return max(numbers, default=0) + 1


class RouterAgent:
    def __init__(self, chatbot: Optional[OpenAIChatbot] = None):
        self.chatbot = chatbot or OpenAIChatbot()

    def route(self, message: str) -> AgentResult:
        text = message.lower()
        rules = [
            ("createUser", ["create user", "new user", "add user", "יצירת משתמש"]),
            ("checkBalance", ["balance", "יתרה"]),
            ("transferMoney", ["transfer", "send money", "pay ", "העברת"]),
            ("requestPayment", ["request payment", "payment request", "בקשת תשלום"]),
            ("approvePayment", ["approve", "אישור"]),
            ("rejectPayment", ["reject", "דחייה", "decline"]),
            ("showTransactions", ["transactions", "history", "היסטוריית"]),
            ("fraudCheck", ["fraud", "suspicious", "חשוד"]),
            ("securityReview", ["security", "audit", "אבטחה"]),
            ("explainLastAction", ["explain", "last action", "last transaction", "הסבר"]),
        ]
        for intent, keywords in rules:
            if any(keyword in text for keyword in keywords):
                return AgentResult("RouterAgent", intent)

        prompt = "Return one intent only from the project intent list for this user message."
        fallback = "unknown"
        intent = self.chatbot.ask(prompt, message, fallback).strip()
        valid = {intent for intent, _ in rules} | {"unknown"}
        if intent not in valid:
            intent = "unknown"
        return AgentResult("RouterAgent", intent, 0.75 if intent != "unknown" else 0.4)


class FreeSpeechParserAgent:
    INTENT_FIELDS = {
        "createUser": ["name", "phone_number", "initial_balance"],
        "checkBalance": ["user_id"],
        "transferMoney": ["sender_id", "receiver_id", "amount"],
        "requestPayment": ["requester_id", "payer_id", "amount"],
        "approvePayment": ["request_id"],
        "rejectPayment": ["request_id"],
        "showTransactions": ["user_id"],
        "fraudCheck": [],
        "securityReview": [],
        "explainLastAction": [],
        "unknown": [],
    }

    def __init__(self, chatbot: Optional[OpenAIChatbot] = None):
        self.chatbot = chatbot or OpenAIChatbot()
        self.router = RouterAgent(self.chatbot)

    def parse(self, message: str, memory: Optional[ShortTermMemory] = None) -> AgentResult:
        parsed = self._parse_with_openai(message)
        if parsed is None or parsed.intent == "unknown":
            parsed = self._parse_locally(message, memory)

        parsed.missing_fields = self._missing_fields(parsed)
        if parsed.missing_fields:
            local_parsed = self._parse_locally(message, memory)
            local_parsed.missing_fields = self._missing_fields(local_parsed)
            if local_parsed.intent == parsed.intent and len(local_parsed.missing_fields) < len(parsed.missing_fields):
                parsed.parameters = {**local_parsed.parameters, **parsed.parameters}
                parsed.missing_fields = self._missing_fields(parsed)

        return AgentResult(
            "FreeSpeechParserAgent",
            {
                "intent": parsed.intent,
                "parameters": parsed.parameters,
                "missing_fields": parsed.missing_fields,
            },
            parsed.confidence,
        )

    def _parse_with_openai(self, message: str) -> Optional[ParsedUserRequest]:
        if not self.chatbot.enabled:
            return None

        schema = {
            "intent": "one of: createUser, checkBalance, transferMoney, requestPayment, approvePayment, rejectPayment, showTransactions, fraudCheck, securityReview, explainLastAction, unknown",
            "parameters": {
                "name": "string",
                "phone_number": "string",
                "initial_balance": "number",
                "user_id": "string like U001",
                "sender_id": "string like U001",
                "receiver_id": "string like U002",
                "requester_id": "string like U001",
                "payer_id": "string like U002",
                "request_id": "string like R0001",
                "amount": "number",
            },
            "confidence": "number from 0 to 1",
        }
        fallback = "{}"
        raw = self.chatbot.ask(
            "You parse natural language for a Bit-like payment agent. Return JSON only. "
            "Do not include markdown or explanations.",
            f"Schema: {json.dumps(schema)}\nUser message: {message}",
            fallback,
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        intent = data.get("intent", "unknown")
        if intent not in self.INTENT_FIELDS:
            intent = "unknown"
        params = self._normalize_parameters(data.get("parameters", {}))
        return ParsedUserRequest(intent, params, confidence=float(data.get("confidence", 0.8)))

    def _parse_locally(self, message: str, memory: Optional[ShortTermMemory]) -> ParsedUserRequest:
        intent = self._infer_intent(message)
        params = self._extract_common_fields(message)
        text = message.lower()

        if intent == "createUser":
            params.setdefault("name", self._extract_name(message))
            params.setdefault("phone_number", self._extract_phone(message))
            if "initial_balance" not in params and "amount" in params:
                params["initial_balance"] = params.pop("amount")
        elif intent == "checkBalance" or intent == "showTransactions":
            params.setdefault("user_id", self._first_user_id(message) or (memory.last_user if memory else None))
        elif intent == "transferMoney":
            users = self._all_user_ids(message)
            if len(users) >= 1:
                params.setdefault("sender_id", users[0].upper())
            if len(users) >= 2:
                params.setdefault("receiver_id", users[1].upper())
        elif intent == "requestPayment":
            users = self._all_user_ids(message)
            if len(users) >= 1:
                params.setdefault("requester_id", users[-1].upper())
            if len(users) >= 2:
                params.setdefault("payer_id", users[0].upper())
        elif intent in {"approvePayment", "rejectPayment"}:
            params.setdefault("request_id", self._first_request_id(message))

        params = {key: value for key, value in params.items() if value not in (None, "")}
        confidence = 0.75 if intent != "unknown" else 0.35
        if "last" in text and intent in {"explainLastAction", "fraudCheck"}:
            confidence = 0.9
        return ParsedUserRequest(intent, self._normalize_parameters(params), confidence=confidence)

    def _infer_intent(self, message: str) -> str:
        text = message.lower()
        if any(word in text for word in ["create", "add", "register", "new user"]) and any(
            word in text for word in ["user", "named", "called", "phone"]
        ):
            return "createUser"
        if any(word in text for word in ["balance", "how much money", "יתרה"]):
            return "checkBalance"
        if any(word in text for word in ["request", "ask"]) and any(word in text for word in ["pay", "payment"]):
            return "requestPayment"
        if any(word in text for word in ["approve", "accept", "confirm"]):
            return "approvePayment"
        if any(word in text for word in ["reject", "decline", "deny"]):
            return "rejectPayment"
        if any(word in text for word in ["history", "transactions", "activity"]):
            return "showTransactions"
        if any(word in text for word in ["fraud", "suspicious", "risk"]):
            return "fraudCheck"
        if any(word in text for word in ["security", "audit", "review"]):
            return "securityReview"
        if "explain" in text or "why" in text:
            return "explainLastAction"
        if any(word in text for word in ["send", "transfer", "pay", "move"]) and len(self._all_user_ids(message)) >= 2:
            return "transferMoney"
        return self.router.route(message).output

    def _missing_fields(self, parsed: ParsedUserRequest) -> List[str]:
        return [field_name for field_name in self.INTENT_FIELDS[parsed.intent] if field_name not in parsed.parameters]

    @staticmethod
    def _normalize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        aliases = {
            "phone": "phone_number",
            "balance": "initial_balance",
            "from_user": "sender_id",
            "to_user": "receiver_id",
            "sender": "sender_id",
            "receiver": "receiver_id",
            "requester": "requester_id",
            "payer": "payer_id",
            "user": "user_id",
            "request": "request_id",
        }
        normalized = {}
        for key, value in parameters.items():
            target = aliases.get(key, key)
            if target in {"amount", "initial_balance"} and value not in (None, ""):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    pass
            normalized[target] = value
        return normalized

    @staticmethod
    def _extract_common_fields(message: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for key, value in re.findall(r"(\w+)=(\"[^\"]+\"|'[^']+'|\S+)", message):
            params[key.lower()] = value.strip("\"'")

        amount_match = re.search(r"(?:amount|balance|for|of|with)\s+(-?\d+(?:\.\d+)?)", message, re.I)
        if amount_match:
            params.setdefault("amount", float(amount_match.group(1)))
        elif re.search(r"\b(send|transfer|pay|ask|request)\b", message, re.I):
            numbers = re.findall(r"(?<![A-Za-z0-9])(-?\d+(?:\.\d+)?)(?![A-Za-z0-9])", message)
            if numbers:
                params.setdefault("amount", float(numbers[-1]))
        return params

    @staticmethod
    def _all_user_ids(message: str) -> List[str]:
        return [user_id.upper() for user_id in re.findall(r"\bU\d{3,}\b", message, flags=re.I)]

    def _first_user_id(self, message: str) -> Optional[str]:
        users = self._all_user_ids(message)
        return users[0].upper() if users else None

    @staticmethod
    def _first_request_id(message: str) -> Optional[str]:
        match = re.search(r"\bR\d{3,}\b", message, flags=re.I)
        return match.group(0).upper() if match else None

    @staticmethod
    def _extract_phone(message: str) -> Optional[str]:
        match = re.search(r"\b0\d{1,2}[- ]?\d{6,7}\b", message)
        return match.group(0) if match else None

    @staticmethod
    def _extract_name(message: str) -> Optional[str]:
        match = re.search(r"(?:named|name is|user called|create user|add)\s+([A-Z][A-Za-z'-]*)", message, re.I)
        return match.group(1) if match else None


class ToolSelector:
    MAP = {
        "createUser": "PaymentSystem",
        "checkBalance": "PaymentSystem",
        "transferMoney": "PaymentSystem",
        "requestPayment": "PaymentSystem",
        "approvePayment": "PaymentSystem",
        "rejectPayment": "PaymentSystem",
        "showTransactions": "PaymentSystem",
        "fraudCheck": "FraudDetectionAgent",
        "securityReview": "SecurityAgent",
        "explainLastAction": "ExplanationAgent",
        "unknown": "FallbackAgent",
    }

    def select(self, intent: str) -> AgentResult:
        return AgentResult("ToolSelector", self.MAP.get(intent, "FallbackAgent"))


class FraudDetectionAgent:
    def review(
        self, transaction: Optional[Transaction], payment_system: PaymentSystem, sender_previous_balance: Optional[float] = None
    ) -> AgentResult:
        if transaction is None:
            return AgentResult("FraudDetectionAgent", "No transaction available for fraud check.", 0.6)

        risk_score = 0.0
        reasons = []
        if transaction.amount >= 5000:
            risk_score += 0.4
            reasons.append("high amount")

        if sender_previous_balance and transaction.amount / sender_previous_balance >= 0.75:
            risk_score += 0.35
            reasons.append("large percentage of sender balance")

        recent_sender_transactions = [
            tx for tx in payment_system.transactions[-5:] if tx.sender_id == transaction.sender_id
        ]
        if len(recent_sender_transactions) >= 3:
            risk_score += 0.25
            reasons.append("multiple recent operations")

        transaction.risk_score = min(risk_score, 1.0)
        if not reasons:
            reasons.append("no suspicious pattern")

        output = {
            "transaction_id": transaction.transaction_id,
            "risk_score": transaction.risk_score,
            "suspicious": transaction.risk_score >= 0.5,
            "reasons": reasons,
        }
        payment_system._log("fraud_check", output)
        return AgentResult("FraudDetectionAgent", output, 0.9)


class PolicyAgent:
    APPROVED_OUTPUT = "Policy approved."

    def __init__(
        self,
        max_single_transfer: float = 10000,
        max_daily_transfer: float = 15000,
        max_daily_transactions: int = 5,
    ):
        self.max_single_transfer = max_single_transfer
        self.max_daily_transfer = max_daily_transfer
        self.max_daily_transactions = max_daily_transactions

    def is_approved(self, result: AgentResult) -> bool:
        return result.output == self.APPROVED_OUTPUT

    def check_transfer(self, sender_id: str, amount: float, payment_system: PaymentSystem) -> AgentResult:
        if amount > self.max_single_transfer:
            return AgentResult("PolicyAgent", f"Transfer exceeds single-transfer limit of {self.max_single_transfer}.", 0.95)

        today = datetime.now().date().isoformat()
        daily_transactions = [
            tx
            for tx in payment_system.transactions
            if tx.sender_id == sender_id and tx.status == "approved" and tx.timestamp.startswith(today)
        ]
        if len(daily_transactions) >= self.max_daily_transactions:
            return AgentResult(
                "PolicyAgent",
                f"Transfer exceeds daily transaction limit of {self.max_daily_transactions}.",
                0.95,
            )

        daily_total = sum(tx.amount for tx in daily_transactions)
        if daily_total + amount > self.max_daily_transfer:
            return AgentResult("PolicyAgent", f"Transfer exceeds daily limit of {self.max_daily_transfer}.", 0.95)
        return AgentResult("PolicyAgent", self.APPROVED_OUTPUT)


class SecurityAgent:
    def review(self, payment_system: PaymentSystem) -> AgentResult:
        checks = {
            "users_have_wallets": set(payment_system.users) == set(payment_system.wallets),
            "audit_log_exists": len(payment_system.audit_log) > 0,
            "transactions_have_status": all(bool(tx.status) for tx in payment_system.transactions),
            "no_negative_wallets": all(wallet.balance >= 0 for wallet in payment_system.wallets.values()),
        }
        return AgentResult("SecurityAgent", {"checks": checks, "passed": all(checks.values())}, 0.9)


class ExplanationAgent:
    def __init__(self, chatbot: Optional[OpenAIChatbot] = None):
        self.chatbot = chatbot or OpenAIChatbot()

    def explain(self, memory: ShortTermMemory) -> AgentResult:
        if memory.last_transaction:
            tx = memory.last_transaction
            fallback = (
                f"Transaction {tx.transaction_id} from {tx.sender_id} to {tx.receiver_id} "
                f"for {tx.amount} was {tx.status}. Reason: {tx.reason}. Risk score: {tx.risk_score}."
            )
            prompt = f"Explain this Bit-like payment transaction briefly for a user: {asdict(tx)}"
            text = self.chatbot.ask("You explain payment-agent decisions clearly and briefly.", prompt, fallback)
            return AgentResult("ExplanationAgent", text, 0.9)

        if memory.last_payment_request:
            req = memory.last_payment_request
            return AgentResult(
                "ExplanationAgent",
                f"Payment request {req.request_id} is {req.status}: {req.requester_id} asked {req.payer_id} for {req.amount}.",
                0.85,
            )

        return AgentResult("ExplanationAgent", "There is no previous business action to explain.", 0.6)


class CriticAgent:
    def review(self, result: AgentResult) -> AgentResult:
        needs_fallback = result.confidence < 0.5 or result.output in (None, "", {})
        output = {"needs_fallback": needs_fallback, "reviewed_agent": result.agent_name}
        return AgentResult("CriticAgent", output, 0.9)


class FallbackAgent:
    def handle(self, message: str) -> AgentResult:
        return AgentResult(
            "FallbackAgent",
            {
                "message": "I could not understand the request.",
                "supported_actions": [
                    "create user",
                    "check balance",
                    "transfer money",
                    "request payment",
                    "approve payment",
                    "reject payment",
                    "show transactions",
                    "fraud check",
                    "security review",
                    "explain last transaction",
                ],
                "original_message": message,
            },
            0.7,
        )


class OpenAIAgentRuntime:
    """OpenAI Agents SDK runtime used for primary natural-language inference."""

    def __init__(
        self,
        payment_system: PaymentSystem,
        memory: ShortTermMemory,
        policy_agent: PolicyAgent,
        fraud_agent: FraudDetectionAgent,
        security_agent: SecurityAgent,
        explanation_agent: ExplanationAgent,
        fallback_agent: FallbackAgent,
        model: Optional[str] = None,
    ):
        load_env_file()
        self.enabled = bool(os.getenv("OPENAI_API_KEY", "").strip()) and sys.version_info >= (3, 10) and OpenAIAgent is not None
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.payment_system = payment_system
        self.memory = memory
        self.policy_agent = policy_agent
        self.fraud_agent = fraud_agent
        self.security_agent = security_agent
        self.explanation_agent = explanation_agent
        self.fallback_agent = fallback_agent
        self.tool_results: List[AgentResult] = []
        self.handoff_events: List[Dict[str, str]] = []
        self.primary_result: Optional[AgentResult] = None
        self.agents: Dict[str, Any] = {}
        self.disabled_reason = self._disabled_reason()

        if self.enabled:
            self._build_agents()

    def _disabled_reason(self) -> Optional[str]:
        if not os.getenv("OPENAI_API_KEY", "").strip():
            return "missing OPENAI_API_KEY"
        if sys.version_info < (3, 10):
            return "OpenAI Agents SDK requires Python 3.10 or newer"
        if OpenAIAgent is None:
            return "openai-agents package is not installed"
        return None

    def run(self, message: str) -> AgentResult:
        if not self.enabled:
            raise RuntimeError("OpenAI Agents SDK runtime is not available.")

        self.tool_results = []
        self.handoff_events = []
        self.primary_result = None
        hooks = self._handoff_hooks()
        result = OpenAIAgentRunner.run_sync(self.agents["intake"], message, hooks=hooks)
        final_text = str(getattr(result, "final_output", "")).strip()

        if self.primary_result is not None:
            output = self.primary_result
        elif self.tool_results:
            output = self.tool_results[-1]
        else:
            output = AgentResult("OpenAIAgentRuntime", final_text, 0.85)

        output.metadata = {
            **(output.metadata or {}),
            "sdk_final_output": final_text,
            "inference": "openai_agents_sdk",
            "handoffs": self.handoff_events,
            "tool_results": self._handoff_names(),
        }
        return output

    def _handoff_hooks(self) -> Optional[Any]:
        if OpenAIRunHooks is None:
            return None

        runtime = self

        class HandoffRecorder(OpenAIRunHooks):
            async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
                runtime.handoff_events.append(
                    {
                        "from": getattr(from_agent, "name", str(from_agent)),
                        "to": getattr(to_agent, "name", str(to_agent)),
                    }
                )

        return HandoffRecorder()

    def _record(self, result: AgentResult, primary: bool = False) -> Dict[str, Any]:
        self.tool_results.append(result)
        if primary or self.primary_result is None:
            self.primary_result = result
        return compact(result)

    def _handoff_names(self) -> List[str]:
        return [result.agent_name for result in self.tool_results]

    @staticmethod
    def _transaction_from_result(result: Optional[AgentResult]) -> Optional[Transaction]:
        if result is None:
            return None
        metadata = result.metadata or {}
        tx_data = metadata.get("transaction")
        if not tx_data and isinstance(result.output, dict):
            tx_data = result.output.get("transaction")
        if isinstance(tx_data, dict):
            return Transaction(**tx_data)
        return None

    def _build_agents(self) -> None:
        @function_tool
        def create_user(name: str, phone_number: str, initial_balance: float = 0) -> Dict[str, Any]:
            """Create a new wallet user."""
            return self._record(self.payment_system.create_user(name, phone_number, initial_balance), primary=True)

        @function_tool
        def check_balance(user_id: str) -> Dict[str, Any]:
            """Return a user's current wallet balance."""
            return self._record(self.payment_system.get_balance(user_id), primary=True)

        @function_tool
        def transfer_money(sender_id: str, receiver_id: str, amount: float) -> Dict[str, Any]:
            """Execute a transfer. This tool is only available after PolicyAgent handoff."""
            return self._record(self.payment_system.transfer_money(sender_id, receiver_id, amount), primary=True)

        @function_tool
        def request_payment(requester_id: str, payer_id: str, amount: float) -> Dict[str, Any]:
            """Create a payment request from requester to payer."""
            return self._record(self.payment_system.request_payment(requester_id, payer_id, amount), primary=True)

        @function_tool
        def approve_payment_request(request_id: str) -> Dict[str, Any]:
            """Approve an existing payment request. This tool is only available after PolicyAgent handoff."""
            return self._record(self.payment_system.approve_payment_request(request_id), primary=True)

        @function_tool
        def reject_payment_request(request_id: str) -> Dict[str, Any]:
            """Reject an existing payment request."""
            return self._record(self.payment_system.reject_payment_request(request_id), primary=True)

        @function_tool
        def show_transactions(user_id: str) -> Dict[str, Any]:
            """Show transactions involving one user."""
            return self._record(self.payment_system.get_transactions(user_id), primary=True)

        @function_tool
        def check_transfer_policy(sender_id: str, amount: float) -> Dict[str, Any]:
            """Check transfer policy before a transfer is executed."""
            result = self.policy_agent.check_transfer(sender_id, amount, self.payment_system)
            return self._record(result, primary=not self.policy_agent.is_approved(result))

        @function_tool
        def check_payment_request_policy(request_id: str) -> Dict[str, Any]:
            """Check transfer policy before a pending payment request is approved."""
            request = self.payment_system.payment_requests.get(request_id)
            if request is None:
                return self._record(AgentResult("PolicyAgent", f"Payment request {request_id} does not exist.", 0.9), primary=True)
            if request.status != "pending":
                return self._record(
                    AgentResult("PolicyAgent", f"Payment request {request_id} is already {request.status}.", 0.95),
                    primary=True,
                )
            result = self.policy_agent.check_transfer(request.payer_id, request.amount, self.payment_system)
            return self._record(result, primary=not self.policy_agent.is_approved(result))

        @function_tool
        def review_last_transaction() -> Dict[str, Any]:
            """Review the latest transaction for fraud risk."""
            transaction = self._transaction_from_result(self.primary_result) or self.memory.last_transaction
            result = self.fraud_agent.review(transaction, self.payment_system)
            if self.primary_result is not None:
                self.primary_result.metadata = {
                    **(self.primary_result.metadata or {}),
                    "fraud_check": result.output,
                }
            return self._record(result, primary=False)

        @function_tool
        def run_security_review() -> Dict[str, Any]:
            """Run a security review of the payment system state."""
            return self._record(self.security_agent.review(self.payment_system), primary=True)

        @function_tool
        def explain_last_action() -> Dict[str, Any]:
            """Explain the latest remembered business action."""
            return self._record(self.explanation_agent.explain(self.memory), primary=True)

        @function_tool
        def fallback_response(original_message: str) -> Dict[str, Any]:
            """Return a fallback response when the request is unsupported or ambiguous."""
            return self._record(self.fallback_agent.handle(original_message), primary=True)

        fraud_agent = OpenAIAgent(
            name="FraudDetectionAgent",
            model=self.model,
            instructions=(
                "You review only completed or recent transactions for fraud risk. "
                "Call review_last_transaction, then summarize the risk briefly."
            ),
            tools=[review_last_transaction],
        )
        security_agent = OpenAIAgent(
            name="SecurityAgent",
            model=self.model,
            instructions="You run system security reviews. Call run_security_review.",
            tools=[run_security_review],
        )
        explanation_agent = OpenAIAgent(
            name="ExplanationAgent",
            model=self.model,
            instructions="You explain the latest remembered action. Call explain_last_action.",
            tools=[explain_last_action],
        )
        fallback_agent = OpenAIAgent(
            name="FallbackAgent",
            model=self.model,
            instructions="You handle unsupported or ambiguous requests. Call fallback_response.",
            tools=[fallback_response],
        )
        payment_agent = OpenAIAgent(
            name="PaymentSystemAgent",
            model=self.model,
            instructions=(
                "You perform non-transfer payment-system actions using tools. "
                "Do not handle direct money transfers or payment-request approvals; those must go through PolicyAgent."
            ),
            tools=[
                create_user,
                check_balance,
                request_payment,
                reject_payment_request,
                show_transactions,
            ],
            handoffs=[fallback_agent],
        )
        transfer_execution_agent = OpenAIAgent(
            name="TransferExecutionAgent",
            model=self.model,
            instructions=(
                "You execute money-moving actions only after PolicyAgent has approved them. "
                "For a direct transfer, call transfer_money. "
                "For approval of a pending payment request, call approve_payment_request. "
                "After an approved transfer creates a transaction, handoff to FraudDetectionAgent for risk review."
            ),
            tools=[transfer_money, approve_payment_request],
            handoffs=[fraud_agent, fallback_agent],
        )
        policy_agent = OpenAIAgent(
            name="PolicyAgent",
            model=self.model,
            instructions=(
                "You evaluate transfer policy before money moves. "
                "For direct transfers, call check_transfer_policy. "
                "For payment request approvals, call check_payment_request_policy. "
                "If approved, handoff to TransferExecutionAgent. "
                "If rejected, stop after explaining the policy result. "
                "Never execute transfers yourself."
            ),
            tools=[check_transfer_policy, check_payment_request_policy],
            handoffs=[transfer_execution_agent, fallback_agent],
        )
        intake_agent = OpenAIAgent(
            name="OrchestratorAgent",
            model=self.model,
            instructions=(
                "You are the entrypoint for a Bit-like payment assistant. "
                "Infer the user's intent from natural language without keyword rules. "
                "Use handoffs instead of performing specialist work yourself: "
                "handoff direct transfer requests and payment-request approvals to PolicyAgent, "
                "handoff non-transfer payment operations to PaymentSystemAgent, "
                "fraud questions to FraudDetectionAgent, security questions to SecurityAgent, "
                "explanation questions to ExplanationAgent, and unclear requests to FallbackAgent. "
                "If a required field is missing, ask a concise follow-up question instead of guessing."
            ),
            handoffs=[policy_agent, payment_agent, fraud_agent, security_agent, explanation_agent, fallback_agent],
        )
        self.agents = {
            "intake": intake_agent,
            "policy": policy_agent,
            "payment": payment_agent,
            "transfer_execution": transfer_execution_agent,
            "fraud": fraud_agent,
            "security": security_agent,
            "explanation": explanation_agent,
            "fallback": fallback_agent,
        }


class OrchestratorAgent:
    def __init__(self):
        self.chatbot = OpenAIChatbot()
        self.payment_system = PaymentSystem()
        self.memory = ShortTermMemory()
        self.router = RouterAgent(self.chatbot)
        self.free_speech_parser = FreeSpeechParserAgent(self.chatbot)
        self.tool_selector = ToolSelector()
        self.fraud_agent = FraudDetectionAgent()
        self.policy_agent = PolicyAgent()
        self.payment_system.policy_agent = self.policy_agent
        self.security_agent = SecurityAgent()
        self.explanation_agent = ExplanationAgent(self.chatbot)
        self.critic_agent = CriticAgent()
        self.fallback_agent = FallbackAgent()
        self.openai_agent_runtime = OpenAIAgentRuntime(
            self.payment_system,
            self.memory,
            self.policy_agent,
            self.fraud_agent,
            self.security_agent,
            self.explanation_agent,
            self.fallback_agent,
            self.chatbot.model,
        )

    def run_free_speech(self, message: str) -> AgentResult:
        if self.openai_agent_runtime.enabled:
            try:
                result = self.openai_agent_runtime.run(message)
                return self._finalize_result(message, "openaiAgentsHandoff", "OpenAIAgentRuntime", result, {})
            except Exception as e:
                fallback_note = {"openai_agents_error": str(e), "inference": "local_fallback"}
        else:
            fallback_note = {
                "inference": "local_fallback",
                "openai_agents_disabled_reason": self.openai_agent_runtime.disabled_reason,
            }

        parsed_result = self.free_speech_parser.parse(message, self.memory)
        parsed = parsed_result.output
        if parsed["intent"] == "unknown":
            result = self.fallback_agent.handle(message)
            result.metadata = {**(result.metadata or {}), **fallback_note}
            return result
        if parsed["missing_fields"]:
            return AgentResult(
                "FreeSpeechParserAgent",
                {
                    "message": "I understood the request but need more details.",
                    "intent": parsed["intent"],
                    "missing_fields": parsed["missing_fields"],
                    "parsed_parameters": parsed["parameters"],
                },
                parsed_result.confidence,
                fallback_note,
            )
        result = self.run(parsed["intent"], **parsed["parameters"])
        result.metadata = {**(result.metadata or {}), **fallback_note}
        return result

    def run(self, message: str, **kwargs: Any) -> AgentResult:
        intent = message if message in ToolSelector.MAP else self.router.route(message).output
        selected_tool = self.tool_selector.select(intent).output
        internal_parameters = self._internal_parameters_for_intent(intent, kwargs)

        if selected_tool == "FallbackAgent":
            result = self.fallback_agent.handle(message)
        elif intent == "createUser":
            result = self.payment_system.create_user(kwargs["name"], kwargs["phone_number"], kwargs.get("initial_balance", 0))
        elif intent == "checkBalance":
            result = self.payment_system.get_balance(kwargs["user_id"])
        elif intent == "transferMoney":
            sender_id = kwargs["sender_id"]
            amount = float(kwargs["amount"])
            previous_balance = self.payment_system.wallets.get(sender_id, Wallet(sender_id, 0)).balance
            policy_result = self.policy_agent.check_transfer(sender_id, amount, self.payment_system)
            if not self.policy_agent.is_approved(policy_result):
                result = policy_result
            else:
                result = self.payment_system.transfer_money(sender_id, kwargs["receiver_id"], amount)
                tx = self._transaction_from_result(result)
                if tx and tx.status == "approved":
                    fraud_result = self.fraud_agent.review(tx, self.payment_system, previous_balance)
                    result.metadata = {**(result.metadata or {}), "fraud_check": fraud_result.output}
        elif intent == "requestPayment":
            result = self.payment_system.request_payment(kwargs["requester_id"], kwargs["payer_id"], float(kwargs["amount"]))
        elif intent == "approvePayment":
            request = self.payment_system.payment_requests.get(kwargs["request_id"])
            if request and request.status == "pending":
                policy_result = self.policy_agent.check_transfer(request.payer_id, request.amount, self.payment_system)
            else:
                policy_result = AgentResult("PolicyAgent", PolicyAgent.APPROVED_OUTPUT)
            if not self.policy_agent.is_approved(policy_result):
                result = policy_result
            else:
                result = self.payment_system.approve_payment_request(kwargs["request_id"])
                tx = self._transaction_from_result(result)
                if tx:
                    fraud_result = self.fraud_agent.review(tx, self.payment_system)
                    result.metadata = {**(result.metadata or {}), "fraud_check": fraud_result.output}
        elif intent == "rejectPayment":
            result = self.payment_system.reject_payment_request(kwargs["request_id"])
        elif intent == "showTransactions":
            result = self.payment_system.get_transactions(kwargs["user_id"])
        elif intent == "fraudCheck":
            result = self.fraud_agent.review(self.memory.last_transaction, self.payment_system)
        elif intent == "securityReview":
            result = self.security_agent.review(self.payment_system)
        elif intent == "explainLastAction":
            result = self.explanation_agent.explain(self.memory)
        else:
            result = self.fallback_agent.handle(message)

        return self._finalize_result(message, intent, selected_tool, result, internal_parameters, kwargs)

    def _finalize_result(
        self,
        message: str,
        intent: str,
        selected_tool: str,
        result: AgentResult,
        internal_parameters: Dict[str, Any],
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        kwargs = kwargs or {}
        critic = self.critic_agent.review(result)
        if critic.output["needs_fallback"]:
            result = self.fallback_agent.handle(message)

        result.metadata = {
            **(result.metadata or {}),
            "parameters": internal_parameters,
        }
        self.memory.update(
            intent,
            result,
            user_id=self._last_user_from_kwargs(kwargs),
            transaction=self._transaction_from_result(result),
            payment_request=self._payment_request_from_result(result),
        )
        self.payment_system._log(
            "orchestrator_run",
            {"message": message, "intent": intent, "tool": selected_tool, "agent": result.agent_name},
        )
        return result

    @staticmethod
    def _internal_parameters_for_intent(intent: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        fields_by_intent = {
            "createUser": ["name", "phone_number", "initial_balance"],
            "checkBalance": ["user_id"],
            "transferMoney": ["sender_id", "receiver_id", "amount"],
            "requestPayment": ["requester_id", "payer_id", "amount"],
            "approvePayment": ["request_id"],
            "rejectPayment": ["request_id"],
            "showTransactions": ["user_id"],
            "fraudCheck": [],
            "securityReview": [],
            "explainLastAction": [],
        }
        parameters = {
            field_name: kwargs[field_name]
            for field_name in fields_by_intent.get(intent, [])
            if field_name in kwargs
        }
        for numeric_field in ("amount", "initial_balance"):
            if numeric_field in parameters:
                parameters[numeric_field] = float(parameters[numeric_field])
        return parameters

    @staticmethod
    def _last_user_from_kwargs(kwargs: Dict[str, Any]) -> Optional[str]:
        for key in ("user_id", "sender_id", "receiver_id", "requester_id", "payer_id"):
            if key in kwargs:
                return kwargs[key]
        return None

    @staticmethod
    def _transaction_from_result(result: AgentResult) -> Optional[Transaction]:
        metadata = result.metadata or {}
        tx_data = metadata.get("transaction")
        if not tx_data and isinstance(result.output, dict):
            tx_data = result.output.get("transaction")
        if isinstance(tx_data, dict):
            return Transaction(**tx_data)
        return None

    @staticmethod
    def _payment_request_from_result(result: AgentResult) -> Optional[PaymentRequest]:
        metadata = result.metadata or {}
        req_data = metadata.get("payment_request")
        if not req_data and isinstance(result.output, dict):
            req_data = result.output.get("payment_request")
        if isinstance(req_data, dict):
            return PaymentRequest(**req_data)
        return None


def compact(result: AgentResult) -> Dict[str, Any]:
    return {
        "agent": result.agent_name,
        "confidence": result.confidence,
        "output": result.output,
        "metadata": result.metadata,
    }


def run_demo() -> List[AgentResult]:
    orch = OrchestratorAgent()
    results = [
        orch.run("create user", name="Alice", phone_number="050-1111111", initial_balance=8000),
        orch.run("create user", name="Bob", phone_number="050-2222222", initial_balance=500),
        orch.run("check balance", user_id="U001"),
        orch.run("check balance", user_id="U002"),
        orch.run("transfer money", sender_id="U001", receiver_id="U002", amount=700),
        orch.run("transfer money", sender_id="U001", receiver_id="U002", amount=-50),
        orch.run("transfer money", sender_id="U002", receiver_id="U001", amount=9000),
        orch.run("transfer money", sender_id="U001", receiver_id="U999", amount=10),
        orch.run("transfer money", sender_id="U001", receiver_id="U001", amount=10),
        orch.run("request payment", requester_id="U001", payer_id="U002", amount=100),
    ]
    request_id = results[-1].output["request_id"]
    results.append(orch.run("approve payment", request_id=request_id))
    results.append(orch.run("approve payment", request_id=request_id))
    results.append(orch.run("transfer money", sender_id="U001", receiver_id="U002", amount=6000))
    results.append(orch.run("fraud check"))
    results.append(orch.run("explain last transaction"))
    results.append(orch.run("security review"))
    results.append(orch.payment_system.save_to_json("payment_state.json"))
    return results


if __name__ == "__main__":
    for index, item in enumerate(run_demo(), 1):
        print(f"\n--- Test {index} ---")
        print(json.dumps(compact(item), indent=2, ensure_ascii=False))
