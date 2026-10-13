"""PromiseLedger — AI Claim Extractor (Owner: Rishav).

Converts natural-language agent statements into structured, atomic, checkable
action claims using an open-weight language model (Llama-3.3-70B-Instruct /
Qwen2.5-72B-Instruct) via an OpenAI-compatible inference endpoint or local
Ollama runtime, with strict schema validation and a clearly labeled
deterministic fallback.

IMPORTANT:
- Uses prompt-based structured extraction only (no model training or fine-tuning).
- Never decides whether claims are true, never inspects tool_events for verification,
  and never generates verdicts (VERIFIED / CONTRADICTED / UNVERIFIED).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class ClaimValidationError(ValueError):
    """Raised when model output or claim payload violates the shared contract."""


REQUIRED_CLAIM_FIELDS = ("claim_id", "action", "target", "source_text", "confidence")
FORBIDDEN_TOP_LEVEL_KEYS = ("results", "verdicts", "tool_events", "summary")
FORBIDDEN_CLAIM_KEYS = ("status", "verdict", "evidence", "explanation")

ACTION_ALIASES = {
    "read": "read_file",
    "open_file": "read_file",
    "inspect_file": "read_file",
    "check_file": "read_file",
    "read_files": "read_file",
    "inspect_files": "read_file",
    "check_files": "read_file",
    "create_report": "generate_report",
    "write_report": "generate_report",
    "build_report": "generate_report",
    "write_file": "write_file",
    "delete_file": "delete_file",
    "run_tests": "run_tests",
    "execute_command": "execute_command",
}

WORD_TO_NUMBER = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_OLLAMA_API_URL = "http://localhost:11434/v1/chat/completions"

SYSTEM_PROMPT = """You are the AI Claim Extractor for PromiseLedger, an AI agent accountability system.
Your sole responsibility is to extract explicit, checkable action claims made by an AI agent in its natural-language statement and return them as atomic structured JSON claims.

STRICT RULES:
1. Return ONLY a valid JSON object matching this exact schema:
{
  "claims": [
    {
      "claim_id": "c1",
      "action": "read_file",
      "target": "file1.txt",
      "source_text": "exact substring from agent statement",
      "confidence": 0.95
    }
  ]
}
2. NEVER decide whether a claim is true or false.
3. NEVER generate tool_events, verdicts (VERIFIED, CONTRADICTED, UNVERIFIED), or evidence fields.
4. Use canonical action names where applicable:
   - "read_file" (for reading, inspecting, checking, or opening files)
   - "generate_report" (for generating, writing, or creating a report)
   - "write_file" (for writing, saving, or modifying a non-report file)
   - "delete_file" (for deleting or removing a file)
   - "run_tests" (for running or executing tests)
   - "execute_command" (for running a shell/CLI command)
5. GROUP CLAIMS RULE:
   - If the agent claims a group action (e.g., "I read all five files" or "I inspected all five files") AND `expected_actions` provides the specific file targets, expand the group claim into separate atomic claims (one per target file, in order) using the group phrase as `source_text`.
   - If the agent claims a group action (e.g., "I read all five files") and individual filenames are NOT present in either the statement or `expected_actions`, DO NOT invent filenames like "file1.txt". Instead, emit a single explicit aggregate claim with target set to "all_5_files" (or the corresponding count/group descriptor).
6. If the statement is empty or contains no checkable action claims, return {"claims": []}.
"""


def build_user_prompt(
    agent_statement: str,
    expected_actions: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the user prompt for the open-weight language model."""
    context_block = "None provided (do NOT invent individual filenames if not mentioned in statement)."
    if expected_actions:
        sanitized_actions = [
            {"type": a.get("type"), "target": a.get("target")}
            for a in expected_actions
            if isinstance(a, dict) and a.get("target")
        ]
        context_block = json.dumps(sanitized_actions, indent=2)

    return (
        f"AGENT STATEMENT:\n{json.dumps(agent_statement)}\n\n"
        f"KNOWN EXPECTED ACTION TARGETS (for resolving group references like 'all five files'):\n"
        f"{context_block}\n\n"
        "Extract all checkable claims into the required JSON format."
    )


def normalize_action(action: str) -> str:
    """Normalize common verb phrases into canonical action identifiers."""
    cleaned = action.strip().lower().replace(" ", "_").replace("-", "_")
    return ACTION_ALIASES.get(cleaned, cleaned)


def parse_json_response(raw_text: str) -> Dict[str, Any]:
    """Parse JSON from raw model text, stripping optional markdown code fences."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ClaimValidationError("Model response is empty or not a string.")

    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ClaimValidationError(f"Malformed JSON in model response: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ClaimValidationError("Top-level JSON value must be an object/dict.")

    return parsed


def _check_no_invented_filenames(
    target: str,
    agent_statement: str,
    expected_actions: Optional[List[Dict[str, Any]]],
) -> None:
    """Ensure individual numbered filenames (e.g. file1.txt) are not invented
    when neither the statement nor expected_actions contains them.
    """
    allowed_targets = set()
    if expected_actions:
        for item in expected_actions:
            if isinstance(item, dict) and isinstance(item.get("target"), str):
                allowed_targets.add(item["target"].strip().lower())

    target_lower = target.strip().lower()
    statement_lower = (agent_statement or "").lower()

    if re.match(r"^file\d+\.[a-z0-9]+$", target_lower):
        if target_lower not in statement_lower and target_lower not in allowed_targets:
            raise ClaimValidationError(
                f"Model invented filename '{target}' not present in statement or expected_actions."
            )


def validate_claims_payload(
    payload: Any,
    agent_statement: Optional[str] = None,
    expected_actions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Validate and normalize a parsed claims dict against the shared contract."""
    if not isinstance(payload, dict):
        raise ClaimValidationError("Claims payload must be a dictionary.")

    for forbidden in FORBIDDEN_TOP_LEVEL_KEYS:
        if forbidden in payload:
            raise ClaimValidationError(
                f"Extractor must not generate '{forbidden}'; verification belongs to Rajdeep's engine."
            )

    if "claims" not in payload:
        raise ClaimValidationError("Missing required top-level key 'claims'.")

    raw_claims = payload["claims"]
    if not isinstance(raw_claims, list):
        raise ClaimValidationError("Field 'claims' must be a list.")

    validated_claims: List[Dict[str, Any]] = []
    seen_ids = set()

    for idx, item in enumerate(raw_claims, start=1):
        if not isinstance(item, dict):
            raise ClaimValidationError(f"Claim at index {idx - 1} must be an object/dict.")

        for forbidden_key in FORBIDDEN_CLAIM_KEYS:
            if forbidden_key in item:
                raise ClaimValidationError(
                    f"Claim at index {idx - 1} contains forbidden verification field '{forbidden_key}'."
                )

        for req_field in REQUIRED_CLAIM_FIELDS:
            if req_field not in item:
                raise ClaimValidationError(
                    f"Claim at index {idx - 1} is missing required field '{req_field}'."
                )

        claim_id = item["claim_id"]
        action = item["action"]
        target = item["target"]
        source_text = item["source_text"]
        confidence = item["confidence"]

        if not isinstance(claim_id, str) or not claim_id.strip():
            raise ClaimValidationError(f"Claim at index {idx - 1} has invalid 'claim_id'.")
        if not isinstance(action, str) or not action.strip():
            raise ClaimValidationError(f"Claim at index {idx - 1} has invalid 'action'.")
        if not isinstance(target, str) or not target.strip():
            raise ClaimValidationError(f"Claim at index {idx - 1} has invalid 'target'.")
        if not isinstance(source_text, str) or not source_text.strip():
            raise ClaimValidationError(f"Claim at index {idx - 1} has invalid 'source_text'.")

        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ClaimValidationError(
                f"Claim at index {idx - 1} 'confidence' must be a float between 0.0 and 1.0."
            )
        conf_float = float(confidence)
        if conf_float < 0.0 or conf_float > 1.0:
            raise ClaimValidationError(
                f"Claim at index {idx - 1} 'confidence' ({conf_float}) is outside [0.0, 1.0]."
            )

        if agent_statement is not None:
            _check_no_invented_filenames(target, agent_statement, expected_actions)

        clean_id = claim_id.strip()
        if clean_id in seen_ids:
            clean_id = f"c{idx}"
        seen_ids.add(clean_id)

        validated_claims.append(
            {
                "claim_id": clean_id,
                "action": normalize_action(action),
                "target": target.strip(),
                "source_text": source_text.strip(),
                "confidence": round(conf_float, 4),
            }
        )

    return {"claims": validated_claims}


def _resolve_input(
    statement_or_payload: Union[str, Dict[str, Any]],
    expected_actions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, Optional[List[Dict[str, Any]]], Optional[str]]:
    """Extract (agent_statement, expected_actions, run_id) from string or dict input."""
    if isinstance(statement_or_payload, str):
        return statement_or_payload, expected_actions, None

    if isinstance(statement_or_payload, dict):
        stmt = statement_or_payload.get("agent_statement", "")
        if not isinstance(stmt, str):
            raise ClaimValidationError("Field 'agent_statement' must be a string.")
        exp = (
            expected_actions
            if expected_actions is not None
            else statement_or_payload.get("expected_actions")
        )
        run_id = statement_or_payload.get("run_id")
        return stmt, exp, run_id

    raise ClaimValidationError(
        "Input must be an agent_statement string or a payload dictionary."
    )


def deterministic_extract_claims(
    agent_statement: str,
    expected_actions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Deterministic rule-based claim extractor used as fallback when the live
    open-weight model endpoint is unavailable or returns invalid output.
    """
    if not isinstance(agent_statement, str) or not agent_statement.strip():
        return {"claims": []}

    statement = agent_statement.strip()
    claims: List[Dict[str, Any]] = []
    claim_counter = 1

    def add_claim(action: str, target: str, source_text: str, confidence: float) -> None:
        nonlocal claim_counter
        for existing in claims:
            if existing["action"] == action and existing["target"].lower() == target.lower():
                return
        claims.append(
            {
                "claim_id": f"c{claim_counter}",
                "action": action,
                "target": target,
                "source_text": source_text.strip(),
                "confidence": round(confidence, 2),
            }
        )
        claim_counter += 1

    expected_read_targets: List[str] = []
    expected_report_targets: List[str] = []
    if expected_actions:
        for item in expected_actions:
            if not isinstance(item, dict):
                continue
            a_type = str(item.get("type", "")).strip().lower()
            a_target = str(item.get("target", "")).strip()
            if not a_target:
                continue
            if a_type in ("read_file", "read", "inspect_file", "check_file"):
                expected_read_targets.append(a_target)
            elif a_type in ("generate_report", "create_report", "write_report"):
                expected_report_targets.append(a_target)

    action_verbs_pattern = (
        r"(?:read|inspected|checked|opened|examined|reviewed|processed|loaded|parsed|"
        r"generated|created|wrote|built|compiled|produced|updated|saved|modified|"
        r"deleted|removed|ran|executed|passed)"
    )
    clauses = [
        c.strip()
        for c in re.split(
            rf"(?:\.\s+|\s*;\s*|(?:\s*,\s*|\s+)(?:and\s+then|and|then|finally|next)\s+(?=(?:i\s+)?{action_verbs_pattern}\b)|\s*,\s+(?=(?:i\s+)?(?:then\s+)?{action_verbs_pattern}\b))",
            statement,
            flags=re.IGNORECASE,
        )
        if c and c.strip()
    ]

    for clause in clauses:
        clause_clean = clause.rstrip(".")
        clause_lower = clause_clean.lower()

        # 1. Group file read claims, e.g., "I read all five files" / "I checked all 5 files"
        group_match = re.search(
            r"\b(?:read|inspected|checked|opened|examined|reviewed|processed)\s+"
            r"(?:all\s+)?(?:the\s+)?"
            r"(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)?\s*"
            r"files\b",
            clause_lower,
        )
        if group_match:
            raw_count = group_match.group("count")
            count: Optional[int] = None
            if raw_count:
                count = (
                    int(raw_count)
                    if raw_count.isdigit()
                    else WORD_TO_NUMBER.get(raw_count)
                )

            explicit_files_in_clause = re.findall(
                r"\b[a-zA-Z0-9_\-]+(?:\.[a-zA-Z0-9]+)+\b", clause_clean
            )

            if explicit_files_in_clause:
                for fname in explicit_files_in_clause:
                    add_claim("read_file", fname, clause_clean, 0.92)
            elif expected_read_targets:
                targets_to_use = (
                    expected_read_targets[:count]
                    if count and count <= len(expected_read_targets)
                    else expected_read_targets
                )
                for target in targets_to_use:
                    add_claim("read_file", target, clause_clean, 0.90)
            else:
                aggregate_target = f"all_{count}_files" if count else "all_files"
                add_claim("read_file", aggregate_target, clause_clean, 0.85)
            continue

        # 2. Report generation claims, e.g., "generated report.md" or "generated the report"
        report_match = re.search(
            r"\b(?:generated|created|wrote|built|compiled|produced)\s+"
            r"(?:the\s+|a\s+|an\s+|final\s+)?"
            r"(?P<report_file>[a-zA-Z0-9_\-]+\.(?:md|txt|pdf|html|json|csv)|report)\b",
            clause_clean,
            flags=re.IGNORECASE,
        )
        if report_match or (
            "report" in clause_lower
            and any(
                v in clause_lower
                for v in ("generated", "created", "wrote", "built", "produced")
            )
        ):
            explicit_report = re.search(
                r"\b([a-zA-Z0-9_\-]+\.(?:md|txt|pdf|html|json|csv))\b",
                clause_clean,
            )
            if explicit_report:
                report_target = explicit_report.group(1)
            elif expected_report_targets:
                report_target = expected_report_targets[0]
            else:
                report_target = "report"
            add_claim("generate_report", report_target, clause_clean, 0.93)
            continue

        # 3. Explicit file read claims, e.g., "I read file1.txt and file2.txt"
        if re.search(
            r"\b(?:read|inspected|checked|opened|examined|loaded|parsed)\b",
            clause_lower,
        ):
            filenames = re.findall(
                r"\b([a-zA-Z0-9_\-/]+\.[a-zA-Z0-9]{1,8})\b", clause_clean
            )
            if filenames:
                for fname in filenames:
                    add_claim("read_file", fname, clause_clean, 0.94)
                continue

        # 4. Explicit file write/update claims
        if re.search(r"\b(?:wrote|updated|saved|modified|created)\b", clause_lower):
            filenames = re.findall(
                r"\b([a-zA-Z0-9_\-/]+\.[a-zA-Z0-9]{1,8})\b", clause_clean
            )
            if filenames:
                for fname in filenames:
                    action_type = (
                        "generate_report"
                        if "report" in fname.lower()
                        else "write_file"
                    )
                    add_claim(action_type, fname, clause_clean, 0.92)
                continue

        # 5. File deletion claims
        if re.search(r"\b(?:deleted|removed)\b", clause_lower):
            filenames = re.findall(
                r"\b([a-zA-Z0-9_\-/]+\.[a-zA-Z0-9]{1,8})\b", clause_clean
            )
            if filenames:
                for fname in filenames:
                    add_claim("delete_file", fname, clause_clean, 0.92)
                continue

        # 6. Running tests
        if re.search(
            r"\b(?:ran|executed|passed)\b.*\b(?:tests|test suite|pytest|unittest)\b",
            clause_lower,
        ):
            add_claim("run_tests", "test_suite", clause_clean, 0.90)
            continue

    return validate_claims_payload(
        {"claims": claims},
        agent_statement=agent_statement,
        expected_actions=expected_actions,
    )


class ClaimExtractor:
    """Open-weight LLM Claim Extractor with validation and deterministic fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_sec: float = 8.0,
        http_client: Optional[
            Callable[[str, Dict[str, str], Dict[str, Any], float], str]
        ] = None,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("OPEN_WEIGHT_API_KEY")
            or os.environ.get("GROQ_API_KEY")
            or os.environ.get("HF_TOKEN")
        )
        self.use_local_ollama = (
            os.environ.get("USE_LOCAL_OLLAMA", "").lower() in ("1", "true", "yes")
        )

        if api_url:
            self.api_url = api_url
        elif self.use_local_ollama:
            self.api_url = os.environ.get("OPEN_WEIGHT_API_URL", DEFAULT_OLLAMA_API_URL)
        elif os.environ.get("GROQ_API_KEY") and not os.environ.get("OPEN_WEIGHT_API_URL"):
            self.api_url = DEFAULT_GROQ_API_URL
        else:
            self.api_url = os.environ.get("OPEN_WEIGHT_API_URL", DEFAULT_HF_API_URL)

        if model_name:
            self.model_name = model_name
        elif os.environ.get("OPEN_WEIGHT_MODEL"):
            self.model_name = os.environ["OPEN_WEIGHT_MODEL"]
        elif os.environ.get("GROQ_API_KEY"):
            self.model_name = DEFAULT_GROQ_MODEL
        else:
            self.model_name = DEFAULT_MODEL

        self.timeout_sec = float(
            os.environ.get("EXTRACTOR_TIMEOUT_SEC", str(timeout_sec))
        )
        self._http_client = http_client or self._default_http_post

    @staticmethod
    def _default_http_post(
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        timeout: float,
    ) -> str:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_bytes = resp.read()
            return raw_bytes.decode("utf-8")

    def _call_open_weight_model(
        self,
        agent_statement: str,
        expected_actions: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if (
            not self.api_key
            and not self.use_local_ollama
            and self._http_client == self._default_http_post
        ):
            raise RuntimeError(
                "Missing OPEN_WEIGHT_API_KEY (or GROQ_API_KEY / HF_TOKEN) and USE_LOCAL_OLLAMA is not enabled."
            )

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "PromiseLedger-ClaimExtractor/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(agent_statement, expected_actions),
                },
            ],
            "response_format": {"type": "json_object"},
        }

        raw_http_body = self._http_client(
            self.api_url, headers, payload, self.timeout_sec
        )
        try:
            envelope = json.loads(raw_http_body)
        except json.JSONDecodeError as exc:
            raise ClaimValidationError(
                f"Endpoint returned non-JSON HTTP payload: {exc}"
            ) from exc

        try:
            content = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ClaimValidationError(
                f"Unexpected OpenAI-compatible response structure: {exc}"
            ) from exc

        if not isinstance(content, str):
            raise ClaimValidationError("Model message content is not a string.")
        return content

    def extract(
        self,
        statement_or_payload: Union[str, Dict[str, Any]],
        expected_actions: Optional[List[Dict[str, Any]]] = None,
        force_fallback: bool = False,
    ) -> Dict[str, Any]:
        """Extract structured claims from an agent statement or run payload."""
        agent_statement, resolved_expected, _ = _resolve_input(
            statement_or_payload, expected_actions
        )

        if not agent_statement.strip():
            return {
                "claims": [],
                "extraction_meta": {
                    "source": "empty_input",
                    "is_fallback": False,
                    "model": None,
                    "adaptation_method": "none",
                    "trained_or_finetuned": False,
                    "fallback_reason": None,
                },
            }

        env_force_fallback = os.environ.get(
            "EXTRACTOR_FORCE_FALLBACK", ""
        ).lower() in ("1", "true", "yes")

        if force_fallback or env_force_fallback:
            fallback_res = deterministic_extract_claims(
                agent_statement, resolved_expected
            )
            return {
                "claims": fallback_res["claims"],
                "extraction_meta": {
                    "source": "deterministic_fallback",
                    "is_fallback": True,
                    "model": None,
                    "adaptation_method": "deterministic_rules",
                    "trained_or_finetuned": False,
                    "fallback_reason": "Fallback explicitly requested via force_fallback.",
                },
            }

        try:
            raw_content = self._call_open_weight_model(
                agent_statement, resolved_expected
            )
            parsed = parse_json_response(raw_content)
            validated = validate_claims_payload(
                parsed,
                agent_statement=agent_statement,
                expected_actions=resolved_expected,
            )
            return {
                "claims": validated["claims"],
                "extraction_meta": {
                    "source": "open_weight_model",
                    "is_fallback": False,
                    "model": self.model_name,
                    "adaptation_method": "prompt_based_zero_shot",
                    "trained_or_finetuned": False,
                    "fallback_reason": None,
                },
            }
        except (
            ClaimValidationError,
            urllib.error.URLError,
            TimeoutError,
            RuntimeError,
            OSError,
        ) as exc:
            fallback_res = deterministic_extract_claims(
                agent_statement, resolved_expected
            )
            return {
                "claims": fallback_res["claims"],
                "extraction_meta": {
                    "source": "deterministic_fallback",
                    "is_fallback": True,
                    "model": None,
                    "adaptation_method": "deterministic_rules",
                    "trained_or_finetuned": False,
                    "fallback_reason": f"{type(exc).__name__}: {exc}",
                },
            }


def extract_claims(
    statement_or_payload: Union[str, Dict[str, Any]],
    expected_actions: Optional[List[Dict[str, Any]]] = None,
    force_fallback: bool = False,
    extractor: Optional[ClaimExtractor] = None,
) -> Dict[str, Any]:
    """Primary entry point for Kalpesh's backend and Rajdeep's verification pipeline."""
    instance = extractor or ClaimExtractor()
    return instance.extract(
        statement_or_payload,
        expected_actions=expected_actions,
        force_fallback=force_fallback,
    )


def extract_claims_strict(
    statement_or_payload: Union[str, Dict[str, Any]],
    expected_actions: Optional[List[Dict[str, Any]]] = None,
    force_fallback: bool = False,
    extractor: Optional[ClaimExtractor] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return strictly `{"claims": [...]}` with no extra metadata keys."""
    res = extract_claims(
        statement_or_payload,
        expected_actions=expected_actions,
        force_fallback=force_fallback,
        extractor=extractor,
    )
    return {"claims": res["claims"]}


DEMO_PAYLOAD: Dict[str, Any] = {
    "run_id": "demo-001",
    "agent_statement": "I inspected all five files and generated the report.",
    "expected_actions": [
        {"action_id": "a1", "type": "read_file", "target": "file1.txt"},
        {"action_id": "a2", "type": "read_file", "target": "file2.txt"},
        {"action_id": "a3", "type": "read_file", "target": "file3.txt"},
        {"action_id": "a4", "type": "read_file", "target": "file4.txt"},
        {"action_id": "a5", "type": "read_file", "target": "file5.txt"},
        {"action_id": "a6", "type": "generate_report", "target": "report.md"},
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PromiseLedger AI Claim Extractor CLI (Owner: Rishav)"
    )
    parser.add_argument(
        "--statement",
        type=str,
        help="Agent natural-language statement to extract claims from.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to JSON file containing run payload.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run extraction on the shared demo-001 scenario.",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Force deterministic fallback mode.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Output only the strict {'claims': [...]} key without extraction_meta.",
    )
    args = parser.parse_args()

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            payload: Union[str, Dict[str, Any]] = json.load(f)
    elif args.statement is not None:
        payload = args.statement
    else:
        payload = DEMO_PAYLOAD

    if args.strict:
        result = extract_claims_strict(payload, force_fallback=args.fallback)
    else:
        result = extract_claims(payload, force_fallback=args.fallback)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
