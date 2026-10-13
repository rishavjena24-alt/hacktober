# PromiseLedger

> **"Don't trust the promise. Verify the evidence."**

PromiseLedger is an AI agent accountability system that detects gaps between what an AI agent claims it has done and what its recorded tool events demonstrate.

---

## Project Structure

```text
PromiseLedger/
│
├── backend/
│   └── claim_extractor.py      # AI Claim Extraction component (Owner: Rishav)
│
├── tests/
│   └── test_claim_extractor.py # Independent unit & acceptance test suite
│
├── examples/
│   ├── sample_input.json       # Shared input contract (demo-001)
│   └── sample_claims.json      # Extracted structured claims output
│
├── .env.example                # Open-weight model environment configuration
├── requirements.txt            # Project dependencies
└── README.md                   # Documentation
```

---

## AI Claim Extraction Component (Owner: Rishav)

`backend/claim_extractor.py` converts natural-language agent statements (e.g., *"I inspected all five files and generated the report."*) into atomic, structured JSON claims conforming to the shared PromiseLedger contract:

```json
{
  "claims": [
    {
      "claim_id": "c1",
      "action": "read_file",
      "target": "file1.txt",
      "source_text": "I inspected all five files",
      "confidence": 0.9
    }
  ]
}
```

### Key Guarantees
1. **Strict Separation of Concerns:** The claim extractor **never** decides whether a claim is true, never inspects `tool_events` for verification, and never emits verdicts (`VERIFIED`, `CONTRADICTED`, `UNVERIFIED`). Verdict calculation belongs exclusively to Rajdeep's deterministic verification engine.
2. **Group Claim & Unspecified Target Policy:**
   * When individual target filenames are available (in the statement or `expected_actions`), group statements like *"I inspected all five files"* expand into 5 atomic `read_file` claims (`c1`–`c5`).
   * When individual filenames are **unavailable** (e.g., *"I checked all five files."* with no `expected_actions`), the extractor preserves the aggregate claim explicitly (`"target": "all_5_files"`) and **never invents filenames**.
3. **Schema Validation & Clearly Labeled Fallback:** Every model response is validated for required fields, types, and `0.0 <= confidence <= 1.0`. If the model endpoint is unavailable, times out, or returns malformed JSON, the extractor automatically switches to `deterministic_extract_claims()` and explicitly labels the result with `"source": "deterministic_fallback"` and `"is_fallback": true`.

---

## Open-Weight Model & License Attribution

| Model Identifier | Creator | Open-Weight License | Access Method |
| :--- | :--- | :--- | :--- |
| **`meta-llama/Llama-3.3-70B-Instruct`** *(Default)* / `llama-3.3-70b-versatile` | Meta AI | [Llama 3.3 Community License Agreement](https://www.llama.com/llama3_3/license/) | Hugging Face Router (`HF_TOKEN`) or Groq (`GROQ_API_KEY`) |
| **`Qwen/Qwen2.5-72B-Instruct`** / `qwen2.5:7b` | Alibaba Cloud (Qwen Team) | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Hugging Face Router (`HF_TOKEN`) or Local Ollama (`USE_LOCAL_OLLAMA=true`) |

* **Adaptation Disclosure:** PromiseLedger uses **prompt-based zero-shot structured extraction** (`SYSTEM_PROMPT`) combined with deterministic post-validation. **No model training or fine-tuning was performed** (`"trained_or_finetuned": false`).

---

## Quick Start & Running Tests

```powershell
# 1. Run all 15 unit & acceptance tests
wsl python3 -m unittest discover -s tests -p "test_*.py" -v

# 2. Run the extractor on examples/sample_input.json (strict shared contract)
wsl python3 backend/claim_extractor.py --input-file examples/sample_input.json --strict

# 3. Extract from a custom statement
wsl python3 backend/claim_extractor.py --statement "I read file1.txt and file2.txt, then generated report.md."
```
