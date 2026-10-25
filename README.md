# study-planner: one agent, two harnesses

A study-planning agent for a university student in Israel: it tracks assignments and exams, remembers study
preferences across sessions, and builds weekly plans that skip Israeli holidays (fetched live from the Hebcal API).

| | |
|---|---|
| `shared/` | domain code used by both versions: tools (`toolkit.py`), planner, memory store, Hebcal client, checks, scenario runner |
| `API/` | raw model API, two providers: Anthropic Messages API (default) and Gemini (`--provider gemini`). The whole harness is hand-written in `API/harness/`, one file per responsibility; Gemini has its own `gemini_loop/tools/caching/tokens.py`, and history, memory, context, compaction are shared |
| `SDK/` | Claude Agent SDK. Claude Code runs the harness; lines tagged `# replaces: <file>` stand in for each harness file |
| `tests/` | offline tests (fake client, fake Hebcal, no key, no network) |
| `scenario.json` | 7 turns over 2 sessions + 8 pass/fail checks |
| `results/` | `api.json` (Anthropic), `api_gemini.json` (Gemini), `sdk.json` |
| `REPORT.md` | comparison |

## Setup

The virtual environment lives one level up, in `t1\.venv` (next to this folder, not inside it).
All commands below are run from this folder:

```
python -m venv ..\.venv
..\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```
..\.venv\Scripts\python -m pytest tests                                          # offline, no key
..\.venv\Scripts\python API\main.py --scenario scenario.json --out results\api.json   # needs ANTHROPIC_API_KEY in the shell env
..\.venv\Scripts\python SDK\main.py --scenario scenario.json --out results\sdk.json   # uses your Claude Code login
..\.venv\Scripts\python compare.py                                               # tables for REPORT.md
```

(After `..\.venv\Scripts\activate`, plain `python` works too.)

### API on Gemini (free tier), in cmd

The key is read only from `GEMINI_API_KEY` (not `GOOGLE_API_KEY`), and never written to code, logs or results.
Set it for this terminal only, run, then clear it:

```
set GEMINI_API_KEY=...
..\.venv\Scripts\python API\main.py --provider gemini --scenario scenario.json --out results\api_gemini.json
set GEMINI_API_KEY=
..\.venv\Scripts\python compare.py api_gemini sdk
```

Model: `gemini-3.8-flash`, `thinking_level=MEDIUM`. Free-tier rate limits are handled with retry and backoff on
HTTP 429/503 (the server's suggested delay when it sends one); the waits show up in the `seconds` column.

For the API key, set it only in the current terminal, never in a file:
PowerShell `$env:ANTHROPIC_API_KEY = "sk-ant-..."`, bash `export ANTHROPIC_API_KEY=...`.
`SDK/main.py` removes `ANTHROPIC_API_KEY` from its own environment, so the SDK version is never billed to the key.

Every scenario run starts from a clean state: it deletes `runs/api`, `runs/api_gemini` or `runs/sdk` (memory,
conversation history, tasks, plans, Hebcal cache, CLAUDE.md). The SDK run also deletes Claude Code's own transcripts
for `runs/sdk` under `~/.claude/projects/`. The providers' server-side prompt caches cannot be cleared; they expire
on their own.
Interactive use: `..\.venv\Scripts\python API\main.py --chat --session mine` (resumes that session) or `..\.venv\Scripts\python SDK\main.py --chat [--resume <session_id>]`.
