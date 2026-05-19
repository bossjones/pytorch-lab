## Python Environment

This project uses `uv` with a `.venv` at the project root.

**Always use `uv run` to execute Python code, NOT bare `python`:**
```bash
# ✅ Correct
uv run pytest
uv run python src/main.py

# ❌ Wrong — venv won't be activated in agent terminal
python src/main.py
source .venv/bin/activate && python src/main.py
```

**Interpreter path:** `.venv/bin/python`
**Package manager:** `uv` — use `uv add <pkg>` not `pip install`
