# MeteoLibre Model CRUSH.md

This file provides guidance for agentic coding agents working in the `meteolibre-model` repository.

## Commands

- **Run all tests:** `python -m unittest discover tests`
- **Run a single test:** `python -m unittest tests/models/test_dc_3dunet.py`
- **Lint the code:** `ruff check .`
- **Format the code:** `ruff format .`

## Code Style

- **Formatting:** Use `ruff` for code formatting.
- **Imports:** Sort imports using `isort` conventions (which `ruff` can enforce). Group imports into standard library, third-party, and first-party modules.
- **Typing:** Use type hints for all function signatures.
- **Naming Conventions:**
    - Use `PascalCase` for class names.
    - Use `snake_case` for functions, methods, and variables.
- **Docstrings:** Write docstrings for all public modules, classes, and functions.
- **Error Handling:** Use specific exception types instead of generic `Exception`.
- **Dependencies:** Manage dependencies using `pyproject.toml`.
