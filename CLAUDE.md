# CLAUDE.md

Guidelines for working on this repository with Claude Code.

## Hard Constraints

- **Do not install new dependencies** without asking the user first. This includes pip packages, system packages, or any additions to `requirements*.txt` files.
- **Do not create virtual environments**. The project runs entirely inside Docker containers.

## Testing

- **Always write tests for new features.** Every new feature, endpoint, model wrapper, or evaluation component must have corresponding tests in `tests/unit/` and/or `tests/integration/` as appropriate.
- **Always run the test suite using `docker compose`**, never with a local `pytest` invocation:

  ```bash
  docker compose run --rm test              # All tests
  docker compose run --rm test-unit         # Unit tests only
  docker compose run --rm test-integration  # Integration tests only
  docker compose run --rm test-e2e          # End-to-end tests only
  docker compose run --rm test-coverage     # With HTML + terminal coverage report
  ```

- **Iterate until all tests pass.** If tests fail, investigate and fix the root cause — do not skip, mark as expected failures, or comment out tests to make the suite green.

## Project Structure

- `src/` — Application source code (backend API, frontend dashboard, models, evaluation, data, training, utils)
- `tests/` — Test suite (`unit/`, `integration/`, `e2e/`, `fixtures/`)
- `scripts/` — Standalone evaluation and training scripts
- `docs/` — Detailed documentation (testing guide, model selection, GPU setup, etc.)
- `results/` — Output storage for metrics and error analysis

## Development Workflow

1. Make changes in `src/` (or `scripts/`, `tests/` as appropriate).
2. Write or update tests in `tests/` alongside the feature code.
3. Run the test suite with `docker compose run --rm test` and fix any failures before committing.
4. Commit with a clear, descriptive message on the designated feature branch.

## Key Documentation

- `docs/TESTING.md` — Full testing framework guide
- `docs/PROJECT-STRUCTURE.md` — Detailed codebase layout
- `docs/KNOWN_ISSUES.md` — Troubleshooting reference
- `README.md` — Quick start and API documentation
