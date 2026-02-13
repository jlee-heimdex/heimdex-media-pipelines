# Heimdex Media Pipelines — Makefile
#
# Usage:
#   make test          Run all tests (excluding golden)
#   make test-golden   Run OCR golden dataset tests (requires media + paddleocr)
#   make ocr-audit     Audit OCR dependencies for known vulnerabilities

.PHONY: test test-golden ocr-audit

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test:
	uv run --no-sync pytest -q

test-golden:
	uv run --no-sync pytest -m ocr_golden -v

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

## Audit OCR optional dependencies for known vulnerabilities.
## Requires: pip install pip-audit
## Installs OCR extras into a temp venv, then runs pip-audit against them.
ocr-audit:
	@echo "=== Auditing OCR dependencies ==="
	@echo "Checking pip-audit is available..."
	@command -v pip-audit >/dev/null 2>&1 || { echo "pip-audit not found. Install with: pip install pip-audit"; exit 1; }
	pip-audit --requirement <(python3 -c "\
		try:\
			import tomllib;\
		except ImportError:\
			import tomli as tomllib;\
		with open('pyproject.toml', 'rb') as f:\
			cfg = tomllib.load(f);\
		deps = cfg['project']['optional-dependencies']['ocr'];\
		print('\n'.join(deps))")
	@echo "=== OCR audit complete ==="
