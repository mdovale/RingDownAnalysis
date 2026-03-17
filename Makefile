# RingDownAnalysis - development and CI targets
# Run `make ci` to run all checks matching GitHub Actions CI.

.PHONY: ci format lint typecheck test

ci: format lint typecheck test
	@echo "All CI checks passed."

format:
	ruff format ringdownanalysis tests examples benchmarks

lint:
	ruff check ringdownanalysis tests examples benchmarks

typecheck:
	mypy ringdownanalysis --exclude 'legacy_ring_down_mc\.py'

test:
	pytest --cov=ringdownanalysis --cov-report=term-missing --cov-report=xml -q
