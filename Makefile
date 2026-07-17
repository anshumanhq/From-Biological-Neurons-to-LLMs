# ============================================================
# Makefile — From Biological Neurons to LLMs
# ============================================================

.PHONY: help validate test index graph all clean

help:
	@echo "Available commands:"
	@echo "  make validate  — Run repository validation"
	@echo "  make test      — Run pytest tests"
	@echo "  make index     — Build research/index.yaml"
	@echo "  make graph     — Build knowledge graph (JSON/DOT/SVG)"
	@echo "  make all       — Run validate, index, graph, test"
	@echo "  make clean     — Remove temporary files"

validate:
	@echo "🔍 Running repository validation..."
	python scripts/validate_repository.py

test:
	@echo "🧪 Running tests..."
	pytest code/tests/ -v --tb=short

index:
	@echo "📇 Building index..."
	python scripts/build_index.py

graph:
	@echo "📊 Building knowledge graph..."
	python scripts/build_graph.py

all: validate index graph test
	@echo "✅ All tasks completed successfully."

clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	@echo "✅ Cleanup complete."
