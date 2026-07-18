#!/usr/bin/env python3
"""
validate_repository.py — Validate the research repository structure.
From Biological Neurons to LLMs

Usage:
    python scripts/validate_repository.py

Checks:
    - Every paper has metadata.yaml
    - Every paper has bibliography.bib
    - Every paper has paper_source.md
    - Every paper has notes.md
    - No empty files (0 bytes)
    - Folder names match the {YYYY_Author_ShortTitle} pattern
    - DOIs are well-formed (if present)
    - Implementation filenames follow the standard
    - Metadata fields are complete
"""

import os
import re
import sys
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

REPO_ROOT = Path(__file__).parent.parent
PAPERS_DIR = REPO_ROOT / "research" / "papers"

# Required files per paper
REQUIRED_FILES = [
    "metadata.yaml",
    "bibliography.bib",
    "paper_source.md",
    "notes.md"
]

# Optional but recommended files
OPTIONAL_FILES = [
    "summary.md",
    "questions.md",
    "timeline.md",
    "equations.tex",
    "implementation_historical.py",
    "implementation_modern.py"
]

# Required metadata fields
REQUIRED_METADATA_FIELDS = [
    "id", "title", "authors", "year", "venue",
    "primary_subject", "historical_importance",
    "implementation", "predecessors", "successors",
    "knowledge_flow", "phase", "era", "status", "last_updated"
]

# Folder name pattern: {YYYY_Author_ShortTitle}
FOLDER_PATTERN = re.compile(r'^(\d{4})_([A-Za-z]+)_([A-Za-z0-9_]+)$')

# DOI pattern
DOI_PATTERN = re.compile(r'^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$')

# ============================================================
# Validation Functions
# ============================================================

def validate_dois():
    """Check that all DOIs in bibliography.bib are well-formed."""
    do_errors = False
    for paper_dir in PAPERS_DIR.iterdir():
        if not paper_dir.is_dir():
            continue
        bib_file = paper_dir / "bibliography.bib"
        if bib_file.exists():
            content = bib_file.read_text()
            # Extract DOI entries
            doi_matches = re.findall(r'doi\s*=\s*\{([^}]+)\}', content)
            for doi in doi_matches:
                if not DOI_PATTERN.match(doi):
                    print(f"⚠️ {paper_dir.name}: Malformed DOI: {doi}")
                    do_errors = True
        else:
            print(f"⚠️ {paper_dir.name}: Missing bibliography.bib")
            do_errors = True
    return do_errors


def validate_metadata():
    """Check that all metadata.yaml files exist and have required fields."""
    has_errors = False
    for paper_dir in PAPERS_DIR.iterdir():
        if not paper_dir.is_dir():
            continue
        metadata_file = paper_dir / "metadata.yaml"
        if not metadata_file.exists():
            print(f"⚠️ {paper_dir.name}: Missing metadata.yaml")
            has_errors = True
            continue

        # Parse metadata (simple YAML-like check)
        content = metadata_file.read_text()
        for field in REQUIRED_METADATA_FIELDS:
            if field not in content:
                print(f"⚠️ {paper_dir.name}: Missing metadata field: {field}")
                has_errors = True
    return has_errors


def validate_folder_names():
    """Check that folder names follow the pattern."""
    has_errors = False
    for paper_dir in PAPERS_DIR.iterdir():
        if not paper_dir.is_dir():
            continue
        if not FOLDER_PATTERN.match(paper_dir.name):
            print(f"⚠️ Invalid folder name: {paper_dir.name}")
            has_errors = True
    return has_errors


def validate_required_files():
    """Check that each paper has all required files."""
    has_errors = False
    for paper_dir in PAPERS_DIR.iterdir():
        if not paper_dir.is_dir():
            continue
        for req_file in REQUIRED_FILES:
            file_path = paper_dir / req_file
            if not file_path.exists():
                print(f"⚠️ {paper_dir.name}: Missing required file: {req_file}")
                has_errors = True
    return has_errors


def validate_empty_files():
    """Check for empty files (0 bytes)."""
    has_errors = False
    for paper_dir in PAPERS_DIR.iterdir():
        if not paper_dir.is_dir():
            continue
        for file_path in paper_dir.glob("*"):
            if file_path.is_file() and file_path.stat().st_size == 0:
                print(f"⚠️ {paper_dir.name}: Empty file: {file_path.name}")
                has_errors = True
    return has_errors


def validate_implementations():
    """Check that implementation filenames follow the standard."""
    has_errors = False
    for paper_dir in PAPERS_DIR.iterdir():
        if not paper_dir.is_dir():
            continue
        # Check for generic implementation.py (should not exist)
        if (paper_dir / "implementation.py").exists():
            print(f"⚠️ {paper_dir.name}: Remove generic implementation.py")
            has_errors = True
        # Check for historical and modern variants
        has_hist = (paper_dir / "implementation_historical.py").exists()
        has_mod = (paper_dir / "implementation_modern.py").exists()
        # If one exists, the other should also exist
        if has_hist != has_mod:
            if has_hist:
                print(f"⚠️ {paper_dir.name}: Has historical but missing modern")
            else:
                print(f"⚠️ {paper_dir.name}: Has modern but missing historical")
            has_errors = True
    return has_errors


# Add these functions inside validate_repository.py

def validate_metadata_content(paper_dir):
    """Check metadata.yaml for schema version, implementation, reproducibility."""
    metadata_file = paper_dir / "metadata.yaml"
    if not metadata_file.exists():
        return False
    try:
        import yaml
        with open(metadata_file, 'r') as f:
            data = yaml.safe_load(f)
        errors = False
        if 'schema_version' not in data:
            print(f"⚠️ {paper_dir.name}: Missing 'schema_version' in metadata")
            errors = True
        if 'implementation' not in data:
            print(f"⚠️ {paper_dir.name}: Missing 'implementation' field")
            errors = True
        elif not isinstance(data['implementation'], dict):
            print(f"⚠️ {paper_dir.name}: 'implementation' should be a dict")
            errors = True
        else:
            for key in ['historical', 'modern']:
                if key not in data['implementation']:
                    print(f"⚠️ {paper_dir.name}: Missing implementation.{key}")
                    errors = True
                elif not isinstance(data['implementation'][key], dict):
                    print(f"⚠️ {paper_dir.name}: implementation.{key} should be a dict")
                    errors = True
                else:
                    if 'status' not in data['implementation'][key]:
                        print(f"⚠️ {paper_dir.name}: Missing implementation.{key}.status")
                        errors = True
                    if 'verified' not in data['implementation'][key]:
                        print(f"⚠️ {paper_dir.name}: Missing implementation.{key}.verified")
                        errors = True
        if 'reproducibility' not in data:
            print(f"⚠️ {paper_dir.name}: Missing 'reproducibility' field")
            errors = True
        else:
            for field in ['python', 'numpy', 'last_tested']:
                if field not in data['reproducibility']:
                    print(f"⚠️ {paper_dir.name}: Missing reproducibility.{field}")
                    errors = True
        return not errors
    except Exception as e:
        print(f"⚠️ {paper_dir.name}: Error reading metadata: {e}")
        return False

# In the main validation function, call this after checking required files.

def main():
    print("=" * 60)
    print("Repository Validation: From Biological Neurons to LLMs")
    print("=" * 60)

    errors_found = False

    # Run all checks
    checks = [
        ("Folder names", validate_folder_names),
        ("Required files", validate_required_files),
        ("Empty files", validate_empty_files),
        ("Metadata fields", validate_metadata),
        ("DOIs", validate_dois),
        ("Implementations", validate_implementations),
    ]

    for name, func in checks:
        print(f"\n📋 Checking: {name}...")
        if func():
            errors_found = True

    # Count papers
    paper_count = len([d for d in PAPERS_DIR.iterdir() if d.is_dir()])
    print(f"\n📊 {paper_count} papers checked.")

    if errors_found:
        print("\n❌ Validation FAILED. Please fix the issues above.")
        sys.exit(1)
    else:
        print("\n✅ All checks passed! Repository is valid.")
        sys.exit(0)


if __name__ == "__main__":
    main()