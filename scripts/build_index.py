#!/usr/bin/env python3
"""
build_index.py — Generate research/index.yaml from paper folders.
From Biological Neurons to LLMs

Usage:
    python scripts/build_index.py

This script scans all paper folders and builds a machine-readable index.
"""

import os
import yaml
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).parent.parent
PAPERS_DIR = REPO_ROOT / "research" / "papers"
INDEX_FILE = REPO_ROOT / "research" / "index.yaml"


def load_metadata(paper_dir):
    """Load metadata.yaml from a paper folder."""
    metadata_file = paper_dir / "metadata.yaml"
    if not metadata_file.exists():
        return None
    try:
        with open(metadata_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"⚠️ Error loading {paper_dir.name}/metadata.yaml: {e}")
        return None


def is_paper_complete(metadata):
    """Check if a paper is complete based on metadata status."""
    if not metadata:
        return False
    status = metadata.get('status', {})
    # Handle both old format (dict with notes/bibliography) and new format
    if isinstance(status, dict):
        # Check if notes and bibliography are marked as complete
        notes_status = status.get('notes', '')
        bib_status = status.get('bibliography', '')
        return notes_status == 'complete' and bib_status == 'complete'
    # If status is a string, check if it's 'complete'
    return status == 'complete'


def build_index():
    """Build the complete index from all paper folders."""
    papers = []
    completed_count = 0
    total_count = 0

    for paper_dir in sorted(PAPERS_DIR.iterdir()):
        if not paper_dir.is_dir():
            continue
        if paper_dir.name.startswith('.'):
            continue

        total_count += 1
        metadata = load_metadata(paper_dir)

        if metadata:
            status = metadata.get('status', {})
            if isinstance(status, dict):
                notes_status = status.get('notes', '')
                bib_status = status.get('bibliography', '')
                is_complete = notes_status == 'complete' and bib_status == 'complete'
            else:
                is_complete = status == 'complete'

            paper_info = {
                'id': metadata.get('id', paper_dir.name),
                'title': metadata.get('title', ''),
                'year': metadata.get('year', 0),
                'authors': metadata.get('authors', []),
                'venue': metadata.get('venue', ''),
                'doi': metadata.get('doi', ''),
                'phase': metadata.get('phase', ''),
                'era': metadata.get('era', ''),
                'status': status,
                'historical_importance': metadata.get('historical_importance', 0)
            }

            if is_complete:
                completed_count += 1

            papers.append(paper_info)
        else:
            # Minimal entry from folder name
            papers.append({
                'id': paper_dir.name,
                'title': paper_dir.name,
                'year': 0,
                'authors': [],
                'venue': '',
                'doi': '',
                'phase': '',
                'era': '',
                'status': 'pending',
                'historical_importance': 0
            })

    index = {
        'project': {
            'name': 'From Biological Neurons to Large Language Models',
            'version': '0.3.0',
            'last_updated': datetime.now().isoformat()
        },
        'summary': {
            'total_papers': total_count,
            'completed': completed_count,
            'pending': total_count - completed_count
        },
        'papers': papers
    }

    with open(INDEX_FILE, 'w') as f:
        yaml.dump(index, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Index generated: {INDEX_FILE}")
    print(f"   Total papers: {total_count}")
    print(f"   Completed: {completed_count}")
    print(f"   Pending: {total_count - completed_count}")


if __name__ == "__main__":
    build_index()
