#!/usr/bin/env python3
"""
build_graph.py — Generate knowledge graph with visualization.
From Biological Neurons to LLMs

Usage:
    python scripts/build_graph.py

Generates:
    research/graph/knowledge_graph.json
    research/graph/knowledge_graph.dot
    research/graph/knowledge_graph.svg (requires graphviz)
"""

import json
import yaml
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
PAPERS_DIR = REPO_ROOT / "research" / "papers"
GRAPH_DIR = REPO_ROOT / "research" / "graph"
GRAPH_FILE = GRAPH_DIR / "knowledge_graph.json"
DOT_FILE = GRAPH_DIR / "knowledge_graph.dot"
SVG_FILE = GRAPH_DIR / "knowledge_graph.svg"


def load_metadata(paper_dir):
    """Load metadata.yaml from a paper folder."""
    metadata_file = paper_dir / "metadata.yaml"
    if not metadata_file.exists():
        return None
    try:
        with open(metadata_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def build_graph():
    """Build the knowledge graph and generate visualizations."""
    # Ensure graph directory exists
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    nodes = []
    edges = []

    for paper_dir in sorted(PAPERS_DIR.iterdir()):
        if not paper_dir.is_dir():
            continue

        metadata = load_metadata(paper_dir)
        if not metadata:
            continue

        node = {
            'id': metadata.get('id', paper_dir.name),
            'label': metadata.get('title', paper_dir.name)[:50] + '...',
            'year': metadata.get('year', 0),
            'phase': metadata.get('phase', ''),
            'historical_importance': metadata.get('historical_importance', 0),
            'has_implementation': metadata.get('implementation', {}).get('historical', False)
        }
        nodes.append(node)

        # Add edges from predecessors and successors
        for pred in metadata.get('predecessors', []):
            edges.append({
                'source': pred,
                'target': metadata.get('id', paper_dir.name),
                'type': 'influences'
            })

        for succ in metadata.get('successors', []):
            edges.append({
                'source': metadata.get('id', paper_dir.name),
                'target': succ,
                'type': 'influences'
            })

    graph = {
        'nodes': nodes,
        'edges': edges,
        'metadata': {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'generated': __import__('datetime').datetime.now().isoformat()
        }
    }

    # Save JSON
    with open(GRAPH_FILE, 'w') as f:
        json.dump(graph, f, indent=2)
    print(f"✅ JSON graph: {GRAPH_FILE}")

    # Generate DOT file
    dot_content = generate_dot(graph)
    with open(DOT_FILE, 'w') as f:
        f.write(dot_content)
    print(f"✅ DOT graph: {DOT_FILE}")

    # Generate SVG using graphviz (if available)
    try:
        result = subprocess.run(
            ['dot', '-Tsvg', str(DOT_FILE), '-o', str(SVG_FILE)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ SVG graph: {SVG_FILE}")
        else:
            print(f"⚠️ Graphviz not available. Install: brew install graphviz (Mac) or apt-get install graphviz (Linux)")
            print(f"   To generate SVG manually: dot -Tsvg {DOT_FILE} -o {SVG_FILE}")
    except FileNotFoundError:
        print(f"⚠️ Graphviz 'dot' command not found. Install graphviz to generate SVG.")
        print(f"   To generate SVG manually: dot -Tsvg {DOT_FILE} -o {SVG_FILE}")

    print(f"\n📊 Nodes: {len(nodes)}, Edges: {len(edges)}")


def generate_dot(graph):
    """Generate Graphviz DOT format."""
    lines = [
        'digraph KnowledgeGraph {',
        '  rankdir=TB;',
        '  node [shape=box, style=filled, fillcolor=lightyellow];',
        '  edge [arrowhead=normal];',
        ''
    ]

    # Add nodes
    for node in graph['nodes']:
        label = node['label'].replace('"', '\\"')
        lines.append(f'  "{node["id"]}" [label="{label}\\n({node["year"]})"];')

    # Add edges
    for edge in graph['edges']:
        lines.append(f'  "{edge["source"]}" -> "{edge["target"]}" [label="{edge["type"]}"];')

    lines.append('}')
    return '\n'.join(lines)


if __name__ == "__main__":
    build_graph()