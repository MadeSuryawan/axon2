# Axon Codebase Improvement Plan

This document outlines a comprehensive implementation plan to improve Axon's performance and result quality.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Performance Improvements](#performance-improvements)
3. [Quality Improvements](#quality-improvements)
4. [Implementation Roadmap](#implementation-roadmap)
5. [Priority Matrix](#priority-matrix)

---

## Executive Summary

Based on comprehensive analysis of the Axon codebase, this plan identifies key areas for improvement:

| Category | Focus Areas | Expected Impact |
| -------- | ----------- | --------------- |
| **Performance** | Incremental indexing, parallel processing, caching | 10-50x faster for large repos |
| **Quality** | Call resolution, type analysis, language support | More accurate knowledge graph |
| **Scalability** | Memory optimization, distributed processing | Handle 10k+ files |

---

## Performance Improvements

### 1. Implement Incremental Indexing (Phase 0)

**Current State:** Phase 0 is reserved but not implemented. Every run does a full re-index.

**Problem:** For large codebases, re-indexing all files on every change is slow.

**Solution:**

- Implement file hash comparison to detect unchanged files
- Only re-process files that have changed
- Preserve unchanged nodes in storage

**Files to Modify:**

- [`src/axon/core/ingestion/pipeline.py`](src/axon/core/ingestion/pipeline.py)
- [`src/axon/core/ingestion/walker.py`](src/axon/core/ingestion/walker.py)
- [`src/axon/core/storage/kuzu_backend.py`](src/axon/core/storage/kuzu_backend.py)

**Implementation:**

```python
# Pseudo-code for incremental hashing
def compute_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def get_changed_files(old_hashes: dict, new_hashes: dict) -> list[str]:
    return [path for path, hash in new_hashes.items() 
            if old_hashes.get(path) != hash]
```

---

### 2. Parallel Phase Execution

**Current State:** Phases 1-11 run sequentially in a loop.

**Problem:** Some phases are independent and could run in parallel.

**Solution:**

- Identify independent phases
- Use asyncio/concurrent.futures for parallel execution
- Phase dependency analysis

**Phase Dependencies:**

```text
Phase 1 (Walker)     → Phase 2 (Structure)
Phase 2 (Structure)  → Phase 3 (Parsing)
Phase 3 (Parsing)    → Phase 4 (Imports), Phase 5 (Calls), Phase 6 (Heritage), Phase 7 (Types)
Phase 4-7 (Local)    → Phase 8 (Community) - CAN RUN IN PARALLEL
Phase 8-11 (Global)  → Must run sequentially after local phases
```

**Implementation:**

```python
# Parallel execution for independent phases
async def run_parallel_phases(phases: list[Phase]) -> dict:
    async with asyncio.TaskGroup() as tg:
        results = [tg.create_task(phase.run()) for phase in phases]
    return {r.key: r.result() for r in results}
```

---

### 3. Enhanced Caching Strategy

**Current State:** Limited caching - parser instances cached, model cached with LRU.

**Improvements:**

#### a) Symbol Lookup Cache

- Cache symbol resolution results
- Invalidate on file changes

```python
@lru_cache(maxsize=10000)
def resolve_symbol(file_path: str, name: str) -> NodeId | None:
    # Current implementation in symbol_lookup.py
```

#### b. Import Resolution Cache

- Cache module-to-file mappings
- Pre-compute for frequently imported modules

#### c. Graph Query Cache

- Cache common queries (e.g., "all entry points")
- TTL-based invalidation for watch mode

---

### 4. Optimized BFS for Process Detection

**Current State:** [`Processes._trace_all_flows()`](src/axon/core/ingestion/processes.py:89) uses simple BFS with `_MAX_FLOW_SIZE = 25`.

**Problem:** Can be slow for deeply nested call chains.

**Improvements:**

- Implement early termination heuristics
- Add memoization for visited paths
- Use iterative deepening instead of BFS

```python
def trace_flow_optimized(self, entry_point: Node, max_depth: int = 10) -> list[Flow]:
    visited = set()  # Memoization
    results = []
    
    def dfs(node, path, depth):
        if depth > max_depth or node.id in visited:
            return
        visited.add(node.id)
        path.append(node)
        
        for callee in self._get_callees(node):
            if not callee.is_dead:
                dfs(callee, path.copy(), depth + 1)
        
        if len(path) > 1:
            results.append(path)
    
    dfs(entry_point, [], 0)
    return results
```

---

### 5. Query Performance Optimization

**Current State:** Fuzzy search scans all tables with Levenshtein.

**Improvements:**

#### a. Index-Based Fuzzy Search

- Use trigram indexes for fuzzy matching
- Pre-compute common query results

#### b. Vector Search Optimization

- Batch multiple vector searches
- Use approximate nearest neighbors (ANN) for large datasets

```python
# Current: scans all nodes
# Improved: use HNSW indexes in KuzuDB
conn.execute("""
    CREATE HNSW INDEX IF NOT EXISTS embedding_idx 
    ON Embedding(vec) USING hnsw 
    WITH (m = 16, ef_construction = 200)
""")
```

---

### 6. Memory Optimization

**Current State:** Full graph loaded in memory, then bulk-loaded to Kuzu.

**Improvements:**

- Stream processing for large graphs
- Memory-mapped file support
- Incremental storage writes

---

### 7. Batch Processing Improvements

**Current State:** Files parsed in parallel, but other phases are sequential.

**Improvements:**

- Batch import resolution
- Batch call resolution
- Batch relationship creation

```python
# Batch import resolution
def process_imports_batch(parse_results: list[ParseResult]) -> list[Rel]:
    # Group by target module
    target_groups = defaultdict(list)
    for pr in parse_results:
        for imp in pr.imports:
            target = resolve_module(imp.module)
            if target:
                target_groups[target].append(imp)
    
    # Process groups in parallel
    return parallel_map(create_import_rels, target_groups.items())
```

---

## Quality Improvements

### 1. Enhanced Call Resolution

**Current State:** [`Calls._resolve_and_link_call()`](src/axon/core/ingestion/calls.py:266) uses confidence scoring:

1. Same-file exact match (1.0)
2. Import-resolved match (1.0)
3. Global fuzzy match (0.5)
4. Receiver method resolution (0.8)

**Improvements:**

#### a. Type-Based Resolution

- Use type annotations to resolve calls
- Track parameter types and return types

```python
def resolve_call_with_types(call: CallInfo, context: TypeContext) -> Resolution:
    # Get parameter types from signature
    param_types = get_parameter_types(call.target_signature)
    
    # Match with argument types
    for caller in get_callers(call.name):
        if types_compatible(caller.param_types, param_types):
            return Resolution(caller, confidence=0.95)
    
    return fallback_resolution(call)
```

#### b. Context-Aware Resolution

- Consider lexical scope
- Handle closures and lambdas
- Track class hierarchies

#### c. Confidence Aggregation

- Aggregate confidence from multiple resolution strategies
- Weight by specificity

---

### 2. Improved Type Analysis

**Current State:** Basic USES_TYPE edges from [`process_types()`](src/axon/core/ingestion/types.py).

**Improvements:**

#### a. Generic Type Parameter Tracking

- Resolve `List[T]`, `Dict[K,V]`, `Optional[T]`
- Build type parameter maps

#### b. Structural Typing

- Support TypeScript interfaces
- Handle duck typing in Python

#### c. Type Inference

- Infer types from usage when annotations are missing
- Use pyright/mypy-like inference (optional, heavy)

---

### 3. Enhanced Dead Code Detection

**Current State:** 4-pass algorithm in [`DeadCode`](src/axon/core/ingestion/dead_code.py):

1. Flag initial dead nodes
2. Clear override false positives
3. Clear protocol conformance false positives
4. Clear protocol stub false positives

**Improvements:**

#### a. Dynamic Call Detection

- Consider runtime registrations (e.g., `@app.register`)
- Handle plugin systems
- Recognize factory patterns

```python
# Enhanced exemptions
DYNAMIC_REGISTRATION_PATTERNS = frozenset({
    "register",
    "add_handler", 
    "connect",
    "subscribe",
    "on_change",
})
```

#### b. Test Coverage Integration

- Integrate with coverage.py data
- Consider test-covered code as "used"

#### c. Configuration-Driven Detection

- Allow project-specific exemptions
- Add `.axon/dead_code_ignore` config

---

### 4. Extended Language Support

**Current State:** Python, TypeScript, JavaScript only.

**Improvements:**

#### a. Go Support (High Priority)

- Use tree-sitter-go
- Handle go.mod analysis

#### b. Rust Support (Medium Priority)

- Use tree-sitter-rust
- Handle Cargo.toml

#### c. Java/Kotlin Support (Lower Priority)

- Use tree-sitter-java/tree-sitter-kotlin

#### d. Ruby/PHP Support (Lower Priority)

- Framework pattern detection

---

### 5. Better Framework Pattern Detection

**Current State:** Limited patterns in [`Processes._matches_framework_pattern()`](src/axon/core/ingestion/processes.py:163).

**Improvements:**

- Add more framework patterns (FastAPI, Django, Rails, etc.)
- Auto-detect framework from project structure
- Handle multiple frameworks in same project

```python
FRAMEWORK_PATTERNS = {
    "fastapi": ["@app.get", "@app.post", "@router.get"],
    "django": ["@path", "@include", "urlpatterns"],
    "flask": ["@app.route", "@bp.route"],
    "react": ["useState", "useEffect", "createElement"],
    "nextjs": ["getServerSideProps", "getStaticProps"],
}
```

---

### 6. Improved Community Detection

**Current State:** Leiden algorithm on call graph only.

**Improvements:**

#### a. Multi-Graph Clustering

- Consider import relationships
- Consider type relationships
- Consider file co-change patterns

#### b. Hierarchical Communities

- Detect communities at multiple granularities
- Identify sub-communities

```python
# Multi-layer community detection
def detect_communities(graph, layers: list[str]) -> dict:
    results = {}
    for layer in layers:
        g_layer = extract_layer(graph, layer)
        results[layer] = leiden_detect(g_layer)
    return results
```

---

### 7. Enhanced Process Detection

**Current State:** BFS from entry points with simple deduplication.

**Improvements:**

#### a. Framework-Aware Process Detection

- Detect API routes as processes
- Detect event handlers as processes
- Detect CI/CD pipelines

#### b. Process Similarity Detection

- Use graph similarity algorithms
- Group similar processes

#### c. Cross-Language Flow Detection

- Track across language boundaries
- Handle FFI and WebAssembly

---

### 8. Better Change Coupling Analysis

**Current State:** Simple co-change calculation from git history.

**Improvements:**

#### a. Weighted Coupling

- Weight by recency
- Weight by change frequency

#### b. Causal Coupling

- Detect if A always precedes B in commits
- Use commit message analysis

#### c. Semantic Coupling

- Combine with call graph
- Identify architectural dependencies

---

### 9. Improved Search Quality

**Current State:** Hybrid search with RRF combining BM25, vector, and fuzzy.

**Improvements:**

#### a. Learning to Rank

- Use click-through data to improve ranking
- Incorporate user feedback

#### b. Context-Aware Search

- Consider current file context
- Boost local definitions

#### c. Semantic Expansion

- Use embeddings to expand queries
- Synonym expansion

---

### 10. MCP Tool Enhancements

**Current State:** 7 tools, 3 resources.

**Improvements:**

#### a. New Tools

- `axon_refactor()` - Suggest refactoring based on impact analysis
- `axon_diagram()` - Generate architecture diagrams
- `axon_summary()` - Generate code summaries using LLMs

#### b. Enhanced Responses

- Include code snippets
- Add visualization hints
- Provide next-step suggestions

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 sprints)

1. ✅ Create visualization scripts (DONE)
2. ⬜ Implement incremental hashing for file change detection
3. ⬜ Add parallel phase execution for phases 8-11
4. ⬜ Optimize BFS with early termination
5. ⬜ Add fuzzy search index
6. ⬜ Incremental global phase optimization for watch mode

### Phase 2: Performance (2-3 sprints)

1. ⬜ Implement full incremental indexing (Phase 0)
2. ⬜ Add comprehensive caching layer
3. ⬜ Optimize batch processing
4. ⬜ Memory optimization for large graphs

### Phase 3: Quality (2-3 sprints)

1. ⬜ Enhanced call resolution with types
2. ⬜ Improved type analysis
3. ⬜ Extended language support (Go, Rust)
4. ⬜ Better framework detection

### Phase 4: Advanced Features (3-4 sprints)

1. ⬜ Learning-to-rank search
2. ⬜ Multi-graph community detection
3. ⬜ New MCP tools
4. ⬜ Integration with external tools

---

## Priority Matrix

| Priority | Feature | Effort | Impact | Quadrant |
| -------- | ------- | ------ | ------ | -------- |
| P0 | Incremental Indexing | High | Very High | Quick Wins |
| P0 | Call Resolution (Types) | High | Very High | Quality |
| P1 | Parallel Phases | Medium | High | Quick Wins |
| P1 | Incremental Global Phases | Medium | High | Quick Wins |
| P1 | Extended Languages | Medium | High | Quality |
| P1 | Enhanced Caching | Medium | High | Performance |
| P2 | Better Search | Medium | Medium | Quality |
| P2 | Community Detection+ | High | Medium | Quality |
| P2 | Memory Optimization | High | Medium | Performance |
| P3 | New MCP Tools | Medium | Medium | Features |
| P3 | LLM Integration | High | Medium | Features |

---

## Notes

- This plan is a living document and should be updated as priorities change
- Focus on measurable improvements (indexing time, search latency, accuracy metrics)
- Consider A/B testing for search quality improvements
- Gather user feedback to prioritize features

---

## Verified Implementation Status (2026-03-09)

This section documents what has been verified as already implemented vs. what needs improvement.

### Already Optimized (No Changes Needed)

| Area | Current Implementation | Status |
| ---- | ---------------------- | -------- |
| **Symbol Lookup** | O(log N) binary search via [`FileSymbolIndex`](src/axon/core/ingestion/symbol_lookup.py:37) | ✅ Already optimized |
| **Process BFS** | Has `_MAX_FLOW_SIZE=25`, `max_depth=6`, `max_branching=3`, visited memoization | ✅ Already has limits |
| **Dead Code** | 4-pass algorithm with framework decorators, protocol, test file exemptions | ✅ Comprehensive |
| **Search** | BM25 FTS + vector + fuzzy (Levenshtein) with RRF | ✅ Hybrid implemented |
| **Storage** | CSV bulk loading, FTS indexes | ✅ Implemented |
| **Parser Caching** | Parser instances cached per language | ✅ Implemented |

### Verified NOT Implemented (Keep in Plan)

| Area | Current State | Plan Item |
| ---- | ------------- | --------- |
| **Incremental Phase 0** | Reserved but not implemented | Section 1 |
| **Parallel Phase Execution** | Sequential pipeline | Section 2 |
| **Enhanced Caching** | Limited caching | Section 3 |
| **Type-Based Call Resolution** | No type info used in resolution | Section Quality 1 |
| **Generic Type Tracking** | Basic USES_TYPE edges only | Section Quality 2 |
| **Extended Languages** | Python, TypeScript, JavaScript only | Section Quality 4 |

### Plan Items Requiring Clarification

| Area | Finding | Recommendation |
| ---- | ------- | -------------- |
| **Section 4 (BFS)** | Code already has limits but DFS could improve accuracy | Keep improvement - focuses on flow completeness |
| **Section 5 (Fuzzy)** | Scans all tables, no trigram index | Keep improvement - add trigram index for fuzzy |

---

## Missing Practical Improvements

After analysis, the following practical improvements should be added:

### 8. Incremental Global Phase Optimization

**Current State:** In watch mode, global phases (community, process, dead code) re-hydrate the entire graph from storage.

**Problem:** For large codebases, full graph hydration is slow even for small file changes.

**Solution:**

- Implement incremental community/process updates using diff-based approach
- Only re-run community detection on affected subgraphs
- Use delta updates for dead code flags

```python
# Incremental global phase pseudo-code
def update_communities_incrementally(graph, changed_node_ids):
    affected_subgraph = extract_subgraph(changed_node_ids, depth=2)
    affected_community = detect_communities(affected_subgraph)
    
    # Merge with existing communities
    for node_id in changed_node_ids:
        old_community = get_community(node_id)
        new_community = affected_community.get(node_id)
        if old_community != new_community:
            update_membership(node_id, old_community, new_community)
```
