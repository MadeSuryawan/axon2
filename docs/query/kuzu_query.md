# Kùzu Query Guide for Axon Knowledge Graph

This document provides useful Cypher queries for exploring your Axon knowledge graph. You can run these queries using:

- **KùzuExplorer** (<http://localhost:8000>) after running `./scripts/visualize-graph.sh`
- **Axon CLI**: `axon cypher "MATCH ..."`
- **Python API**: Using the `kuzu` Python package

---

## Table of Contents

1. [Database Overview](#database-overview)
2. [File & Folder Queries](#file--folder-queries)
3. [Function & Method Queries](#function--method-queries)
4. [Class & Interface Queries](#class--interface-queries)
5. [Call Graph Analysis](#call-graph-analysis)
6. [Import Analysis](#import-analysis)
7. [Type Analysis](#type-analysis)
8. [Community Detection](#community-detection)
9. [Process/Execution Flow](#processexecution-flow)
10. [Dead Code Detection](#dead-code-detection)
11. [Change Coupling](#change-coupling)
12. [Search & Filtering](#search--filtering)

---

## Database Overview

### Get Node Count by Type

```cypher
MATCH (n) 
RETURN labels(n)[0] as node_type, count(*) as count 
ORDER BY count DESC
```

**Explanation**: Returns the count of all nodes grouped by their type (File, Function, Class, etc.)

---

### Get Relationship Count by Type

```cypher
MATCH ()-[r]->() 
RETURN type(r) as relationship_type, count(*) as count 
ORDER BY count DESC
```

**Explanation**: Returns the count of all relationships grouped by type (CALLS, IMPORTS, etc.)

---

### View Full Schema

```cypher
CALL table_info();
```

**Explanation**: Displays the complete database schema including all node tables and their properties.

---

## File & Folder Queries

### List All Files

```cypher
MATCH (f:File) 
RETURN f.name, f.file_path, f.language 
ORDER BY f.name
LIMIT 50
```

**Explanation**: Returns all source files in the project with their paths and languages.

---

### Find Files by Pattern

```cypher
MATCH (f:File) 
WHERE f.file_path CONTAINS 'auth' OR f.file_path CONTAINS 'test'
RETURN f.name, f.file_path
ORDER BY f.file_path
```

**Explanation**: Finds files matching a specific pattern in their path.

---

### Get Folder Structure

```cypher
MATCH (folder:Folder)-[:CONTAINS]->(f:File)
RETURN folder.name as folder, collect(f.name) as files
ORDER BY folder.name
LIMIT 20
```

**Explanation**: Shows the folder structure and lists files within each folder.

---

### Files with Most Symbols

```cypher
MATCH (f:File)-[:DEFINES]->(s:Symbol)
WITH f, count(s) as symbol_count
RETURN f.name, symbol_count
ORDER BY symbol_count DESC
LIMIT 10
```

**Explanation**: Finds files that define the most symbols (functions, classes, etc.)

---

## Function & Method Queries

### List All Functions

```cypher
MATCH (f:Function) 
RETURN f.name, f.file_path, f.signature
ORDER BY f.name
LIMIT 50
```

**Explanation**: Returns all top-level functions with their signatures.

---

### List All Methods

```cypher
MATCH (m:Method) 
RETURN m.name, m.class_name, m.file_path
ORDER BY m.class_name, m.name
LIMIT 50
```

**Explanation**: Returns all class methods, grouped by their owning class.

---

### Functions in a Specific File

```cypher
MATCH (f:File {name: 'main.py'})-[:DEFINES]->(s:Symbol)
RETURN s.name, s.kind
```

**Explanation**: Finds all symbols (functions, classes, methods) defined in a specific file.

---

### Find Function by Name

```cypher
MATCH (f:Function) 
WHERE f.name = 'validate_user'
RETURN f.name, f.file_path, f.start_line, f.content
```

**Explanation**: Finds a specific function by exact name.

---

### Functions with Long Lines

```cypher
MATCH (f:Function)
WHERE f.end_line - f.start_line > 50
RETURN f.name, f.file_path, f.start_line, f.end_line, 
       f.end_line - f.start_line as lines
ORDER BY lines DESC
LIMIT 10
```

**Explanation**: Finds functions with more than 50 lines - useful for identifying complex/large functions.

---

## Class & Interface Queries

### List All Classes

```cypher
MATCH (c:Class) 
RETURN c.name, c.file_path, c.start_line
ORDER BY c.name
LIMIT 50
```

**Explanation**: Returns all class definitions.

---

### List All Interfaces

```cypher
MATCH (i:Interface) 
RETURN i.name, i.file_path
ORDER BY i.name
LIMIT 50
```

**Explanation**: Returns all interface/protocol definitions (TypeScript).

---

### Class Inheritance (EXTENDS)

```cypher
MATCH (child:Class)-[:EXTENDS]->(parent:Class)
RETURN child.name as child_class, parent.name as parent_class
ORDER BY child.name
```

**Explanation**: Shows class inheritance relationships.

---

### Interface Implementation (IMPLEMENTS)

```cypher
MATCH (class:Class)-[:IMPLEMENTS]->(interface:Interface)
RETURN class.name as class_name, interface.name as interface_name
ORDER BY class.name
```

**Explanation**: Shows which classes implement which interfaces.

---

### Class Members (Methods)

```cypher
MATCH (c:Class {name: 'UserService'})-[:DEFINES]->(m:Method)
RETURN m.name, m.signature
ORDER BY m.name
```

**Explanation**: Lists all methods defined in a specific class.

---

## Call Graph Analysis

### Direct Function Calls

```cypher
MATCH (caller:Symbol)-[r:CALLS]->(callee:Symbol)
RETURN caller.name as caller, callee.name as callee
LIMIT 50
```

**Explanation**: Shows direct call relationships between functions/methods.

---

### Who Calls a Specific Function?

```cypher
MATCH (caller:Symbol)-[:CALLS]->(callee:Function {name: 'validate'})
RETURN DISTINCT caller.name as caller, caller.file_path as file
ORDER BY caller.name
```

**Explanation**: Finds all functions that call a specific function.

---

### What Does a Function Call?

```cypher
MATCH (caller:Function {name: 'process_request'})-[:CALLS]->(callee:Symbol)
RETURN callee.name as called_function, callee.kind as type
ORDER BY callee.name
```

**Explanation**: Shows all functions/methods called by a specific function.

---

### Call Chain (Depth 2)

```cypher
MATCH (start:Function {name: 'main'})-[:CALLS*1..2]->(end:Function)
RETURN start.name, collect(DISTINCT end.name) as called_functions
LIMIT 10
```

**Explanation**: Shows functions reachable within 2 hops from the main function.

---

### Most Called Functions

```cypher
MATCH (caller:Symbol)-[r:CALLS]->(callee:Symbol)
WITH callee, count(r) as call_count
RETURN callee.name, call_count
ORDER BY call_count DESC
LIMIT 15
```

**Explanation**: Identifies the most frequently called functions - good candidates for critical/shared logic.

---

### Uncalled Functions

```cypher
MATCH (f:Function)
WHERE NOT (f)-[:CALLS]->(:Symbol)
AND NOT (:Symbol)-[:CALLS]->(f)
RETURN f.name, f.file_path
ORDER BY f.name
```

**Explanation**: Finds functions that neither call others nor are called by others - potential dead code or isolated utilities.

---

## Import Analysis

### All Import Relationships

```cypher
MATCH (importer:File)-[r:IMPORTS]->(imported:File)
RETURN importer.name as importing_file, imported.name as imported_file
LIMIT 50
```

**Explanation**: Shows which files import which other files.

---

### Find External Dependencies

```cypher
MATCH (f:File)-[r:IMPORTS]->(i:File)
WHERE i.file_path STARTS WITH 'node_modules' 
   OR i.file_path STARTS WITH 'site-packages'
   OR i.file_path STARTS WITH 'lib'
RETURN f.name, i.file_path
ORDER BY f.name
```

**Explanation**: Identifies external library dependencies.

---

### Most Imported Files

```cypher
MATCH (importer:File)-[:IMPORTS]->(imported:File)
WITH imported, count(*) as import_count
RETURN imported.name, import_count
ORDER BY import_count DESC
LIMIT 15
```

**Explanation**: Finds files that are imported most frequently - likely core utilities.

---

## Type Analysis

### Type Usage (USES_TYPE)

```cypher
MATCH (symbol:Symbol)-[r:USES_TYPE]->(type:TypeAlias)
RETURN symbol.name as symbol, type.name as type_used
LIMIT 50
```

**Explanation**: Shows which symbols use which type aliases.

---

### Variables with Specific Types

```cypher
MATCH (s:Symbol)-[:USES_TYPE]->(t:TypeAlias {name: 'UserConfig'})
RETURN s.name as variable, s.file_path as file
ORDER BY s.name
```

**Explanation**: Finds all variables using a specific type.

---

## Community Detection

### List All Communities

```cypher
MATCH (c:Community) 
RETURN c.name, c.cohesion, c.symbol_count
ORDER BY c.symbol_count DESC
```

**Explanation**: Shows all detected code communities/clusters with their cohesion scores.

---

### Members of a Community

```cypher
MATCH (s:Symbol)-[:MEMBER_OF]->(c:Community {name: 'auth'})
RETURN s.name, s.kind, s.file_path
ORDER BY s.kind, s.name
```

**Explanation**: Lists all symbols belonging to a specific community (e.g., 'auth').

---

### Cross-Community Dependencies

```cypher
MATCH (s1:Symbol)-[:CALLS]->(s2:Symbol)
WHERE s1 <> s2
WITH s1, s2,
     s1.file_path as file1, s2.file_path as file2
WHERE file1 <> file2
RETURN file1, file2, count(*) as call_count
ORDER BY call_count DESC
LIMIT 20
```

**Explanation**: Finds dependencies between different parts of the codebase.

---

## Process/Execution Flow

### List All Processes

```cypher
MATCH (p:Process) 
RETURN p.name, p.entry_point
ORDER BY p.name
```

**Explanation**: Shows all detected execution flows/processes.

---

### Steps in a Process

```cypher
MATCH (p:Process {name: 'user-auth'})-[:STEP_IN_PROCESS]->(s:Symbol)
RETURN s.name, s.kind, s.file_path
ORDER BY s.name
```

**Explanation**: Lists all symbols involved in a specific execution flow.

---

### Entry Points

```cypher
MATCH (s:Symbol) 
WHERE s.is_entry_point = true
RETURN s.name, s.kind, s.file_path
ORDER BY s.kind, s.name
```

**Explanation**: Shows all detected entry points (main functions, handlers, etc.)

---

## Dead Code Detection

### Find All Dead Code

```cypher
MATCH (s:Symbol) 
WHERE s.is_dead = true
RETURN s.name, s.kind, s.file_path, s.start_line
ORDER BY s.file_path, s.name
```

**Explanation**: Lists all unreachable/dead code symbols.

---

### Dead Code Count by File

```cypher
MATCH (s:Symbol) 
WHERE s.is_dead = true
WITH s.file_path as file, count(s) as dead_count
RETURN file, dead_count
ORDER BY dead_count DESC
```

**Explanation**: Shows files with the most dead code.

---

### Dead Functions

```cypher
MATCH (f:Function) 
WHERE f.is_dead = true
RETURN f.name, f.file_path
ORDER BY f.file_path
```

**Explanation**: Lists all unreachable functions.

---

## Change Coupling

### Coupled Files (Change Together)

```cypher
MATCH (a:File)-[r:COUPLED_WITH]->(b:File)
RETURN a.name as file_a, b.name as file_b, r.strength, r.co_changes
ORDER BY r.strength DESC
LIMIT 20
```

**Explanation**: Finds files that frequently change together - indicates strong coupling.

---

### Files Most Coupled with a Specific File

```cypher
MATCH (target:File {name: 'user.py'})-[r:COUPLED_WITH]->(coupled:File)
RETURN coupled.name, r.strength, r.co_changes
ORDER BY r.strength DESC
```

**Explanation**: Shows which files change most frequently with a specific file.

---

## Search & Filtering

### Search by Name (Fuzzy)

```cypher
MATCH (s:Symbol) 
WHERE s.name CONTAINS 'auth'
RETURN s.name, s.kind, s.file_path
ORDER BY s.name
LIMIT 20
```

**Explanation**: Simple text search for symbols containing 'auth'.

---

### Find Exportable Functions

```cypher
MATCH (f:Function) 
WHERE f.is_exported = true
RETURN f.name, f.file_path
ORDER BY f.name
```

**Explanation**: Finds functions that are exported (in `__all__` or exported).

---

### Find Decorated Functions

```cypher
MATCH (s:Symbol) 
WHERE s.decorators IS NOT NULL AND s.decorators <> '[]'
RETURN s.name, s.decorators, s.file_path
LIMIT 20
```

**Explanation**: Finds functions with decorators (useful for finding routes, handlers, etc.)

---

### Complex Functions (Many Parameters)

```cypher
MATCH (f:Function)
WHERE f.signature CONTAINS ','
WITH f, length(f.signature) - length(replace(f.signature, ',', '')) + 1 as param_count
WHERE param_count > 5
RETURN f.name, f.signature, param_count
ORDER BY param_count DESC
LIMIT 10
```

**Explanation**: Finds functions with many parameters - potential candidates for refactoring.

---

## Advanced Queries

### Full Code Context for a Symbol

```cypher
MATCH (s:Symbol {name: 'validate_user'})
RETURN s {
  .name, 
  .kind, 
  .file_path, 
  .start_line, 
  .end_line, 
  .content,
  .signature,
  .is_dead,
  .is_entry_point,
  .is_exported
}
```

**Explanation**: Returns all properties of a specific symbol.

---

### Analyze Module Dependencies

```cypher
MATCH (f:File)-[:IMPORTS]->(dep:File)
WITH f, collect(dep.name) as dependencies
WHERE size(dependencies) > 10
RETURN f.name, size(dependencies) as dep_count, dependencies
ORDER BY dep_count DESC
LIMIT 10
```

**Explanation**: Finds files with the most dependencies - high coupling indicators.

---

### Find Circular Dependencies

```cypher
MATCH path = (a:File)-[:IMPORTS*]->(a)
WHERE length(path) > 1
RETURN a.name as file, length(path) as cycle_length
LIMIT 10
```

**Explanation**: Detects circular import dependencies between files.

---

## Tips for Using KùzuExplorer

1. **Visualization**: Click the graph icon to see results as a node-link diagram
2. **Table View**: Click the table icon for spreadsheet-like results
3. **JSON View**: Click the `< >` icon for raw JSON output
4. **Schema Panel**: Use the schema panel to explore available node properties
5. **Query History**: Your query history is preserved in the session

---

## Related Documentation

- [Axon README](../../README.md)
- [Kùzu Documentation](https://docs.kuzudb.com)
- [KùzuExplorer Guide](https://docs.kuzudb.com/visualization/kuzu-explorer)
