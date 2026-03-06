
# **Objective**

Refactor into a class-based structure. The goal is to improve readability and maintainability by encapsulating state while ensuring **zero changes in functional behavior**.

## **Refactoring Rules**

### **1. Class Architecture**

- **Encapsulation**: Create a class named `[CLASS_NAME]`.
- **State Management**: Convert long-lived local variables (e.g., `KnowledgeGraph`, `seen` sets, lookup indexes) into private instance variables (e.g., `self._graph`).
- **Initialization**: Pass dependencies and raw data (like `parse_data`) into the [**init**](file:///Users/madesuryawan/Documents/Source_Codes/axon/src/axon/core/ingestion/calls.py#199-212) method.
- **Entry Point**: Provide a single public method (e.g., `process_X()`) that orchestrates the execution.

### **2. Logical Decomposition**

- **Private Helpers**: Break down the main loop into small, focused private methods (prefixed with `_`).
- **Explicit Context**: When a method depends on a specific ID or state calculated in a loop (like [source_id](file:///Users/madesuryawan/Documents/Source_Codes/axon/src/axon/core/ingestion/calls.py#494-513)), pass it explicitly rather than relying purely on instance state to avoid temporal coupling bugs.
- **Blocklists**: Keep constants like blocklists inside the class as private class-level attributes.

### **3. Strict Behavioral Parity**

- **Logic Mapping**: Every branch, confidence score, and proximity heuristic must match the original implementation exactly.
- **Line Numbers**: Ensure line-number-based lookups and ID generation are preserved.
- **Heuristics**: If the original code uses a proximity heuristic (e.g., "shortest path length"), ensure the refactored loop correctly compares all candidates before returning.

## **Implementation Steps for the Agent**

1. **Research**: Read the original implementation and identify all state variables and logic branches.
2. **Skeleton**: Create the new class structure in the target file.
3. **Port Logic**: Move the logic into the new private methods, paying close attention to state transitions.
4. **Clean up**: Remove the old functional implementation and update imports.
5. **Verify**: Compare the logic of the new class against the previous git commit to ensure 100% equivalence.

---

**Example Template for the Agent:**
"Refactor the `process_imports` function in [src/axon/core/ingestion/imports.py](file:///Users/madesuryawan/Documents/Source_Codes/axon/src/axon/core/ingestion/imports.py) into an `Imports` class. Follow the Axon Style: encapsulate the graph and index as state, but ensure the resolution heuristic for cross-file imports remains identical to the original implementation."
