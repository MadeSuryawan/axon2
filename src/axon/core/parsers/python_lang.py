"""
Python language parser using tree-sitter.

This module provides a parser that extracts functions, classes, methods,
imports, calls, type annotations, and inheritance relationships from
Python source code using the tree-sitter parsing library.

Design Decisions:
- Uses tree-sitter's incremental parsing for efficient AST traversal
- Separates call extraction from symbol extraction to avoid double-counting
- Builtin types are filtered out to reduce noise in type reference analysis
- Decorator names are captured to support metadata extraction

Key Responsibilities:
- Parse Python source into an Abstract Syntax Tree (AST)
- Extract symbol definitions (functions, classes, methods)
- Extract import statements and module references
- Extract function/method calls and exception handlers
- Extract type annotations and inheritance relationships
"""

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from axon.core.parsers.base import (
    CallInfo,
    ImportInfo,
    LanguageParser,
    ParseResult,
    SymbolInfo,
    TypeRef,
)


class PythonParser(LanguageParser):
    """
    Parses Python source code using tree-sitter.

    This parser extracts structured information from Python source files
    including symbol definitions, imports, calls, type annotations, and
    inheritance relationships. The extracted data is used for code analysis,
    dependency tracking, and knowledge graph construction.

    Attributes:
        PY_LANGUAGE: The tree-sitter language instance for Python.
        _BUILTIN_TYPES: Frozenset of Python builtin types to filter out
            from type reference analysis.

    Example:
        >>> parser = PythonParser()
        >>> result = parser.parse("def hello() -> str: pass", "hello.py")
        >>> result.symbols[0].name
        'hello'
    """

    # Language instance - shared across all parser instances for efficiency
    # Using class variable avoids repeated language initialization
    PY_LANGUAGE = Language(tspython.language())

    # Builtin types that are commonly used but not useful for type analysis.
    # These are filtered out to reduce noise in the extracted type references.
    _BUILTIN_TYPES: frozenset[str] = frozenset(
        {
            "str",
            "int",
            "float",
            "bool",
            "None",
            "list",
            "dict",
            "set",
            "tuple",
            "Any",
            "Optional",
            "bytes",
            "complex",
            "object",
            "type",
        },
    )

    def __init__(self) -> None:
        """Initialize the Python parser with a tree-sitter parser instance."""
        self._parser = Parser(self.PY_LANGUAGE)

    def parse(self, content: str, file_path: str) -> ParseResult:
        """
        Parse Python source and return structured information.

        This is the main entry point for parsing a Python source file.
        It performs two phases:
        1. Walk the AST to extract symbol definitions, imports, and annotations
        2. Recursively extract function/method calls at each scope boundary

        The two-phase approach avoids double-counting calls that would occur
        if we extracted them during the initial AST walk.

        Args:
            content: The Python source code as a string.
            file_path: The path to the source file (used for context).

        Returns:
            ParseResult containing all extracted symbols, imports, calls,
            type references, heritage relationships, and exports.
        """
        tree = self._parser.parse(bytes(content, "utf8"))
        result = ParseResult()
        root = tree.root_node

        # Phase 1: Extract definitions, imports, and annotations
        # Pass empty class_name since we're at module level
        self._walk(root, content, result, class_name="")

        # Phase 2: Extract module-level and nested calls
        # This is done separately to avoid double-counting
        self._extract_calls_recursive(root, result)

        return result

    # ---------------------------------------------------------------------------
    # Core AST Traversal
    # ---------------------------------------------------------------------------

    def _walk(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """
        Recursively walk the AST to extract definitions and annotations.

        This method traverses the AST tree, identifying and extracting
        different types of nodes. Call extraction is intentionally handled
        separately via _extract_calls_recursive at each scope boundary
        (module, class, function) to avoid double-counting.

        Args:
            node: The current AST node to process.
            content: The source content for text extraction.
            result: The parse result to populate.
            class_name: The enclosing class name (empty string for module-level).
        """
        for child in node.children:
            match child.type:
                case "function_definition":
                    self._extract_function(child, content, result, class_name)
                case "class_definition":
                    self._extract_class(child, content, result)
                case "import_statement":
                    self._extract_import(child, result)
                case "import_from_statement":
                    self._extract_import_from(child, result)
                case "decorated_definition":
                    self._extract_decorated(child, content, result, class_name)
                case "expression_statement":
                    # Only extract variable annotations here; calls are
                    # handled by the scope-level _extract_calls_recursive.
                    self._extract_annotations_from_expression(child, result)
                case _:
                    # Continue recursion for other node types
                    self._walk(child, content, result, class_name)

    # ---------------------------------------------------------------------------
    # Function and Method Extraction
    # ---------------------------------------------------------------------------

    def _extract_function(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """
        Extract a function or method definition.

        Extracts all relevant information from a function definition node
        including its name, signature, parameters, return type, and body.
        Nested functions and classes within the function body are also
        processed recursively.

        Args:
            node: The function_definition AST node.
            content: The source content for text extraction.
            result: The parse result to populate.
            class_name: The owning class name (empty for standalone functions).
        """
        # Extract function name - early return if missing
        name_node = node.child_by_field_name("name")
        if not name_node or not name_node.text:
            return

        name = name_node.text.decode("utf8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        node_content = content[node.start_byte : node.end_byte]

        # Determine kind: methods have an enclosing class, standalone are functions
        kind = "method" if class_name else "function"
        signature = self._build_signature(node, content)

        # Add the symbol to results
        result.symbols.append(
            SymbolInfo(
                name=name,
                kind=kind,
                start_line=start_line,
                end_line=end_line,
                content=node_content,
                signature=signature,
                class_name=class_name,
            ),
        )

        # Extract parameter type annotations
        self._extract_param_types(node, result)

        # Extract return type annotation if present and not a builtin
        return_type = node.child_by_field_name("return_type")
        if return_type is not None:
            type_name = self._extract_type_name(return_type)
            if type_name and type_name not in self._BUILTIN_TYPES:
                result.type_refs.append(
                    TypeRef(
                        name=type_name,
                        kind="return",
                        line=return_type.start_point[0] + 1,
                    ),
                )

        # Process nested definitions within the function body.
        # Nested functions/classes inside a function are not methods,
        # so we pass class_name="" to keep them as standalone symbols.
        body = node.child_by_field_name("body")
        if body is not None:
            self._walk(body, content, result, class_name="")

    def _build_signature(self, func_node: Node, content: str) -> str:
        """
        Build a human-readable signature string for a function.

        Constructs a signature in the form:
        - "def func_name(param1: type1, param2: type2) -> return_type"

        Returns an empty string if required nodes are missing.

        Args:
            func_node: The function_definition AST node.
            content: The source content (unused but kept for API consistency).

        Returns:
            A string representation of the function signature.
        """
        # Performance: Use early returns to avoid unnecessary processing
        name_node = func_node.child_by_field_name("name")
        params_node = func_node.child_by_field_name("parameters")
        return_type = func_node.child_by_field_name("return_type")

        if name_node is None or params_node is None:
            return ""

        if not name_node.text or not params_node.text:
            return ""

        name = name_node.text.decode("utf8")
        params = params_node.text.decode("utf8")
        sig = f"def {name}{params}"

        if return_type is not None:
            sig += f" -> {return_type.text.decode('utf8')}"

        return sig

    # ---------------------------------------------------------------------------
    # Parameter Type Extraction
    # ---------------------------------------------------------------------------

    def _extract_param_types(self, func_node: Node, result: ParseResult) -> None:
        """
        Extract type annotations from function parameters.

        Processes both typed_parameter and typed_default_parameter nodes
        to extract type information for each parameter.

        Args:
            func_node: The function_definition AST node.
            result: The parse result to populate with type references.
        """
        params_node = func_node.child_by_field_name("parameters")
        if params_node is None:
            return

        for param in params_node.children:
            if param.type in ("typed_parameter", "typed_default_parameter"):
                self._extract_typed_param(param, result)

    def _extract_typed_param(self, param_node: Node, result: ParseResult) -> None:
        """
        Extract a single typed parameter's type reference.

        Extracts the parameter name and its type annotation, creating
        a TypeRef entry for non-builtin types.

        Args:
            param_node: A typed_parameter or typed_default_parameter node.
            result: The parse result to populate with type reference.
        """
        # Find the parameter name - iterate through children
        param_name = ""
        for child in param_node.children:
            if child.type == "identifier":
                if not (text := child.text):
                    continue
                param_name = text.decode("utf8")
                break

        # Extract the type annotation
        type_node = param_node.child_by_field_name("type")
        if type_node is None:
            return

        type_name = self._extract_type_name(type_node)
        if type_name and type_name not in self._BUILTIN_TYPES:
            result.type_refs.append(
                TypeRef(
                    name=type_name,
                    kind="param",
                    line=type_node.start_point[0] + 1,
                    param_name=param_name,
                ),
            )

    # ---------------------------------------------------------------------------
    # Type Name Extraction
    # ---------------------------------------------------------------------------

    @staticmethod
    def _extract_type_name(type_node: Node) -> str:
        """
        Extract the primary type name from a type annotation node.

        Handles various type annotation patterns:
        - Simple types: "User" -> "User"
        - Generic types: "list[User]" -> "list"
        - Optional types: "Optional[User]" -> "Optional"
        - Complex types: Falls back to first identifier found

        Args:
            type_node: The type annotation AST node.

        Returns:
            The primary type name as a string, or empty string if not found.
        """
        result = ""

        # Handle "type" wrapper node (e.g., type[User])
        if type_node.type == "type" and type_node.children:
            inner = type_node.children[0]
            if inner.type == "identifier":
                text = inner.text
                if text:
                    result = text.decode("utf8")
            elif inner.type == "generic_type":
                # e.g., Optional[User] — extract the generic type name
                result = PythonParser._find_first_identifier(inner)
            else:
                # Fallback: search for first identifier anywhere in the node
                result = PythonParser._find_first_identifier(inner)
        # Handle direct identifier type (e.g., User)
        elif type_node.type == "identifier":
            text = type_node.text
            if text:
                result = text.decode("utf8")
        # Fallback: DFS for first identifier
        else:
            result = PythonParser._find_first_identifier(type_node)

        return result

    @staticmethod
    def _find_first_identifier(node: Node) -> str:
        """
        Depth-first search for the first identifier node.

        Used as a fallback when type extraction fails to find a direct match.
        This handles complex nested type expressions.

        Args:
            node: The AST node to search within.

        Returns:
            The first identifier's text, or empty string if none found.
        """
        if node.type == "identifier":
            text = node.text
            if not text:
                return ""
            return text.decode("utf8")
        for child in node.children:
            found = PythonParser._find_first_identifier(child)
            if found:
                return found
        return ""

    # ---------------------------------------------------------------------------
    # Decorator Extraction
    # ---------------------------------------------------------------------------

    def _extract_decorated(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """
        Extract a decorated function or class, capturing decorator names.

        Tree-sitter represents decorated definitions with a decorated_definition
        node containing one or more decorator nodes followed by the actual
        definition (function_definition or class_definition).

        Args:
            node: The decorated_definition AST node.
            content: The source content for nested extraction.
            result: The parse result to populate.
            class_name: The enclosing class name.
        """
        decorators: list[str] = []
        definition_node: Node | None = None

        # Parse children to find decorators and the actual definition
        for child in node.children:
            if child.type == "decorator":
                dec_name = self._extract_decorator_name(child)
                if dec_name:
                    decorators.append(dec_name)
            elif child.type in ("function_definition", "class_definition"):
                definition_node = child

        if definition_node is None:
            return

        # Track symbol count before extraction to associate decorators
        count_before = len(result.symbols)

        # Extract the underlying definition
        if definition_node.type == "function_definition":
            self._extract_function(definition_node, content, result, class_name)
        else:
            self._extract_class(definition_node, content, result)

        # Attach decorators to the newly added symbol
        if count_before < len(result.symbols):
            result.symbols[count_before].decorators = decorators

    def _extract_decorator_name(self, decorator_node: Node) -> str:
        """
        Extract the dotted name from a decorator node.

        Handles three common decorator forms:
        - Simple: @staticmethod -> "staticmethod"
        - Attribute: @app.route -> "app.route"
        - Call: @server.list_tools() -> "server.list_tools"

        Args:
            decorator_node: The decorator AST node.

        Returns:
            The decorator name as a string, or empty string if not extractable.
        """
        for child in decorator_node.children:
            if not child.text:
                continue
            if child.type == "identifier":
                return child.text.decode("utf8")
            if child.type == "attribute":
                return child.text.decode("utf8")
            if child.type == "call":
                func = child.child_by_field_name("function")
                if func and (text := func.text):
                    return text.decode("utf8")
        return ""

    # ---------------------------------------------------------------------------
    # Class Extraction
    # ---------------------------------------------------------------------------

    def _extract_class(
        self,
        node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """
        Extract a class definition and its contents.

        Processes class definitions including:
        - Class name and location
        - Superclass relationships (heritage)
        - Body contents (methods, nested classes)

        Args:
            node: The class_definition AST node.
            content: The source content for text extraction.
            result: The parse result to populate.
        """
        # Extract class name - early return if missing
        name_node = node.child_by_field_name("name")
        if name_node is None or not name_node.text:
            return

        class_name = name_node.text.decode("utf8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        node_content = content[node.start_byte : node.end_byte]

        # Add class symbol to results
        result.symbols.append(
            SymbolInfo(
                name=class_name,
                kind="class",
                start_line=start_line,
                end_line=end_line,
                content=node_content,
            ),
        )

        # Extract superclass relationships
        superclasses = node.child_by_field_name("superclasses")
        if superclasses is not None:
            for child in superclasses.children:
                if child.is_named and child.type == "identifier":
                    if not (text := child.text):
                        continue
                    parent_name = text.decode("utf8")
                    result.heritage.append((class_name, "extends", parent_name))

        # Process body contents with class_name context
        body = node.child_by_field_name("body")
        if body is not None:
            self._walk(body, content, result, class_name=class_name)

    # ---------------------------------------------------------------------------
    # Import Extraction
    # ---------------------------------------------------------------------------

    def _extract_import(self, node: Node, result: ParseResult) -> None:
        """
        Extract a plain ``import X`` statement.

        Handles various import forms:
        - Simple: import os
        - Dotted: import os.path
        - Aliased: import os as operating_system
        - Multiple: import os, sys

        For dotted imports like "os.path", extracts "path" as the local name
        since that's what's available in the local namespace.

        Args:
            node: The import_statement AST node.
            result: The parse result to populate.
        """
        # import_statement children: "import", dotted_name [, ",", dotted_name ...]
        for child in node.children:
            if child.type == "dotted_name":
                if not (text := child.text):
                    continue
                module = text.decode("utf8")
                # For "import os.path" the imported name is "path"
                parts = module.split(".")
                result.imports.append(
                    ImportInfo(
                        module=module,
                        names=[parts[-1]],
                    ),
                )
            elif child.type == "aliased_import":
                # Handle "import X as alias" form
                alias_node = child.child_by_field_name("alias")
                if not (name_node := child.child_by_field_name("name")):
                    continue
                if not (text := name_node.text):
                    continue
                module = text.decode("utf8")
                parts = module.split(".")
                alias = ""
                if alias_node and (text := alias_node.text):
                    alias = text.decode("utf8")
                result.imports.append(
                    ImportInfo(
                        module=module,
                        names=[parts[-1]],
                        alias=alias,
                    ),
                )

    def _extract_import_from(self, node: Node, result: ParseResult) -> None:
        """
        Extract a ``from X import Y`` statement.

        Handles both absolute and relative imports:
        - from os.path import join
        - from . import utils
        - from ..models import User

        Args:
            node: The import_from_statement AST node.
            result: The parse result to populate.
        """
        module_name_node = node.child_by_field_name("module_name")
        if module_name_node is None:
            return

        is_relative = module_name_node.type == "relative_import"
        if not (text := module_name_node.text):
            return
        module = text.decode("utf8")

        # Extract imported names - they appear after the "import" keyword
        names: list[str] = []
        past_import = False
        for child in node.children:
            if child.type == "import":
                past_import = True
                continue
            if past_import and child.type == "dotted_name":
                if not (text := child.text):
                    continue
                names.append(text.decode("utf8"))

        result.imports.append(
            ImportInfo(
                module=module,
                names=names,
                is_relative=is_relative,
            ),
        )

    # ---------------------------------------------------------------------------
    # Variable Annotation and Export Extraction
    # ---------------------------------------------------------------------------

    def _extract_annotations_from_expression(
        self,
        node: Node,
        result: ParseResult,
    ) -> None:
        """
        Extract variable annotations and __all__ from an expression_statement.

        Handles two cases:
        - Type annotations: "x: int = 5"
        - Export declarations: "__all__ = ['foo', 'bar']"

        Args:
            node: The expression_statement AST node.
            result: The parse result to populate.
        """
        for child in node.children:
            if child.type == "assignment":
                self._try_extract_variable_annotation(child, result)
                self._try_extract_all_exports(child, result)

    def _try_extract_variable_annotation(
        self,
        assignment_node: Node,
        result: ParseResult,
    ) -> None:
        """
        Extract a type reference from a variable annotation if present.

        Handles annotated assignments like "name: str = 'default'".
        Only extracts non-builtin types.

        Args:
            assignment_node: An assignment AST node.
            result: The parse result to populate.
        """
        type_node = assignment_node.child_by_field_name("type")
        if type_node is None:
            return

        type_name = self._extract_type_name(type_node)
        if type_name and type_name not in self._BUILTIN_TYPES:
            result.type_refs.append(
                TypeRef(
                    name=type_name,
                    kind="variable",
                    line=type_node.start_point[0] + 1,
                ),
            )

    @staticmethod
    def _try_extract_all_exports(assignment_node: Node, result: ParseResult) -> None:
        """
        Extract names from ``__all__ = [...]`` or ``__all__ = (...)`` assignments.

        The __all__ variable explicitly declares the public API of a module.
        This extraction supports tooling that needs to understand what's
        intentionally exported vs. what's just implementation detail.

        Args:
            assignment_node: An assignment AST node.
            result: The parse result to populate with export names.
        """
        left = assignment_node.child_by_field_name("left")
        right = assignment_node.child_by_field_name("right")

        if left is None or right is None:
            return

        if not (text := left.text):
            return

        # Check this is an __all__ assignment
        if left.type != "identifier" or text.decode("utf8") != "__all__":
            return

        # __all__ must be assigned a list or tuple
        if right.type not in ("list", "tuple"):
            return

        # Extract string contents from each element
        for child in right.children:
            if child.type == "string":
                if not (text := child.text):
                    continue
                text = text.decode("utf8")
                # Strip surrounding quotes (single, double, or triple)
                for quote in ('"""', "'''", '"', "'"):
                    if text.startswith(quote) and text.endswith(quote):
                        text = text[len(quote) : -len(quote)]
                        break
                result.exports.append(text)

    # ---------------------------------------------------------------------------
    # Call Extraction - Recursive Phase
    # ---------------------------------------------------------------------------

    def _extract_calls_recursive(self, node: Node, result: ParseResult) -> None:
        """
        Recursively find and extract all call nodes and exception references.

        This method traverses the AST at each scope level (module, class, function)
        to extract:
        - Function/method calls (call nodes)
        - Exception handlers (except_clause nodes)
        - Raise statements (raise_statement nodes)

        The recursive approach ensures we capture calls at all nesting levels.

        Args:
            node: The current AST node to process.
            result: The parse result to populate.
        """
        # Base case: found a call node - extract and recurse into its children
        if node.type == "call":
            self._extract_call(node, result)
            for child in node.children:
                self._extract_calls_recursive(child, result)
            return

        # Handle exception handlers: except SomeError:
        if node.type == "except_clause":
            self._extract_exception_handler(node, result)

        # Handle raise statements: raise SomeError
        if node.type == "raise_statement":
            self._extract_raise_statement(node, result)

        # Continue recursion for all other node types
        for child in node.children:
            self._extract_calls_recursive(child, result)

    def _extract_exception_handler(self, node: Node, result: ParseResult) -> None:
        """
        Extract exception types from an except clause.

        Handles various exception handler forms:
        - Single: except Error:
        - Multiple: except (ErrorA, ErrorB):
        - Aliased: except Error as e:
        - Multiple with alias: except (ErrorA, ErrorB) as e:

        Args:
            node: The except_clause AST node.
            result: The parse result to populate.
        """
        for child in node.children:
            if child.type == "identifier":
                # Single exception: except Error:
                if not (text := child.text):
                    continue
                result.calls.append(
                    CallInfo(
                        name=text.decode("utf8"),
                        line=child.start_point[0] + 1,
                    ),
                )
            elif child.type == "tuple":
                # Multiple exceptions: except (ErrorA, ErrorB):
                for elem in child.children:
                    if elem.type != "identifier":
                        continue
                    if not (text := elem.text):
                        continue
                    result.calls.append(
                        CallInfo(
                            name=text.decode("utf8"),
                            line=elem.start_point[0] + 1,
                        ),
                    )
            elif child.type == "as_pattern":
                # Aliased exception: except Error as e
                self._extract_aliased_exception(child, result)

    def _extract_aliased_exception(self, node: Node, result: ParseResult) -> None:
        """
        Extract exception types from an aliased except pattern.

        Handles the "as" alias pattern in exception handlers:
        - except Error as e:
        - except (ErrorA, ErrorB) as e:

        Args:
            node: The as_pattern AST node.
            result: The parse result to populate.
        """
        for sub in node.children:
            if sub.type == "identifier":
                # Single aliased exception
                if not (text := sub.text):
                    continue
                result.calls.append(
                    CallInfo(
                        name=text.decode("utf8"),
                        line=sub.start_point[0] + 1,
                    ),
                )
                break
            if sub.type == "tuple":
                # Multiple aliased exceptions
                for elem in sub.children:
                    if elem.type == "identifier":
                        if not (text := elem.text):
                            continue
                        result.calls.append(
                            CallInfo(
                                name=text.decode("utf8"),
                                line=elem.start_point[0] + 1,
                            ),
                        )
                break

    def _extract_raise_statement(self, node: Node, result: ParseResult) -> None:
        """
        Extract exception class references from a raise statement.

        Handles "raise SomeError" (without parentheses) which references
        the exception class rather than instantiating it.

        Args:
            node: The raise_statement AST node.
            result: The parse result to populate.
        """
        for child in node.children:
            if child.type != "identifier":
                continue
            text = ""
            if not (text := child.text):
                continue
            result.calls.append(
                CallInfo(
                    name=text.decode("utf8"),
                    line=child.start_point[0] + 1,
                ),
            )

    # ---------------------------------------------------------------------------
    # Individual Call Extraction
    # ---------------------------------------------------------------------------

    def _extract_call(self, call_node: Node, result: ParseResult) -> None:
        """
        Extract a single call node into a CallInfo.

        Handles two main cases:
        - Simple function calls: func() - identifier type
        - Method calls: obj.method() - attribute type

        Args:
            call_node: The call AST node.
            result: The parse result to populate.
        """
        # Try to get the function node from the "function" field
        func_node = call_node.child_by_field_name("function")

        # Fallback: find first named child if field is missing
        if func_node is None:
            func_node = self._find_fallback_function_node(call_node)

        if func_node is None:
            return

        line = call_node.start_point[0] + 1
        arguments = self._extract_identifier_arguments(call_node)

        # Dispatch based on function node type
        if func_node.type == "identifier":
            self._extract_simple_call(func_node, line, arguments, result)
        elif func_node.type == "attribute":
            self._extract_method_call(func_node, line, arguments, result)

    def _find_fallback_function_node(self, call_node: Node) -> Node | None:
        """
        Find the function node when the 'function' field is missing.

        Tree-sitter doesn't always populate the 'function' field, so this
        provides a fallback by finding the first named child.

        Args:
            call_node: The call AST node.

        Returns:
            The function Node if found, None otherwise.
        """
        for child in call_node.children:
            if child.is_named:
                return child
        return None

    def _extract_simple_call(
        self,
        func_node: Node,
        line: int,
        arguments: list[str],
        result: ParseResult,
    ) -> None:
        """
        Extract a simple function call (identifier-based).

        Handles direct function calls like:
        - foo()
        - bar(arg1, arg2)

        Args:
            func_node: The identifier AST node representing the function.
            line: The line number of the call.
            arguments: List of identifier argument names.
            result: The parse result to populate.
        """
        text = func_node.text
        if not text:
            return
        result.calls.append(
            CallInfo(
                name=text.decode("utf8"),
                line=line,
                arguments=arguments,
            ),
        )

    def _extract_method_call(
        self,
        func_node: Node,
        line: int,
        arguments: list[str],
        result: ParseResult,
    ) -> None:
        """
        Extract a method/attribute call.

        Handles method calls and attribute access chains like:
        - obj.method()
        - self.logger.info()
        - obj.method1().method2()

        Args:
            func_node: The attribute AST node representing the method.
            line: The line number of the call.
            arguments: List of identifier argument names.
            result: The parse result to populate.
        """
        name, receiver = self._extract_attribute_call(func_node)
        result.calls.append(
            CallInfo(
                name=name,
                line=line,
                receiver=receiver,
                arguments=arguments,
            ),
        )

    def _extract_attribute_call(self, attr_node: Node) -> tuple[str, str]:
        """
        Extract (method_name, receiver) from an attribute node.

        For chained calls like "obj.method1().method2()", the outer call's
        function is "attribute(call(...), 'method2')". We extract "method2"
        as the name and the root of the chain as the receiver.

        Examples:
            - obj.method() -> ("method", "obj")
            - self.logger.info() -> ("info", "self")
            - get_user().save() -> ("save", "get_user")

        Args:
            attr_node: The attribute AST node.

        Returns:
            A tuple of (method_name, receiver).
        """
        # Extract method name - find rightmost identifier in reverse order
        method_name = ""
        for child in reversed(attr_node.children):
            if child.type != "identifier":
                continue
            text = child.text
            if not text:
                continue
            method_name = text.decode("utf8")
            break

        # Extract receiver (the object the method is called on)
        receiver = ""
        obj_node = attr_node.children[0] if attr_node.children else None
        if obj_node is not None:
            if obj_node.type == "identifier":
                text = obj_node.text
                if not text:
                    return method_name, ""
                receiver = text.decode("utf8")
            elif obj_node.type == "attribute":
                # Nested attribute: self.logger.info -> receiver is "self"
                receiver = self._root_identifier(obj_node)
            elif obj_node.type == "call":
                # Chained call: get_user().save() -> receiver is "get_user"
                receiver = self._root_identifier(obj_node)

        return method_name, receiver

    @staticmethod
    def _extract_identifier_arguments(call_node: Node) -> list[str]:
        """
        Extract bare identifier arguments from a call node.

        Returns names of arguments that are plain identifiers (not literals,
        calls, or attribute accesses). These are typically callback references
        like "map(transform, items)" or "Depends(get_db)".

        Args:
            call_node: The call AST node.

        Returns:
            List of identifier argument names.
        """
        args_node = call_node.child_by_field_name("arguments")
        if args_node is None:
            return []

        identifiers: list[str] = []
        for child in args_node.children:
            if child.type == "identifier":
                text = child.text
                if not text:
                    continue
                identifiers.append(text.decode("utf8"))
            elif child.type == "keyword_argument":
                # Extract identifier from keyword argument value
                value_node = child.child_by_field_name("value")
                if value_node is not None and value_node.type == "identifier":
                    text = value_node.text
                    if not text:
                        continue
                    identifiers.append(text.decode("utf8"))
        return identifiers

    def _root_identifier(self, node: Node) -> str:
        """
        Walk down into the leftmost identifier of an expression.

        This is used to find the root object in chained calls or
        attribute access chains. For example:
        - self.logger.info -> "self"
        - get_user().save -> "get_user"

        Args:
            node: The AST node to traverse.

        Returns:
            The root identifier name, or empty string if not found.
        """
        current: Node | None = node
        while current is not None:
            if current.type == "identifier":
                text = current.text
                if not text:
                    break
                return text.decode("utf8")
            if current.children:
                current = current.children[0]
            else:
                break
        return ""
