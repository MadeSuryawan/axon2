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

from dataclasses import dataclass

from tree_sitter import Language, Node, Parser
from tree_sitter_python import language

from axon.core.parsers.base import (
    CallInfo,
    ImportInfo,
    LanguageParser,
    ParseResult,
    SymbolInfo,
    TypeRef,
)


class PyTypeExtractor:
    """
    Extracts type annotations from Python source code.

    Handles parameter types, return types, variable annotations,
    and complex type expressions.
    """

    # Builtin types that are commonly used but not useful for type analysis.
    _BUILTIN_TYPES = frozenset(
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

    @staticmethod
    def extract_type_name(type_node: Node) -> str:
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
        node_type = type_node.type

        # Handle "type" wrapper node (e.g., type[User])
        if node_type == "type" and type_node.children:
            inner = type_node.children[0]
            if inner.type == "identifier":
                if not (text := inner.text):
                    return ""
                result = text.decode("utf8")
            elif inner.type == "generic_type":
                # e.g., Optional[User] — extract the generic type name
                result = PyTypeExtractor._find_first_identifier(inner)
            else:
                # Fallback: search for first identifier anywhere in the node
                result = PyTypeExtractor._find_first_identifier(inner)
        # Handle direct identifier type (e.g., User)
        elif node_type == "identifier":
            if not (text := type_node.text):
                return ""
            result = text.decode("utf8")
        # Fallback: DFS for first identifier
        else:
            result = PyTypeExtractor._find_first_identifier(type_node)

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
            if not (text := node.text):
                return ""
            return text.decode("utf8")

        for child in node.children:
            if not (found := PyTypeExtractor._find_first_identifier(child)):
                continue
            return found
        return ""

    @classmethod
    def is_builtin_type(cls, type_name: str) -> bool:
        """Check if the given type name is a builtin type to filter out."""
        return type_name in cls._BUILTIN_TYPES

    def extract_param_types(
        self,
        func_node: Node,
        result: ParseResult,
    ) -> None:
        """
        Extract type annotations from function parameters.

        Processes both typed_parameter and typed_default_parameter nodes
        to extract type information for each parameter.

        Args:
            func_node: The function_definition AST node.
            result: The parse result to populate with type references.
        """
        if not (params_node := func_node.child_by_field_name("parameters")):
            return

        for param in params_node.children:
            if param.type not in ("typed_parameter", "typed_default_parameter"):
                continue
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
            if child.type != "identifier":
                continue
            if not (text := child.text):
                continue
            param_name = text.decode("utf8")
            break

        # Extract the type annotation
        if not (type_node := param_node.child_by_field_name("type")):
            return

        if not (type_name := self.extract_type_name(type_node)) or self.is_builtin_type(type_name):
            return

        result.type_refs.append(
            TypeRef(
                name=type_name,
                kind="param",
                line=type_node.start_point[0] + 1,
                param_name=param_name,
            ),
        )

    @classmethod
    def extract_return_type(cls, func_node: Node, result: ParseResult) -> None:
        """
        Extract return type annotation if present and not a builtin.

        Args:
            func_node: The function_definition AST node.
            result: The parse result to populate with type references.
        """
        if not (
            (return_type := func_node.child_by_field_name("return_type"))
            and (type_name := cls.extract_type_name(return_type))
            and not cls.is_builtin_type(type_name)
        ):
            return

        result.type_refs.append(
            TypeRef(
                name=type_name,
                kind="return",
                line=return_type.start_point[0] + 1,
            ),
        )

    def extract_variable_annotation(
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
        if not (type_node := assignment_node.child_by_field_name("type")):
            return

        if not (type_name := self.extract_type_name(type_node)) or self.is_builtin_type(type_name):
            return

        result.type_refs.append(
            TypeRef(
                name=type_name,
                kind="variable",
                line=type_node.start_point[0] + 1,
            ),
        )


class PyDecoratorExtractor:
    """
    Extracts decorator information from decorated definitions.

    Handles various decorator forms:
    - Simple: @staticmethod
    - Attribute: @app.route
    - Call: @server.list_tools()
    """

    def extract_decorator_name(self, decorator_node: Node) -> str:
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
            child_type = child.type
            if child_type == "identifier":
                return child.text.decode("utf8")
            if child_type == "attribute":
                return child.text.decode("utf8")
            if (
                child_type == "call"
                and (func := child.child_by_field_name("function"))
                and (text := func.text)
            ):
                return text.decode("utf8")
        return ""

    def extract_decorators(
        self,
        node: Node,
    ) -> tuple[list[str], Node | None]:
        """
        Extract all decorator names from a decorated_definition node.

        Args:
            node: The decorated_definition AST node.

        Returns:
            A tuple of (list of decorator names, the definition node).
        """
        decorators: list[str] = []
        definition_node: Node | None = None

        # Parse children to find decorators and the actual definition
        for child in node.children:
            if child.type == "decorator":
                dec_name = self.extract_decorator_name(child)
                if dec_name:
                    decorators.append(dec_name)
            elif child.type in ("function_definition", "class_definition"):
                definition_node = child

        return decorators, definition_node


class PyImportExtractor:
    """
    Extracts import statements from Python source code.

    Handles various import forms:
    - Simple: import os
    - Dotted: import os.path
    - Aliased: import os as operating_system
    - Multiple: import os, sys
    - From: from os.path import join
    - Relative: from . import utils
    """

    def extract_import(self, node: Node, result: ParseResult) -> None:
        """
        Extract a plain ``import X`` statement.

        Handles various import forms:
        - Simple: import os
        - Dotted: import os.path
        - Aliased: import os as operating_system
        - Multiple: import os, sys

        For dotted imports like "import os.path", extracts "path" as the local name
        since that's what's available in the local namespace.

        Args:
            node: The import_statement AST node.
            result: The parse result to populate.
        """
        imports = result.imports
        # import_statement children: "import", dotted_name [, ",", dotted_name ...]
        for child in node.children:
            child_type = child.type
            if child_type == "dotted_name":
                if not (text := child.text):
                    continue
                module = text.decode("utf8")
                # For "import os.path" the imported name is "path"
                parts = module.split(".")
                imports.append(
                    ImportInfo(
                        module=module,
                        names=[parts[-1]],
                    ),
                )
                continue
            if child_type == "aliased_import":
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
                imports.append(
                    ImportInfo(
                        module=module,
                        names=[parts[-1]],
                        alias=alias,
                    ),
                )

    def extract_import_from(self, node: Node, result: ParseResult) -> None:
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
        if not (module_name_node := node.child_by_field_name("module_name")):
            return

        if not (text := module_name_node.text):
            return

        result.imports.append(
            ImportInfo(
                module=text.decode("utf8"),
                names=self._extract_imported_names(node),
                is_relative=module_name_node.type == "relative_import",
            ),
        )

    def _extract_imported_names(self, node: Node) -> list[str]:
        """
        Extract imported names from an import_from_statement node.

        Args:
            node: The import_from_statement AST node.

        Returns:
            A list of imported names.
        """
        # Extract imported names - they appear after the "import" keyword
        names: list[str] = []
        past_import = False
        for child in node.children:
            if child.type == "import":
                past_import = True
                continue
            if not past_import:
                continue
            if not (text := child.text):
                continue
            if child.type in ("dotted_name", "identifier"):
                names.append(text.decode("utf8"))
            elif child.type == "import_as_names":
                names.extend(self._extract_childrens(child))
            elif child.type == "wildcard_import":
                names.append("*")
        return names

    def _extract_childrens(self, child: Node) -> list[str]:
        """
        Extract names from import_as_names node.

        Args:
            child: The import_as_names AST node.

        Returns:
            A list of imported names.
        """
        names: list[str] = []
        for sub in child.children:
            if sub.type in ("dotted_name", "identifier"):
                if not (sub_text := sub.text):
                    continue
                names.append(sub_text.decode("utf8"))
            elif sub.type == "aliased_import":
                if not (name_node := sub.child_by_field_name("name")):
                    continue
                if not (name_text := name_node.text):
                    continue
                names.append(name_text.decode("utf8"))
        return names


class PyFunctionExtractor:
    """
    Extracts function and method definitions from Python source code.

    Handles standalone functions, methods within classes, nested functions,
    and decorated definitions.
    """

    def __init__(self, type_extractor: PyTypeExtractor) -> None:
        """
        Initialize the function extractor.

        Args:
            type_extractor: The type extractor for parameter/return types.
        """
        self._type_extractor = type_extractor

    def build_signature(self, func_node: Node, content: str) -> str:
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
        if not (name_node := func_node.child_by_field_name("name")) or not (
            params_node := func_node.child_by_field_name("parameters")
        ):
            return ""

        if not (name_text := name_node.text) or not (params_text := params_node.text):
            return ""

        sig = f"def {name_text.decode('utf8')}{params_text.decode('utf8')}"

        if return_type := func_node.child_by_field_name("return_type"):
            if not (return_type_text := return_type.text):
                return sig
            sig += f" -> {return_type_text.decode('utf8')}"

        return sig

    def extract_function(
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
        if not (name_node := node.child_by_field_name("name")) or not (name_text := name_node.text):
            return

        # # Determine kind: methods have an enclosing class, standalone are functions
        # kind = "method" if class_name else "function"
        # signature = self.build_signature(node, content)

        # Add the symbol to results
        result.symbols.append(
            SymbolInfo(
                name=name_text.decode("utf8"),
                kind="method" if class_name else "function",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                content=content[node.start_byte : node.end_byte],
                signature=self.build_signature(node, content),
                class_name=class_name,
            ),
        )

        # Extract parameter type annotations
        self._type_extractor.extract_param_types(node, result)

        # Extract return type annotation if present and not a builtin
        self._type_extractor.extract_return_type(node, result)


class PyClassExtractor:
    """Extracts class definitions and inheritance relationships from Python source."""

    def __init__(self, type_extractor: PyTypeExtractor) -> None:
        """
        Initialize the class extractor.

        Args:
            type_extractor: The type extractor for type annotations.
        """
        self._type_extractor = type_extractor

    def extract_class(
        self,
        node: Node,
        content: str,
        result: ParseResult,
    ) -> str | None:
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

        Returns:
            The class name if extracted successfully, None otherwise.
        """
        # Extract class name - early return if missing
        if not (name_node := node.child_by_field_name("name")) or not (name_text := name_node.text):
            return None

        class_name = name_text.decode("utf8")

        # Add class symbol to results
        result.symbols.append(
            SymbolInfo(
                name=class_name,
                kind="class",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                content=content[node.start_byte : node.end_byte],
            ),
        )

        # Extract superclass relationships
        if not (superclasses := node.child_by_field_name("superclasses")):
            return class_name

        for child in superclasses.children:
            if not child.is_named:
                continue
            if not (text := child.text):
                continue
            if child.type in {"identifier", "attribute"}:
                parent_name = text.decode("utf8")
                result.heritage.append((class_name, "extends", parent_name))
            elif child.type == "subscript":
                # e.g. class Foo(Generic[T]): — capture "Generic"
                if not (base := child.child_by_field_name("value")):
                    continue
                if not (base_text := base.text):
                    continue
                parent_name = base_text.decode("utf8")
                result.heritage.append((class_name, "extends", parent_name))

        # Return the class name so the caller can process the body
        return class_name

    def extract_variable_annotation(
        self,
        assignment_node: Node,
        result: ParseResult,
    ) -> None:
        """
        Extract a type reference from a variable annotation if present.

        Args:
            assignment_node: An assignment AST node.
            result: The parse result to populate.
        """
        self._type_extractor.extract_variable_annotation(assignment_node, result)

    @staticmethod
    def extract_all_exports(assignment_node: Node, result: ParseResult) -> None:
        """
        Extract names from ``__all__ = [...]`` or ``__all__ = (...)`` assignments.

        The __all__ variable explicitly declares the public API of a module.
        This extraction supports tooling that needs to understand what's
        intentionally exported vs. what's just implementation detail.

        Args:
            assignment_node: An assignment AST node.
            result: The parse result to populate with export names.
        """
        if not (left := assignment_node.child_by_field_name("left")) or not (
            right := assignment_node.child_by_field_name("right")
        ):
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
            if child.type != "string":
                continue
            if not (text := child.text):
                continue
            text = text.decode("utf8")
            # Strip surrounding quotes (single, double, or triple)
            for quote in ('"""', "'''", '"', "'"):
                if not text.startswith(quote) or not text.endswith(quote):
                    continue
                text = text[len(quote) : -len(quote)]
                break
            result.exports.append(text)


class PyCallExtractor:
    """
    Extracts function/method calls, exception handlers, and raise statements.

    This extractor handles:
    - Simple function calls: foo()
    - Method calls: obj.method()
    - Chained calls: obj.method1().method2()
    - Exception handlers: except Error:
    - Raise statements: raise SomeError
    """

    def __init__(self) -> None:
        """Initialize the call extractor."""

    def extract_calls_recursive(self, node: Node, result: ParseResult) -> None:
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
        node_type = node.type
        if node_type == "call":
            self._extract_call(node, result)
            for child in node.children:
                self.extract_calls_recursive(child, result)
            return

        # Handle exception handlers: except SomeError:
        if node_type == "except_clause":
            self._extract_exception_handler(node, result)
            for child in node.children:
                self.extract_calls_recursive(child, result)
            return  # prevent fall-through to generic child recursion

        # Handle raise statements: raise SomeError
        if node_type == "raise_statement":
            self._extract_raise_statement(node, result)
            for child in node.children:
                self.extract_calls_recursive(child, result)
            return  # prevent fall-through to generic child recursion

        # Continue recursion for all other node types
        for child in node.children:
            self.extract_calls_recursive(child, result)

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
        if not func_node:
            func_node = self._find_fallback_function_node(call_node)

        if not func_node:
            return

        node_type = func_node.type
        line = call_node.start_point[0] + 1
        arguments = self._extract_identifier_arguments(call_node)

        # Dispatch based on function node type
        if node_type == "identifier":
            self._extract_simple_call(func_node, line, arguments, result)
        elif node_type == "attribute":
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
            if not child.is_named:
                continue
            return child

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
        if not (text := func_node.text):
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
            if not (text := child.text):
                continue
            method_name = text.decode("utf8")
            break

        # Extract receiver (the object the method is called on)
        receiver = ""
        obj_node = attr_node.children[0] if attr_node.children else None
        if obj_node is not None:
            obj_type = obj_node.type
            if obj_type == "identifier":
                if text := obj_node.text:
                    receiver = text.decode("utf8")
            elif obj_type == "attribute":
                # Nested attribute: self.logger.info -> receiver is "self"
                receiver = self._root_identifier(obj_node)
            elif obj_type == "call":
                # Chained call: get_user().save() -> receiver is "get_user"
                receiver = self._root_identifier(obj_node)

        return method_name, receiver

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
        while current:
            if current.type == "identifier" and (text := current.text):
                return text.decode("utf8")
            if not (children := current.children):
                break
            current = children[0]
        return ""

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
        if not (args_node := call_node.child_by_field_name("arguments")):
            return []

        identifiers: list[str] = []
        for child in args_node.children:
            child_type = child.type
            if child_type == "identifier":
                if not (text := child.text):
                    continue
                identifiers.append(text.decode("utf8"))
            elif child_type == "keyword_argument":
                # Extract identifier from keyword argument value
                if (
                    not (value_node := child.child_by_field_name("value"))
                    or value_node.type != "identifier"
                ):
                    continue
                if not (text := value_node.text):
                    continue
                identifiers.append(text.decode("utf8"))
        return identifiers

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
            child_type = child.type
            if child_type == "identifier":
                # Single exception: except Error:
                if not (text := child.text):
                    continue
                result.calls.append(
                    CallInfo(
                        name=text.decode("utf8"),
                        line=child.start_point[0] + 1,
                    ),
                )
            elif child_type == "tuple":
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
            elif child_type == "as_pattern":
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
            if not (text := child.text):
                continue
            result.calls.append(
                CallInfo(
                    name=text.decode("utf8"),
                    line=child.start_point[0] + 1,
                ),
            )


@dataclass(frozen=True)
class BodyContent:
    node_type: str
    definition_node: Node
    content: str
    result: ParseResult
    count_before: int


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
    PY_LANGUAGE = Language(language())

    def __init__(self) -> None:
        """Initialize the Python parser with a tree-sitter parser instance."""
        self._parser = Parser(self.PY_LANGUAGE)

        # Initialize extractors
        self._type_extractor = PyTypeExtractor()
        self._function_extractor = PyFunctionExtractor(self._type_extractor)
        self._class_extractor = PyClassExtractor(self._type_extractor)
        self._import_extractor = PyImportExtractor()
        self._call_extractor = PyCallExtractor()
        self._decorator_extractor = PyDecoratorExtractor()

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
        self._call_extractor.extract_calls_recursive(root, result)

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
                    self._function_extractor.extract_function(child, content, result, class_name)
                    self._walk_function_body(child, content, result)
                case "class_definition":
                    args = child, content, result
                    if extracted_class_name := self._class_extractor.extract_class(*args):
                        self._walk_class_body(child, content, result, extracted_class_name)
                case "import_statement":
                    self._import_extractor.extract_import(child, result)
                case "import_from_statement":
                    self._import_extractor.extract_import_from(child, result)
                case "decorated_definition":
                    self._extract_decorated(child, content, result, class_name)
                case "expression_statement":
                    # Only extract variable annotations here; calls are
                    # handled by the scope-level _extract_calls_recursive.
                    self._extract_annotations_from_expression(child, result)
                case _:
                    # Continue recursion for other node types
                    self._walk(child, content, result, class_name)

    def _walk_function_body(
        self,
        func_node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """Walk the body of a function to extract nested definitions."""
        if not (body := func_node.child_by_field_name("body")):
            return
        self._walk(body, content, result, class_name="")

    def _walk_class_body(
        self,
        class_node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Walk the body of a class to extract methods and nested definitions."""
        if not (body := class_node.child_by_field_name("body")):
            return
        self._walk(body, content, result, class_name=class_name)

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
        decorators, definition_node = self._decorator_extractor.extract_decorators(node)

        if definition_node is None:
            return

        # Track symbol count before extraction to associate decorators
        count_before = len(result.symbols)
        node_type = definition_node.type

        # Extract the underlying definition
        if node_type == "function_definition":
            self._function_extractor.extract_function(definition_node, content, result, class_name)
        elif node_type == "class_definition":
            self._class_extractor.extract_class(definition_node, content, result)

        # Attach decorators to the newly added symbol
        if count_before < len(result.symbols):
            result.symbols[count_before].decorators = decorators

        args = BodyContent(node_type, definition_node, content, result, count_before)
        self._process_body_content(args)

    def _process_body_content(self, deps: BodyContent) -> None:
        """Process the body content of a function or class."""
        # Process body contents if it's a class with the class_name context

        node_type = deps.node_type
        if (
            node_type == "class_definition"
            and (body := deps.definition_node.child_by_field_name("body"))
            and deps.count_before < len(deps.result.symbols)
        ):
            # Get the class name from the symbol we just added
            new_class_name = deps.result.symbols[deps.count_before].name
            self._walk(body, deps.content, deps.result, class_name=new_class_name)
            return

        if node_type == "function_definition" and (
            body := deps.definition_node.child_by_field_name("body")
        ):
            # Process nested definitions within the function body.
            self._walk(body, deps.content, deps.result, class_name="")

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
            if child.type != "assignment":
                continue
            self._class_extractor.extract_variable_annotation(child, result)
            self._class_extractor.extract_all_exports(child, result)
