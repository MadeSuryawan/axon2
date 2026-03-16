"""
TypeScript / TSX / JavaScript parser using tree-sitter.

This module provides a parser that extracts symbols (functions, classes,
methods, interfaces, type aliases), imports, call expressions, type
annotation references, and heritage (extends/implements) relationships
from TypeScript, TSX, and JavaScript source files using tree-sitter.

Design Decisions:
- Uses tree-sitter's incremental parsing for efficient AST traversal
- Separates extraction concerns into specialized extractor classes
- Builtin types are filtered out to reduce noise in type reference analysis
- Supports multiple dialects: typescript, tsx, and javascript

Key Responsibilities:
- Parse TypeScript/JS source into an Abstract Syntax Tree (AST)
- Extract symbol definitions (functions, classes, methods, interfaces)
- Extract import statements (ES modules and CommonJS require)
- Extract function/method calls and constructor calls
- Extract type annotations and heritage relationships
- Extract export statements
"""

from collections.abc import Callable

from tree_sitter import Language, Node, Parser
from tree_sitter_javascript import language
from tree_sitter_typescript import language_tsx, language_typescript

from axon.core.parsers.base import (
    CallInfo,
    ImportInfo,
    LanguageParser,
    ParseResult,
    SymbolInfo,
    TypeRef,
)


class TsTypeExtractor:
    """
    Extracts type annotations from TypeScript/JavaScript source code.

    Handles parameter types, return types, variable annotations,
    and various type expression patterns. Filters out builtin types
    to reduce noise in type reference analysis.
    """

    # Builtin types that are commonly used but not useful for type analysis.
    _BUILTIN_TYPES = frozenset(
        {
            "string",
            "number",
            "boolean",
            "void",
            "any",
            "unknown",
            "never",
            "null",
            "undefined",
            "object",
        },
    )

    @staticmethod
    def type_annotation_name(annotation_node: Node) -> str:
        """
        Return the simple type name from a type_annotation node.

        Handles type_identifier, predefined_type, and identifier children.
        For compound types (unions, generics, etc.) returns the text of
        the first recognizable child.

        Args:
            annotation_node: The type_annotation AST node.

        Returns:
            The type name as a string, or empty string if not found.
        """
        for child in annotation_node.children:
            if child.type not in ("type_identifier", "predefined_type", "identifier"):
                continue
            if not (text := child.text):
                continue
            return text.decode()
        return ""

    @staticmethod
    def string_value(string_node: Node) -> str:
        """
        Extract the raw string value from a tree-sitter string node.

        String nodes look like: string -> [quote, string_fragment, quote].
        Tries to find the string_fragment first, then falls back to
        stripping outer quotes from the whole text.

        Args:
            string_node: The string AST node.

        Returns:
            The decoded string value without quotes.
        """
        for child in string_node.children:
            if child.type != "string_fragment":
                continue
            if not (child_text := child.text):
                continue
            return child_text.decode()

        if not (node_text := string_node.text):
            return ""
        # Fallback: strip outer quotes from the whole text.
        text = node_text.decode()
        if len(text) >= 2 and text[0] in ("'", '"', "`") and text[-1] in ("'", '"', "`"):
            return text[1:-1]
        return text

    @classmethod
    def is_builtin_type(cls, type_name: str) -> bool:
        """Check if the given type name is a builtin type to filter out."""
        return type_name.lower() in cls._BUILTIN_TYPES

    def extract_function_types(
        self,
        func_node: Node,
        func_name: str,
        result: ParseResult,
    ) -> None:
        """
        Extract parameter types and return type from a function-like node.

        Processes both formal_parameters and type_annotation nodes to
        extract type information for parameters and return types.

        Args:
            func_node: The function AST node (function_declaration, method_definition, etc.).
            func_name: The name of the function (for context).
            result: The parse result to populate with type references.
        """
        # Try to get parameters from the "parameters" field first
        # Some nodes use "formal_parameters" via children iteration
        if not (params := func_node.child_by_field_name("parameters")):
            for child in func_node.children:
                if child.type == "formal_parameters":
                    params = child
                    break
        else:
            self._extract_param_types(params, result)

        # Return type: type_annotation directly on the function node
        type_name, line = self.get_type_annotation(func_node.children)
        if not type_name:
            return
        result.type_refs.append(
            TypeRef(
                name=type_name,
                kind="return",
                line=line,
            ),
        )

    def _extract_param_types(self, params_node: Node, result: ParseResult) -> None:
        """
        Extract type annotations from function parameters.

        Args:
            params_node: The parameters AST node.
            result: The parse result to populate.
        """
        for param in params_node.children:
            if param.type not in ("required_parameter", "optional_parameter"):
                continue

            # Get parameter name
            param_name_node = param.child_by_field_name("name")
            if not param_name_node:
                # Fallback: first identifier child
                for sub in param.children:
                    if sub.type != "identifier":
                        continue
                    param_name_node = sub
                    break

            if not param_name_node:
                continue

            if not (param_text := param_name_node.text):
                continue
            param_name = param_text.decode()

            # Get type annotation
            type_name, line = self.get_type_annotation(param.children)
            if not type_name:
                continue
            result.type_refs.append(
                TypeRef(
                    name=type_name,
                    kind="param",
                    line=line,
                    param_name=param_name,
                ),
            )

    def get_type_annotation(self, nodes: list[Node]) -> tuple[str, int]:
        """
        Extract type annotations from a list of nodes.

        Args:
            nodes: List of AST nodes to search for type annotations.
        """
        for node in nodes:
            if node.type != "type_annotation":
                continue
            if not (type_name := self.type_annotation_name(node)) or self.is_builtin_type(
                type_name,
            ):
                continue
            return type_name, node.start_point[0] + 1
        return "", 0


class TsImportExtractor:
    """
    Extracts import statements from TypeScript/JavaScript source code.

    Handles both ES module imports and CommonJS require() calls:
    - ES modules: import { A, B } from '...'
    - Default imports: import Foo from '...'
    - Namespace imports: import * as utils from '...'
    - CommonJS: const foo = require('./bar')
    """

    def __init__(self, type_extractor: TsTypeExtractor) -> None:
        """
        Initialize the import extractor.

        Args:
            type_extractor: The type extractor for string value extraction.
        """
        self._type_extractor = type_extractor

    def extract_import(self, node: Node, result: ParseResult) -> None:
        """
        Handle ES module import statements.

        Extracts module path, imported names, and handles various import forms:
        - Named imports: import { A, B } from '...'
        - Default import: import Foo from '...'
        - Namespace import: import * as utils from '...'
        - Side-effect import: import './module'

        Args:
            node: The import_statement AST node.
            result: The parse result to populate.
        """

        if not (module_str := self._get_source_value(node)):
            return

        if not (import_clause := self._get_import_clause(node)):
            return

        names, alias = self._extract_imported_names(import_clause)
        result.imports.append(
            ImportInfo(
                module=module_str,
                names=names,
                is_relative=module_str.startswith("."),
                alias=alias,
            ),
        )

    def _get_source_value(self, node: Node) -> str | None:
        """
        Get the source value from an import statement.

        Args:
            node: The import_statement AST node from tree-sitter.

        Returns:
            The source value extracted from the node.
        """
        # Try to get source from the "source" field first
        if source_node := node.child_by_field_name("source"):
            return self._type_extractor.string_value(source_node)

        # Fallback: look for a string child after 'from'
        for child in node.children:
            if child.type != "string":
                continue
            return self._type_extractor.string_value(child)

    def _get_import_clause(self, node: Node) -> Node | None:
        """
        Get the import_clause from an import_statement node.

        Args:
            node: The import_statement AST node from tree-sitter.

        Returns:
            The import_clause extracted from the node.
        """
        for child in node.children:
            if child.type != "import_clause":
                continue
            return child

    def _extract_imported_names(
        self,
        import_clause: Node,
    ) -> tuple[list[str], str]:
        """
        Extract imported names from an import_clause node.

        Processes the children of an import_clause AST node to handle all
        ES module import patterns:
        - Named imports: import { A, B } from '...'
        - Namespace imports: import * as utils from '...'
        - Default imports: import Foo from '...'

        Args:
            import_clause: The import_clause AST node from tree-sitter.

        Returns:
            A tuple containing:
            - list[str]: The imported names extracted from the clause.
            - str: The namespace alias (only populated for namespace imports, otherwise empty).
        """
        names: list[str] = []
        alias = ""
        for clause_child in import_clause.children:
            if clause_child.type == "named_imports":
                names.extend(self._import_a_b_from(clause_child))
            elif clause_child.type == "namespace_import":
                clause_names, clause_alias = self._import_star_as(clause_child)
                alias = clause_alias
                names.extend(clause_names)
            elif clause_child.type == "identifier":
                # import Foo from '...' (default import)
                if not (text := clause_child.text):
                    continue
                names.append(text.decode())

        return names, alias

    def _import_a_b_from(self, clause_child: Node) -> list[str]:
        """
        Extract imported names from a named_imports node.

        Args:
            clause_child: The named_imports AST node from tree-sitter.

        Returns:
            list[str]: The imported names extracted from the clause.
        """
        names = []
        for spec in clause_child.children:
            if spec.type != "import_specifier":
                continue
            if not (name_node := spec.child_by_field_name("name")) or not (text := name_node.text):
                continue
            names.append(text.decode())
        return names

    def _import_star_as(self, clause_child: Node) -> tuple[list[str], str]:
        """
        Extract imported names from a namespace_import node.

        Args:
            clause_child: The namespace_import AST node from tree-sitter.

        Returns:
            A tuple containing:
            - list[str]: The imported names extracted from the clause.
            - str: The namespace alias.
        """
        names = []
        alias = ""
        for ns_child in clause_child.children:
            if ns_child.type != "identifier":
                continue
            if not (text := ns_child.text):
                continue
            alias = text.decode()
            names.append(alias)
            break
        return names, alias

    def extract_require(
        self,
        declarator_node: Node,
        var_name: str,
        call_node: Node,
        result: ParseResult,
    ) -> None:
        """
        If the call is ``require('./foo')``, emit an ImportInfo.

        Handles CommonJS require() pattern: const foo = require('./bar')

        Args:
            declarator_node: The variable_declarator AST node.
            var_name: The variable name being assigned.
            call_node: The call_expression AST node.
            result: The parse result to populate.
        """
        func_node = call_node.child_by_field_name("function")
        if func_node is None:
            return

        func_text = func_node.text
        if not func_text or func_text.decode() != "require":
            return

        args = call_node.child_by_field_name("arguments")
        if args is None:
            return

        module_str = ""
        for arg_child in args.children:
            if arg_child.type == "string":
                module_str = self._type_extractor.string_value(arg_child)
                break

        if not module_str:
            return

        result.imports.append(
            ImportInfo(
                module=module_str,
                names=[var_name],
                is_relative=module_str.startswith("."),
            ),
        )


class TsFunctionExtractor:
    """
    Extracts function and method definitions from TypeScript/JavaScript source.

    Handles standalone functions, methods within classes, arrow functions,
    function expressions, and decorated definitions.
    """

    def __init__(self, type_extractor: TsTypeExtractor) -> None:
        """
        Initialize the function extractor.

        Args:
            type_extractor: The type extractor for parameter/return types.
        """
        self._type_extractor = type_extractor

    @staticmethod
    def build_signature(node: Node, name: str) -> str:
        """
        Build a human-readable signature line for a function-like node.

        Includes the parameter list and return type (if present).

        Args:
            node: The function AST node.
            name: The function name.

        Returns:
            A string representation of the function signature.
        """
        params_text: str | None = None
        return_type: str | None = None

        for child in node.children:
            child_type = child.type
            if child_type == "formal_parameters" and (text := child.text):
                params_text = text.decode()
            elif child_type == "type_annotation" and (text := child.text):
                return_type = text.decode()

        sig = f"{name}{params_text}"
        if return_type:
            sig += return_type
        return sig

    @staticmethod
    def unwrap_to_function(node: Node) -> Node | None:
        """
        Return the underlying function node, unwrapping wrapper calls.

        Handles direct arrow_function/function_expression as well as
        wrapper patterns like asyncHandler(async (req, res) => {...}).

        Args:
            node: The AST node to unwrap.

        Returns:
            The underlying function node, or None if not found.
        """
        if node.type in ("arrow_function", "function_expression"):
            return node
        if node.type != "call_expression":
            return None
        if not (args := node.child_by_field_name("arguments")):
            return None
        for arg in args.children:
            if arg.type not in ("arrow_function", "function_expression"):
                continue
            return arg

    def extract_function_declaration(
        self,
        node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """
        Extract a function declaration.

        Processes function_declaration nodes and extracts their name,
        signature, and type annotations.

        Args:
            node: The function_declaration AST node.
            content: The source content for text extraction.
            result: The parse result to populate.
        """
        if not (name_node := node.child_by_field_name("name")):
            return

        if not (name_text := name_node.text):
            return

        name = name_text.decode()

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="function",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                content=content[node.start_byte : node.end_byte],
                signature=self.build_signature(node, name),
            ),
        )

        self._type_extractor.extract_function_types(node, name, result)

    def extract_method(
        self,
        node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """
        Extract a method_definition inside a class body.

        Args:
            node: The method_definition AST node.
            content: The source content for text extraction.
            result: The parse result to populate.
        """

        if not (name_node := node.child_by_field_name("name")):
            return

        if not (name_text := name_node.text):
            return

        name = name_text.decode()

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="method",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                content=content[node.start_byte : node.end_byte],
                signature=self.build_signature(node, name),
                class_name=self._find_parent_class_name(node),
            ),
        )

        self._type_extractor.extract_function_types(node, name, result)

    def extract_assigned_function(
        self,
        declarator_node: Node,
        name: str,
        func_node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """
        Extract an arrow function or function expression assigned to a variable.

        Handles patterns like:
        - const foo = () => {...}
        - const bar = function() {...}

        Args:
            declarator_node: The variable_declarator AST node.
            name: The variable name.
            func_node: The arrow_function or function_expression node.
            content: The source content for text extraction.
            result: The parse result to populate.
        """
        outer = declarator_node.parent
        if not outer:
            outer = declarator_node

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="function",
                start_line=outer.start_point[0] + 1,
                end_line=outer.end_point[0] + 1,
                content=content[outer.start_byte : outer.end_byte],
                signature=self.build_signature(func_node, name),
            ),
        )

        self._type_extractor.extract_function_types(func_node, name, result)

    def extract_module_exports_function(
        self,
        assignment_node: Node,
        sym_name: str,
        right_node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """
        Extract function from module.exports assignment.

        Handles patterns like:
        - module.exports = function() {...}
        - module.exports = { foo: function() {...} }

        Args:
            assignment_node: The assignment_expression AST node.
            sym_name: The exported symbol name.
            right_node: The right side of the assignment.
            content: The source content for text extraction.
            result: The parse result to populate.
        """
        if not (func_node := self.unwrap_to_function(right_node)):
            return

        result.symbols.append(
            SymbolInfo(
                name=sym_name,
                kind="function",
                start_line=assignment_node.start_point[0] + 1,
                end_line=assignment_node.end_point[0] + 1,
                content=content[assignment_node.start_byte : assignment_node.end_byte],
                signature=self.build_signature(func_node, sym_name),
            ),
        )

        self._type_extractor.extract_function_types(func_node, sym_name, result)

    @staticmethod
    def _find_parent_class_name(node: Node) -> str:
        """
        Walk up the tree to find the enclosing class name.

        Args:
            node: The AST node to start from.

        Returns:
            The class name if found, empty string otherwise.
        """
        current = node.parent
        while current is not None:
            if (
                current.type in ("class_declaration", "class_expression")
                and (name_node := current.child_by_field_name("name"))
                and (text := name_node.text)
            ):
                return text.decode()
            current = current.parent
        return ""


class TsClassExtractor:
    """Extracts class definitions, interfaces, and type aliases from TypeScript/JavaScript source."""

    def __init__(self, type_extractor: TsTypeExtractor) -> None:
        """
        Initialize the class extractor.

        Args:
            type_extractor: The type extractor for type annotations.
        """
        self._type_extractor = type_extractor

    @staticmethod
    def _get_name_text(node: Node) -> str | None:
        """
        Get the name text from a node.

        Args:
            node: The AST node to get the name from.

        Returns:
            The name text if found, None otherwise.
        """
        if not (name_node := node.child_by_field_name("name")):
            return None
        if not (name_text := name_node.text):
            return None
        return name_text.decode()

    def extract_class(
        self,
        node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """
        Extract a class declaration and its heritage relationships.

        Args:
            node: The class_declaration AST node.
            content: The source content for text extraction.
            result: The parse result to populate.
        """
        if not (name := self._get_name_text(node)):
            return

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="class",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                content=content[node.start_byte : node.end_byte],
            ),
        )

        for child in node.children:
            if child.type != "class_heritage":
                continue
            self._extract_class_heritage(name, child, result)

    def _extract_class_heritage(
        self,
        class_name: str,
        heritage_node: Node,
        result: ParseResult,
    ) -> None:
        """
        Extract extends and implements clauses from class heritage.

        Args:
            class_name: The name of the class.
            heritage_node: The class_heritage AST node.
            result: The parse result to populate.
        """
        for child in heritage_node.children:
            if child.type == "extends_clause":
                self._extract_heritage(result, child.children, class_name, "extends")
            elif child.type == "implements_clause":
                self._extract_heritage(result, child.children, class_name, "implements")

    @staticmethod
    def _extract_heritage(
        result: ParseResult,
        children: list[Node],
        class_name: str,
        kind: str,
    ) -> None:
        """
        Extract extends and implements clauses from class heritage.

        Args:
            result: The parse result to populate.
            children: The children of the class_heritage AST node.
            class_name: The name of the class.
            kind: The kind of heritage (extends or implements).
        """
        for sub in children:
            if sub.type in ("identifier", "type_identifier") and (text := sub.text):
                result.heritage.append((class_name, kind, text.decode()))

            elif sub.type == "generic_type" and (name_node := sub.child_by_field_name("name")):
                if not (text := name_node.text):
                    continue
                result.heritage.append((class_name, kind, text.decode()))

    def extract_interface(
        self,
        node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """
        Extract an interface declaration and its extends clause.

        Args:
            node: The interface_declaration AST node.
            content: The source content for text extraction.
            result: The parse result to populate.
        """

        if not (name := self._get_name_text(node)):
            return

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="interface",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                content=content[node.start_byte : node.end_byte],
            ),
        )

        for child in node.children:
            if child.type != "extends_type_clause":
                continue
            self._extract_heritage(result, child.children, name, "extends")

    def extract_type_alias(
        self,
        node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """
        Extract a type alias declaration.

        Args:
            node: The type_alias_declaration AST node.
            content: The source content for text extraction.
            result: The parse result to populate.
        """
        if not (name_text := self._get_name_text(node)):
            return

        result.symbols.append(
            SymbolInfo(
                name=name_text,
                kind="type_alias",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                content=content[node.start_byte : node.end_byte],
            ),
        )


class TsCallExtractor:
    """
    Extracts function/method calls and constructor calls from TypeScript/JavaScript source.

    Handles call expressions, method calls, and new expressions.
    """

    def __init__(self) -> None:
        """Initialize the call extractor."""

    @staticmethod
    def extract_identifier_arguments(call_node: Node) -> list[str]:
        """
        Extract bare identifier arguments from a call_expression node.

        Returns names of arguments that are plain identifiers (not literals,
        calls, or attribute accesses).

        Args:
            call_node: The call_expression AST node.

        Returns:
            List of identifier argument names.
        """

        if not (args_node := call_node.child_by_field_name("arguments")):
            return []

        identifiers: list[str] = []
        for child in args_node.children:
            if child.type != "identifier":
                continue
            if not (text := child.text):
                continue
            identifiers.append(text.decode())
        return identifiers

    def extract_call(
        self,
        node: Node,
        result: ParseResult,
    ) -> None:
        """
        Extract a call expression.

        Handles both simple function calls and method calls:
        - foo()
        - obj.method()

        Args:
            node: The call_expression AST node.
            result: The parse result to populate.
        """
        if not (func_node := node.child_by_field_name("function")):
            return

        line = node.start_point[0] + 1
        arguments = self.extract_identifier_arguments(node)

        if func_node.type == "member_expression":
            self._extract_method_call(func_node, line, arguments, result)
        elif func_node.type == "identifier":
            if not (name := func_node.text):
                return
            if (decoded_name := name.decode()) == "require":
                # Skip require() since it's handled as an import
                return
            result.calls.append(CallInfo(name=decoded_name, line=line, arguments=arguments))

    def _extract_method_call(
        self,
        func_node: Node,
        line: int,
        arguments: list[str],
        result: ParseResult,
    ) -> None:
        """
        Extract a method call from a member_expression.

        Args:
            func_node: The member_expression AST node.
            line: The line number of the call.
            arguments: List of identifier argument names.
            result: The parse result to populate.
        """

        if not (prop_node := func_node.child_by_field_name("property")):
            return
        if not (prop_text := prop_node.text):
            return

        receiver = ""
        if (obj_node := func_node.child_by_field_name("object")) and (obj_text := obj_node.text):
            receiver = obj_text.decode()

        result.calls.append(
            CallInfo(
                name=prop_text.decode(),
                line=line,
                receiver=receiver,
                arguments=arguments,
            ),
        )

    def extract_new_expression(
        self,
        node: Node,
        result: ParseResult,
    ) -> None:
        """
        Extract a new expression (constructor call).

        Handles ``new ClassName(args)`` patterns.

        Args:
            node: The new_expression AST node.
            result: The parse result to populate.
        """
        if not (constructor_node := node.child_by_field_name("constructor")):
            return

        line = node.start_point[0] + 1
        arguments = self.extract_identifier_arguments(node)

        if constructor_node.type == "identifier":
            if not (constructor_text := constructor_node.text):
                return
            result.calls.append(
                CallInfo(
                    name=constructor_text.decode(),
                    line=line,
                    arguments=arguments,
                ),
            )
        elif constructor_node.type == "member_expression":
            self._extract_method_call(constructor_node, line, arguments, result)


class TsExportExtractor:
    """
    Extracts export statements from TypeScript/JavaScript source.

    Handles various export forms:
    - Named exports: export function foo() {}
    - Default exports: export default class Foo {}
    - Export from: export { name1, name2 }
    - module.exports: module.exports = {...}
    - exports.X: exports.name = fn
    """

    def __init__(self, function_extractor: TsFunctionExtractor) -> None:
        """
        Initialize the export extractor.

        Args:
            function_extractor: The function extractor for module.exports functions.
        """
        self._function_extractor = function_extractor
        self._child_types = (
            "function_declaration",
            "class_declaration",
            "interface_declaration",
            "type_alias_declaration",
        )

    def extract_export(
        self,
        node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """
        Handle export statements.

        Handles:
        - export function foo() {}
        - export class Bar {}
        - export interface Baz {}
        - export type Qux = ...
        - export const foo = ...
        - export { name1, name2 }

        Args:
            node: The export_statement AST node.
            content: The source content for text extraction.
            result: The parse result to populate.
        """
        for child in node.children:
            if child.type not in self._child_types:
                continue
            if (name_node := child.child_by_field_name("name")) and (text := name_node.text):
                result.exports.append(text.decode())
            elif child.type in ("lexical_declaration", "variable_declaration"):
                self._append_exports(child.children, "variable_declarator", result)
            elif child.type == "export_clause":
                # export { name1, name2 }
                self._append_exports(child.children, "export_specifier", result)

    def _append_exports(
        self,
        nodes: list[Node],
        node_type: str,
        result: ParseResult,
    ) -> None:
        """
        Append exports from a node to the result.

        Args:
            nodes: list of the AST node.
            node_type: The type of the node.
            result: The parse result to populate.
        """
        for spec in nodes:
            if spec.type != node_type:
                continue
            if (name_node := spec.child_by_field_name("name")) and (text := name_node.text):
                result.exports.append(text.decode())

    def extract_module_exports(
        self,
        node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """
        Handle module.exports and exports patterns.

        Handles:
        - module.exports = X
        - module.exports = { A, B }
        - exports.name = fn

        Args:
            node: The expression_statement AST node.
            content: The source content for text extraction.
            result: The parse result to populate.
        """
        for child in node.children:
            if child.type != "assignment_expression":
                continue

            left = child.child_by_field_name("left")
            right = child.child_by_field_name("right")
            if left is None or right is None:
                continue

            if not (left_text := left.text):
                continue

            if left_text.decode() in ("module.exports", "exports"):
                self._handle_module_exports_assignment(left, right, child, content, result)
            else:
                # exports.X = fn / module.exports.X = fn
                self._handle_named_export(left, child, content, result)

    def _handle_module_exports_assignment(
        self,
        left: Node,
        right: Node,
        assignment_node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """
        Handle module.exports = X assignments.

        Args:
            left: The left side of the assignment.
            right: The right side of the assignment.
            assignment_node: The assignment_expression node.
            content: The source content.
            result: The parse result to populate.
        """

        if not (left_text := left.text):
            return

        if right.type == "identifier" and (right_text := right.text):
            result.exports.append(right_text.decode())
        elif right.type == "object":
            # module.exports = { Foo, Bar, baz: something }
            for prop in right.children:
                if prop.type == "shorthand_property_identifier" and (prop_text := prop.text):
                    result.exports.append(prop_text.decode())
                elif (
                    prop.type == "pair"
                    and (key_node := prop.child_by_field_name("key"))
                    and (key_text := key_node.text)
                ):
                    result.exports.append(key_text.decode())
            return  # Don't process as function for object exports

        self._right_is_function(right, result, left_text, assignment_node, content)

    def _right_is_function(
        self,
        right: Node,
        result: ParseResult,
        left_text: bytes,
        assignment_node: Node,
        content: str,
    ) -> None:
        """
        Check if the right side of an assignment is a function.

        Args:
            right: The right side of the assignment.
            result: The parse result to populate.
            left_text: The left side of the assignment.
            assignment_node: The assignment_expression node.
            content: The source content.

        Returns:
            None
        """
        # Check if right is a function and extract it
        if not (self._function_extractor.unwrap_to_function(right)):
            return

        # left_text is bytes, need to decode first for string operations
        left_decoded = left_text.decode()
        sym_name = left_decoded.split(".")[-1] if "." in left_decoded else left_decoded

        if sym_name == "module":
            right_text = right.text
            sym_name = right_text.decode() if right_text else ""

        args = assignment_node, sym_name, right, content, result
        self._function_extractor.extract_module_exports_function(*args)

    def _handle_named_export(
        self,
        left: Node,
        assignment_node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """
        Handle exports.X = fn patterns.

        Args:
            left: The left side of the assignment (member_expression).
            assignment_node: The assignment_expression node.
            content: The source content.
            result: The parse result to populate.
        """
        if left.type != "member_expression":
            return

        prop_node = left.child_by_field_name("property")
        if not (obj_node := left.child_by_field_name("object")) or not prop_node:
            return

        if not (obj_text := obj_node.text):
            return

        if obj_text.decode() not in ("exports", "module.exports"):
            return

        if not (prop_text := prop_node.text):
            return

        sym_name = prop_text.decode()
        result.exports.append(sym_name)

        # Extract function if right side is a function
        if not (right := assignment_node.child_by_field_name("right")):
            return

        if self._function_extractor.unwrap_to_function(right):
            self._function_extractor.extract_module_exports_function(
                assignment_node,
                sym_name,
                right,
                content,
                result,
            )


class TypeScriptParser(LanguageParser):
    """
    Parses TypeScript, TSX, or JavaScript files using tree-sitter.

    This parser extracts structured information from TypeScript/JS source
    files including symbol definitions, imports, calls, type annotations,
    and heritage relationships. The extracted data is used for code analysis,
    dependency tracking, and knowledge graph construction.

    Attributes:
        TS_LANGUAGE: The tree-sitter language instance for TypeScript.
        TSX_LANGUAGE: The tree-sitter language instance for TSX.
        JS_LANGUAGE: The tree-sitter language instance for JavaScript.
        _BUILTIN_TYPES: Frozenset of TypeScript/JS builtin types to filter out
            from type reference analysis.

    Example:
        >>> parser = TypeScriptParser("typescript")
        >>> result = parser.parse("function hello(): string { return 'hi'; }", "hello.ts")
        >>> result.symbols[0].name
        'hello'
    """

    # Language instances - shared across all parser instances for efficiency
    TS_LANGUAGE = Language(language_typescript())
    TSX_LANGUAGE = Language(language_tsx())
    JS_LANGUAGE = Language(language())

    _DIALECT_MAP: dict[str, Language] = {
        "typescript": TS_LANGUAGE,
        "tsx": TSX_LANGUAGE,
        "javascript": JS_LANGUAGE,
    }

    # Dispatch table mapping node types to handler methods that need (node, content, result)
    _DISPATCH_3_ARGS: dict[str, Callable[[Node, str, ParseResult], None]]

    # Dispatch table mapping node types to handler methods that need (node, result)
    _DISPATCH_2_ARGS: dict[str, Callable[[Node, ParseResult], None]]

    def __init__(self, dialect: str = "typescript") -> None:
        """
        Initialize the TypeScript parser.

        Args:
            dialect: One of "typescript", "tsx", or "javascript".

        Raises:
            ValueError: If the dialect is not supported.
        """
        if dialect not in self._DIALECT_MAP:
            details = (
                f"Unknown dialect {dialect!r}. "
                f"Expected one of: {', '.join(sorted(self._DIALECT_MAP))}"
            )
            raise ValueError(details)
        self.dialect = dialect
        self._language = self._DIALECT_MAP[dialect]
        self._parser = Parser(self._language)

        # Initialize extractors
        self._type_extractor = TsTypeExtractor()
        self._import_extractor = TsImportExtractor(self._type_extractor)
        self._function_extractor = TsFunctionExtractor(self._type_extractor)
        self._class_extractor = TsClassExtractor(self._type_extractor)
        self._call_extractor = TsCallExtractor()
        self._export_extractor = TsExportExtractor(self._function_extractor)

        # Build dispatch tables after extractors are initialized
        self._build_dispatch_tables()

    def parse(self, content: str, file_path: str) -> ParseResult:
        """
        Parse TypeScript/JS source and return structured information.

        This is the main entry point for parsing a TypeScript or JavaScript
        source file. It performs a single pass through the AST to extract
        all relevant information.

        Args:
            content: The TypeScript/JS source code as a string.
            file_path: The path to the source file (used for context).

        Returns:
            ParseResult containing all extracted symbols, imports, calls,
            type references, heritage relationships, and exports.
        """
        tree = self._parser.parse(content.encode("utf-8"))
        result = ParseResult()
        self._walk(tree.root_node, content, result)
        return result

    def _build_dispatch_tables(self) -> None:
        """
        Build dispatch tables mapping node types to handler methods.

        This method initializes the dispatch dictionaries after all extractors
        are set up, allowing handler methods to be called via dictionary lookup
        instead of a large match statement.
        """
        # Handlers that need (node, content, result)
        self._DISPATCH_3_ARGS = {
            "export_statement": self._export_extractor.extract_export,
            "function_declaration": self._function_extractor.extract_function_declaration,
            "lexical_declaration": self._extract_variable_declaration,
            "variable_declaration": self._extract_variable_declaration,
            "class_declaration": self._class_extractor.extract_class,
            "interface_declaration": self._class_extractor.extract_interface,
            "type_alias_declaration": self._class_extractor.extract_type_alias,
            "expression_statement": self._export_extractor.extract_module_exports,
            "method_definition": self._function_extractor.extract_method,
        }
        # Handlers that need (node, result)
        self._DISPATCH_2_ARGS = {
            "import_statement": self._import_extractor.extract_import,
            "call_expression": self._call_extractor.extract_call,
            "new_expression": self._call_extractor.extract_new_expression,
        }

    # ---------------------------------------------------------------------------
    # Core AST Traversal
    # ---------------------------------------------------------------------------

    def _walk(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        visited: set[int] | None = None,
    ) -> None:
        """
        Walk the tree recursively, dispatching on node type.

        Uses a visited set (keyed by node id) to avoid processing
        the same subtree twice - e.g. class bodies that are walked by both
        _extract_class and the generic child recursion.

        Args:
            node: The current AST node to process.
            content: The source content for text extraction.
            result: The parse result to populate.
            visited: Set of already visited node IDs.
        """
        if not visited:
            visited = set()

        node_key = node.id
        if node_key in visited:
            return

        visited.add(node_key)
        self._match_case(node.type, node, content, result)

        # Continue recursion for all children
        for child in node.children:
            self._walk(child, content, result, visited)

    def _match_case(self, ntype: str, node: Node, content: str, result: ParseResult) -> None:
        """
        Dispatch to appropriate extractor based on node type.

        Uses dictionary-based dispatch tables to route node types to their
        respective handler methods, reducing cyclomatic complexity.

        Args:
            ntype: The node type.
            node: The AST node.
            content: The source content for text extraction.
            result: The parse result to populate.
        """
        # Try 3-arg handlers first (node, content, result)
        if handler := self._DISPATCH_3_ARGS.get(ntype):
            handler(node, content, result)
        # Try 2-arg handlers (node, result)
        elif handler := self._DISPATCH_2_ARGS.get(ntype):
            handler(node, result)

    def _extract_variable_declaration(
        self,
        node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """
        Handle variable declarations with arrow functions or require() calls.

        Processes lexical_declaration and variable_declaration nodes to
        extract:
        - Arrow functions assigned to variables
        - Function expressions
        - require() calls (treated as imports)

        Args:
            node: The declaration AST node.
            content: The source content for text extraction.
            result: The parse result to populate.
        """
        for child in node.children:
            if child.type != "variable_declarator":
                continue

            value_node = child.child_by_field_name("value")
            if not (name_node := child.child_by_field_name("name")) or not value_node:
                continue

            if not (name_text := name_node.text):
                continue

            var_name = name_text.decode()
            if value_node.type in ("arrow_function", "function_expression"):
                self._function_extractor.extract_assigned_function(
                    child,
                    var_name,
                    value_node,
                    content,
                    result,
                )
            elif value_node.type == "call_expression":
                self._import_extractor.extract_require(child, var_name, value_node, result)

            type_name, line = self._type_extractor.get_type_annotation(child.children)
            if not type_name:
                continue
            result.type_refs.append(
                TypeRef(
                    name=type_name,
                    kind="variable",
                    line=line,
                ),
            )
