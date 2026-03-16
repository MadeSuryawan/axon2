"""
Phase 3: Code parsing for Axon.

Takes file entries from the walker, parses each one with the appropriate
tree-sitter parser, and adds symbol nodes (Function, Class, Method, Interface,
TypeAlias, Enum) to the knowledge graph with DEFINES relationships from File
to Symbol.

The main entry point is :meth:`Parsing.process_parsing`, which parses files
in parallel, extracts symbols, and populates the knowledge graph.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor as TPool
from dataclasses import dataclass
from logging import getLogger
from threading import Lock

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.ingestion.walker import FileEntry
from axon.core.parsers.base import LanguageParser, ParseResult, SymbolInfo
from axon.core.parsers.python_lang import PythonParser
from axon.core.parsers.typescript import TypeScriptParser

logger = getLogger(__name__)


@dataclass
class FileParseData:
    """
    Parse results for a single file, kept for later phases.

    Attributes:
        file_path: Relative path to the parsed file.
        language: Language identifier (e.g., "python", "typescript").
        parse_result: The structured parse result containing symbols,
            imports, calls, heritage, and type references.
    """

    file_path: str
    language: str
    parse_result: ParseResult


# Dataclass to group symbol node creation parameters (avoids too many arguments)
@dataclass
class SymbolNodeParams:
    """Parameters for creating a symbol node in the knowledge graph."""

    symbol_id: str
    label: NodeLabel
    symbol: SymbolInfo
    file_entry: FileEntry
    symbol_name: str
    props: dict[str, list[str]]
    is_exported: bool


class Parsing:
    """
    Handles code parsing for the Axon knowledge graph.

    Parses source files using tree-sitter parsers (Python, TypeScript,
    JavaScript), extracts symbols (functions, classes, methods, interfaces,
    etc.), and populates the knowledge graph with Symbol nodes and DEFINES
    relationships from File nodes.

    Parser instances are cached per language to avoid repeated instantiation
    of tree-sitter Parser objects, which is expensive.

    Example usage:
        parser = Parsing(graph=my_graph)
        parse_data = parser.process_parsing(files)
    """

    # Mapping from symbol kinds to GraphNode labels
    _KIND_TO_LABEL: dict[str, NodeLabel] = {
        "function": NodeLabel.FUNCTION,
        "class": NodeLabel.CLASS,
        "method": NodeLabel.METHOD,
        "interface": NodeLabel.INTERFACE,
        "type_alias": NodeLabel.TYPE_ALIAS,
        "enum": NodeLabel.ENUM,
    }

    # Default maximum number of worker threads for parallel parsing
    _DEFAULT_MAX_WORKERS: int = 8

    _PARSER_FACTORIES: dict[str, Callable[[], LanguageParser]] = {
        "python": PythonParser,
        "typescript": lambda: TypeScriptParser(dialect="typescript"),
        "tsx": lambda: TypeScriptParser(dialect="tsx"),
        "javascript": lambda: TypeScriptParser(dialect="javascript"),
    }

    def __init__(
        self,
        graph: KnowledgeGraph,
        max_workers: int = _DEFAULT_MAX_WORKERS,
    ) -> None:
        """
        Initialize the Parsing processor.

        Args:
            graph: The knowledge graph to populate with parsed symbols.
            max_workers: Maximum number of threads for parallel parsing.
        """
        self._graph = graph
        self._max_workers = max_workers
        # Parser cache: maps language -> parser instance
        self._parser_cache: dict[str, LanguageParser] = {}
        self._parser_cache_lock = Lock()

    def process_parsing(
        self,
        files: list[FileEntry],
    ) -> list[FileParseData]:
        """
        Parse every file and populate the knowledge graph with symbol nodes.

        Parsing is done in parallel using a thread pool (tree-sitter releases
        the GIL during C parsing). Graph mutation remains sequential since
        :class:`KnowledgeGraph` is not thread-safe.

        For each symbol discovered during parsing a graph node is created with
        the appropriate label (Function, Class, Method, etc.) and a DEFINES
        relationship is added from the owning File node to the new symbol node.

        This method serves as the main entry point for the parsing phase.

        Args:
            files: File entries produced by the walker phase.

        Returns:
            A list of :class:`FileParseData` objects that carry the full parse
            results (imports, calls, heritage, type_refs) for use by later phases.
        """
        # Phase 1: Parse all files in parallel using thread pool
        all_parse_data = self._parse_all_files_parallel(files)

        # Phase 2: Graph mutation (sequential — not thread-safe)
        # Populate graph with symbol nodes and DEFINES relationships
        self._populate_graph_with_symbols(files, all_parse_data)

        return all_parse_data

    def _parse_all_files_parallel(
        self,
        files: list[FileEntry],
    ) -> list[FileParseData]:
        """
        Parse all files in parallel using a thread pool.

        Tree-sitter releases the GIL during C parsing, making this an
        effective way to speed up parsing of multiple files.

        Args:
            files: List of file entries to parse.

        Returns:
            List of FileParseData objects containing parse results.
        """
        logger.info("Running ThreadPoolExecutor, parsing %d files", len(files))
        with TPool(max_workers=self._max_workers) as executor:
            all_parse_data = list(
                executor.map(
                    lambda f: self._parse_file(f.path, f.content, f.language),
                    files,
                ),
            )
        return all_parse_data

    def _populate_graph_with_symbols(
        self,
        files: list[FileEntry],
        all_parse_data: list[FileParseData],
    ) -> None:
        """
        Populate the knowledge graph with symbol nodes and DEFINES relationships.

        For each file, creates GraphNodes for discovered symbols (functions,
        classes, methods, interfaces, etc.) and adds DEFINES relationships
        from the File node to each Symbol node.

        Args:
            files: Original list of file entries.
            all_parse_data: Parse results for each file.
        """
        for file_entry, parse_data in zip(files, all_parse_data, strict=True):
            # Generate file node ID
            file_id = generate_id(NodeLabel.FILE, file_entry.path)

            # Extract exported names from this file
            exported_names = set(parse_data.parse_result.exports)

            # Build class -> base class names mapping for class nodes
            class_bases = self._build_class_bases_map(parse_data.parse_result)

            # Create symbol nodes and DEFINES relationships
            self._add_symbols_to_graph(file_entry, parse_data, file_id, exported_names, class_bases)

    def _build_class_bases_map(self, parse_result: ParseResult) -> dict[str, list[str]]:
        """
        Build a mapping from class names to their base class names.

        This mapping is used to attach base class information to class nodes
        in the graph. Only "extends" relationships are tracked (not "implements").

        Args:
            parse_result: The parse result containing heritage information.

        Returns:
            Dictionary mapping class name -> list of base class names.
        """
        class_bases: dict[str, list[str]] = {}
        for cls_name, kind, parent_name in parse_result.heritage:
            if kind != "extends":
                continue
            class_bases.setdefault(cls_name, []).append(parent_name)
        return class_bases

    def _add_symbols_to_graph(
        self,
        file_entry: FileEntry,
        parse_data: FileParseData,
        file_id: str,
        exported_names: set[str],
        class_bases: dict[str, list[str]],
    ) -> None:
        """
        Add all symbols from a parsed file to the knowledge graph.

        For each symbol, creates a GraphNode with the appropriate label and
        properties, then adds a DEFINES relationship from the File node.

        Args:
            file_entry: The original file entry.
            parse_data: Parse results for this file.
            file_id: ID of the file node in the graph.
            exported_names: Set of exported symbol names from this file.
            class_bases: Mapping of class names to base classes.
        """
        for symbol in parse_data.parse_result.symbols:
            # Get the appropriate node label for this symbol kind
            if not (label := self._KIND_TO_LABEL.get(symbol.kind)):
                logger.warning(
                    "Unknown symbol kind %r for %s in %s, skipping",
                    symbol.kind,
                    symbol.name,
                    file_entry.path,
                )
                continue

            # Create symbol name: for methods, prepend class name to disambiguate
            # across different classes (e.g., "UserService.get_user")
            symbol_name = self._get_symbol_name(symbol)

            # Generate unique ID for this symbol
            symbol_id = generate_id(label, file_entry.path, symbol_name)

            # Build properties dict for the symbol node
            props = self._build_symbol_properties(symbol, class_bases)

            # Check if symbol is exported (visible outside the module)
            is_exported = symbol.name in exported_names

            # Use dataclass to group parameters for symbol node creation
            node_params = SymbolNodeParams(
                symbol_id=symbol_id,
                label=label,
                symbol=symbol,
                file_entry=file_entry,
                symbol_name=symbol_name,
                props=props,
                is_exported=is_exported,
            )

            # Create and add the symbol node to the graph
            self._create_symbol_node(node_params)

            # Create DEFINES relationship from File to Symbol
            self._create_defines_relationship(file_id, symbol_id)

    def _get_symbol_name(self, symbol: SymbolInfo) -> str:
        """
        Generate the appropriate name for a symbol node.

        For methods, uses "ClassName.method_name" format to disambiguate
        methods with the same name in different classes.

        Args:
            symbol: The symbol from parsing.

        Returns:
            The name to use for the symbol node.
        """
        if symbol.kind == "method" and symbol.class_name:
            return f"{symbol.class_name}.{symbol.name}"
        return symbol.name

    def _build_symbol_properties(
        self,
        symbol: SymbolInfo,
        class_bases: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """
        Build the properties dictionary for a symbol node.

        Includes decorators (if any) and base classes (for classes).

        Args:
            symbol: The symbol from parsing.
            class_bases: Mapping of class names to base classes.

        Returns:
            Dictionary of properties to attach to the symbol node.
        """
        props = {}
        if symbol.decorators:
            props["decorators"] = symbol.decorators
        if symbol.kind == "class" and symbol.name in class_bases:
            props["bases"] = class_bases[symbol.name]
        return props

    def _create_symbol_node(self, params: SymbolNodeParams) -> None:
        """
        Create and add a symbol node to the knowledge graph.

        Args:
            params: Dataclass containing all parameters for node creation.
        """
        node = GraphNode(
            id=params.symbol_id,
            label=params.label,
            name=params.symbol.name,
            file_path=params.file_entry.path,
            start_line=params.symbol.start_line,
            end_line=params.symbol.end_line,
            content=params.symbol.content,
            signature=params.symbol.signature,
            class_name=params.symbol.class_name,
            language=params.file_entry.language,
            is_exported=params.is_exported,
            properties=params.props,
        )
        self._graph.add_node(node)

    def _create_defines_relationship(self, source_id: str, target_id: str) -> None:
        """
        Create a DEFINES relationship from a File node to a Symbol node.

        Args:
            source_id: ID of the File node (source).
            target_id: ID of the Symbol node (target).
        """
        rel_id = f"defines:{source_id}->{target_id}"
        rel = GraphRelationship(
            id=rel_id,
            type=RelType.DEFINES,
            source=source_id,
            target=target_id,
        )
        self._graph.add_relationship(rel)

    def _parse_file(self, file_path: str, content: str, language: str) -> FileParseData:
        """
        Parse a single file and return structured parse data.

        If parsing fails for any reason the returned :class:`FileParseData` will
        contain an empty :class:`ParseResult` so that downstream phases can
        safely skip it.

        Args:
            file_path: Relative path to the file (used for identification).
            content: Raw source code of the file.
            language: Language identifier (e.g., "python", "typescript").

        Returns:
            A :class:`FileParseData` carrying the parse result.
        """
        try:
            parser = self._get_parser(language)
            result = parser.parse(content, file_path)
        except (RuntimeError, ValueError, SystemError, OSError):
            logger.warning(
                "Failed to parse %s (%s), skipping",
                file_path,
                language,
                exc_info=True,
            )
            result = ParseResult()

        return FileParseData(
            file_path=file_path,
            language=language,
            parse_result=result,
        )

    def _get_parser(self, language: str) -> LanguageParser:
        """
        Return the appropriate tree-sitter parser for *language*.

        Parser instances are cached per language to avoid repeated instantiation
        of tree-sitter ``Parser`` objects, which is expensive.

        Args:
            language: One of "python", "typescript", or "javascript".

        Returns:
            A :class:`LanguageParser` instance ready to parse source code.

        Raises:
            ValueError: If *language* is not supported.
        """
        # Return cached parser if available
        if cached := self._parser_cache.get(language):
            return cached
        with self._parser_cache_lock:
            if cached := self._parser_cache.get(language):
                return cached
            # Create new parser based on language
            parser = self._create_parser(language)
            # Cache the parser for future use
            self._parser_cache[language] = parser
        return parser

    def _create_parser(self, language: str) -> LanguageParser:
        """
        Create a new parser instance for the specified language.

        Args:
            language: The language to create a parser for.

        Returns:
            A new LanguageParser instance.

        Raises:
            ValueError: If the language is not supported.
        """
        if factory := self._PARSER_FACTORIES.get(language):
            return factory()

        # Unsupported language - raise error with helpful message
        details = f"Unsupported language {language!r}. Expected one of: {', '.join(self._PARSER_FACTORIES)}"
        raise ValueError(details)
