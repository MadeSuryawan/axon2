"""
Phase 2: Structure processing for Axon.

Takes a list of file entries (path, content, language) and populates the
knowledge graph with File and Folder nodes connected by CONTAINS relationships.
"""

from logging import getLogger
from pathlib import PurePosixPath

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.ingestion.walker import FileEntry

logger = getLogger(__name__)


class Structure:
    """
    Handles structure processing for the Axon knowledge graph.

    Processes file entries and creates a hierarchical representation
    with File and Folder nodes connected by CONTAINS relationships.
    The main entry point is :meth:`Structure.process_structure`.
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        """
        Initialize the Structure processor.

        Args:
            graph: The knowledge graph to populate with structure nodes
                and relationships.
        """
        self._graph = graph

    def process_structure(self, files: list[FileEntry]) -> None:
        """
        Build File/Folder nodes and CONTAINS relationships from a list of files.

        This is the main entry point. It:
        1. Collects all unique folder paths from file paths.
        2. Creates Folder nodes for each unique directory.
        3. Creates File nodes for each file entry.
        4. Creates CONTAINS relationships for folder hierarchy.
        5. Creates CONTAINS relationships for folder-to-file connections.

        Args:
            files: File entries to process. Each entry carries the relative path,
                raw content, and detected language.
        """
        # Step 1: Collect all unique folder paths from the file list
        folder_paths = self._collect_folder_paths(files)
        logger.debug("Collected %d unique folder paths", len(folder_paths))

        # Step 2: Create Folder nodes for each unique directory
        self._create_folder_nodes(folder_paths)

        # Step 3: Create File nodes for each file entry
        self._create_file_nodes(files)

    def _collect_folder_paths(self, files: list[FileEntry]) -> set[str]:
        """
        Extract all unique folder paths from the given file list.

        For each file entry, extracts all parent directories from the path.
        The current directory (".") is excluded as it represents root.

        Args:
            files: List of file entries to extract folder paths from.

        Returns:
            A set of unique folder path strings.
        """
        return {
            parent_str
            for file_info in files
            for parent in PurePosixPath(file_info.path).parents
            if (parent_str := f"{parent}") != "."
        }

    def _create_folder_nodes(self, folder_paths: set[str]) -> None:
        """
        Create Folder nodes in the graph for each unique folder path.

        Checks if a folder node already exists before creating to avoid
        duplicates. Each folder node stores the folder name and its
        full path.

        Args:
            folder_paths: Set of folder path strings to create nodes for.
        """
        for dir_path in folder_paths:
            # Only create if it doesn't already exist
            if self._graph.get_node(folder_id := generate_id(NodeLabel.FOLDER, dir_path)):
                continue

            self._graph.add_node(
                GraphNode(
                    id=folder_id,
                    label=NodeLabel.FOLDER,
                    name=PurePosixPath(dir_path).name,
                    file_path=dir_path,
                ),
            )
            self._add_relationship(f"{PurePosixPath(dir_path).parent}", folder_id)

    def _create_file_nodes(self, files: list[FileEntry]) -> None:
        """
        Create File nodes in the graph for each file entry.

        Each file node stores the file name, full path, content, and
        detected language.

        Args:
            files: List of file entries to create nodes for.
        """
        for file_info in files:
            file_id = generate_id(NodeLabel.FILE, file_info.path)
            self._graph.add_node(
                GraphNode(
                    id=file_id,
                    label=NodeLabel.FILE,
                    name=PurePosixPath(file_info.path).name,
                    file_path=file_info.path,
                    content=file_info.content,
                    language=file_info.language,
                ),
            )
            self._add_relationship(f"{PurePosixPath(file_info.path).parent}", file_id)

    def _add_relationship(self, parent_str: str, target_id: str) -> None:
        """Add a CONTAINS relationship from the source to the target."""
        if parent_str == ".":
            return

        parent_id = generate_id(NodeLabel.FOLDER, parent_str)
        self._graph.add_relationship(
            GraphRelationship(
                id=f"contains:{parent_id}->{target_id}",
                type=RelType.CONTAINS,
                source=parent_id,
                target=target_id,
            ),
        )
