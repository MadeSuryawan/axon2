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

        # Step 4: Create folder hierarchy relationships (parent contains child)
        self._create_folder_hierarchy(folder_paths)

        # Step 5: Create folder-to-file relationships
        self._create_folder_file_relationships(files)

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
        folder_paths: set[str] = set()

        # Iterate through each file and collect all parent directories
        for file_info in files:
            pure = PurePosixPath(file_info.path)
            for parent in pure.parents:
                parent_str = str(parent)
                if parent_str == ".":
                    # Skip the current directory marker
                    continue
                folder_paths.add(parent_str)

        return folder_paths

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
            # Generate a unique ID for this folder
            folder_id = generate_id(NodeLabel.FOLDER, dir_path)

            # Only create if it doesn't already exist
            if self._graph.get_node(folder_id) is None:
                self._graph.add_node(
                    GraphNode(
                        id=folder_id,
                        label=NodeLabel.FOLDER,
                        name=PurePosixPath(dir_path).name,
                        file_path=dir_path,
                    ),
                )

    def _create_file_nodes(self, files: list[FileEntry]) -> None:
        """
        Create File nodes in the graph for each file entry.

        Each file node stores the file name, full path, content, and
        detected language.

        Args:
            files: List of file entries to create nodes for.
        """
        for file_info in files:
            # Generate a unique ID for this file
            file_id = generate_id(NodeLabel.FILE, file_info.path)

            # Create the file node with all metadata
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

    def _create_folder_hierarchy(self, folder_paths: set[str]) -> None:
        """
        Create CONTAINS relationships representing folder hierarchy.

        For each folder, creates a relationship from its parent folder
        to the child folder. Root-level folders (with no parent) are skipped.

        Args:
            folder_paths: Set of folder path strings to create relationships for.
        """
        for dir_path in folder_paths:
            pure = PurePosixPath(dir_path)
            parent_str = str(pure.parent)

            if parent_str == ".":
                # Top-level folder has no parent — skip creating relationship
                continue

            # Generate IDs for parent and child folders
            parent_id = generate_id(NodeLabel.FOLDER, parent_str)
            child_id = generate_id(NodeLabel.FOLDER, dir_path)

            # Create the CONTAINS relationship
            rel_id = f"contains:{parent_id}->{child_id}"
            self._graph.add_relationship(
                GraphRelationship(
                    id=rel_id,
                    type=RelType.CONTAINS,
                    source=parent_id,
                    target=child_id,
                ),
            )

    def _create_folder_file_relationships(self, files: list[FileEntry]) -> None:
        """
        Create CONTAINS relationships from folders to files.

        For each file, creates a relationship from its immediate parent
        folder to the file. Root-level files (with no parent folder) are skipped.

        Args:
            files: List of file entries to create relationships for.
        """
        for file_info in files:
            pure = PurePosixPath(file_info.path)
            parent_str = str(pure.parent)

            if parent_str == ".":
                # Root-level file — no containing folder exists
                continue

            # Generate IDs for parent folder and file
            parent_id = generate_id(NodeLabel.FOLDER, parent_str)
            file_id = generate_id(NodeLabel.FILE, file_info.path)

            # Create the CONTAINS relationship
            rel_id = f"contains:{parent_id}->{file_id}"
            self._graph.add_relationship(
                GraphRelationship(
                    id=rel_id,
                    type=RelType.CONTAINS,
                    source=parent_id,
                    target=file_id,
                ),
            )
