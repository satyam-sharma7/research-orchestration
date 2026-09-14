import uuid
from datetime import datetime
from pinecone import Pinecone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from backend.core.config import PINECONE_API_KEY


# ──────────────────────────────────────────────────────────────────────
# Pydantic models — these define the schema the AI agent must follow
# when calling each tool.
# ──────────────────────────────────────────────────────────────────────

class MemoryUpsert(BaseModel):
    """Schema for saving a single memory."""
    text: str = Field(
        description=(
            "A concise, standalone factual statement, preference, or goal to be"
            " saved in memory."
        )
    )
    category: str = Field(
        description=(
            "The classification type of the memory. Use one of: 'preference',"
            " 'fact', 'goal', or 'constraint'."
        )
    )
    entity_links: str = Field(
        description=(
            "Comma-separated list of primary entities, topics, or proper nouns"
            " related to this memory (e.g., 'Python, PostgreSQL, User')."
        )
    )


class MemoryQuery(BaseModel):
    """Schema for searching memories."""
    query: str = Field(
        description="The search query — a natural-language question or topic."
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "Optional filter. Only return memories of this category:"
            " 'preference', 'fact', 'goal', or 'constraint'."
        )
    )
    top_k: int = Field(
        default=5,
        description="Number of results to return (1–20)."
    )


class MemoryUpdate(BaseModel):
    """Schema for updating an existing memory by its ID."""
    memory_id: str = Field(
        description="The unique ID of the memory to update."
    )
    text: str = Field(
        description="The new text content to replace the old memory."
    )
    category: Optional[str] = Field(
        default=None,
        description="New category (leave empty to keep the current one)."
    )
    entity_links: Optional[str] = Field(
        default=None,
        description="New entity links (leave empty to keep the current ones)."
    )


class MemoryDelete(BaseModel):
    """Schema for deleting one or more memories by ID."""
    memory_ids: List[str] = Field(
        description="List of memory IDs to delete."
    )


# ──────────────────────────────────────────────────────────────────────
# Core class — one instance per user, safe for concurrent access
# because Pinecone's client is thread-safe and each user gets their
# own namespace, so there are no cross-user conflicts.
# ──────────────────────────────────────────────────────────────────────

class PineconeSemanticMemory:
    """Manages long-term semantic memory for a single user.

    Each user's vectors live in a separate Pinecone namespace (the user_id),
    so multiple users can read/write concurrently without conflicts.

    Usage:
        mem = PineconeSemanticMemory(user_id="abc-123")
        mem.save_memories([MemoryUpsert(text="...", category="fact", entity_links="Python")])
        results = mem.search_memories(MemoryQuery(query="What languages do I know?"))
        mem.update_memory(MemoryUpdate(memory_id="...", text="updated text"))
        mem.delete_memories(MemoryDelete(memory_ids=["..."]))
    """

    INDEX_NAME = "job-search-agent"
    EMBED_MODEL = "multilingual-e5-large"

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(self.INDEX_NAME)

    # ── helpers ───────────────────────────────────────────────────────

    def _embed(self, texts: List[str], input_type: str = "passage") -> List[List[float]]:
        """Generate embeddings for a list of texts using Pinecone Inference.

        Args:
            texts: The strings to embed.
            input_type: 'passage' when storing, 'query' when searching.

        Returns:
            A list of embedding vectors (list of floats).
        """
        response = self.pc.inference.embed(
            model=self.EMBED_MODEL,
            inputs=texts,
            parameters={"input_type": input_type},
        )
        return [item.values for item in response]

    @staticmethod
    def _timestamp() -> int:
        """Integer timestamp in YYYYMMDDHHmmss format for sorting."""
        return int(datetime.now().strftime("%Y%m%d%H%M%S"))

    # ── SAVE (upsert new memories) ────────────────────────────────────

    def save_memories(self, content: List[MemoryUpsert]) -> str:
        """Save one or more new memories to the vector store.

        Each memory is embedded, assigned a unique ID, and stored with its
        full text in metadata so it can be retrieved later.

        Args:
            content: List of MemoryUpsert objects.

        Returns:
            A confirmation string with the count of saved memories.
        """
        if not content:
            return "No memories provided."

        texts = [item.text for item in content]
        embeddings = self._embed(texts, input_type="passage")

        vectors: List[Dict[str, Any]] = []
        created_ids: List[str] = []

        for i, embedding in enumerate(embeddings):
            vec_id = str(uuid.uuid4())
            created_ids.append(vec_id)
            vectors.append({
                "id": vec_id,
                "values": embedding,
                "metadata": {
                    "user_id": self.user_id,
                    "text": content[i].text,            # store the full text
                    "category": content[i].category,
                    "entity_links": content[i].entity_links,
                    "created_at": self._timestamp(),
                    "updated_at": "",
                },
            })

        try:
            self.index.upsert(vectors=vectors, namespace=self.user_id)
            return (
                f"Successfully saved {len(vectors)} memory(ies). "
                f"IDs: {', '.join(created_ids)}"
            )
        except Exception as e:
            return f"Error saving memories: {e}"

    # ── SEARCH (semantic retrieval) ───────────────────────────────────

    def search_memories(self, params: MemoryQuery) -> List[Dict[str, Any]]:
        """Search memories by semantic similarity.

        Embeds the query with input_type='query', then finds the closest
        vectors in this user's namespace.

        Args:
            params: MemoryQuery with the search query, optional category
                    filter, and top_k.

        Returns:
            A list of dicts, each containing:
              - id: the memory's unique ID
              - score: cosine similarity (0–1, higher = more relevant)
              - text: the original saved text
              - category: preference / fact / goal / constraint
              - entity_links: related entities
              - created_at: when it was saved
        """
        top_k = max(1, min(params.top_k, 20))  # clamp to [1, 20]

        query_embedding = self._embed([params.query], input_type="query")[0]

        # Build optional metadata filter
        filter_dict = None
        if params.category:
            filter_dict = {"category": {"$eq": params.category}}

        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.user_id,
                include_metadata=True,
                filter=filter_dict,
            )
        except Exception as e:
            return [{"error": f"Search failed: {e}"}]

        memories = []
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            memories.append({
                "id": match["id"],
                "score": round(match["score"], 4),
                "text": meta.get("text", ""),
                "category": meta.get("category", ""),
                "entity_links": meta.get("entity_links", ""),
                "created_at": meta.get("created_at", ""),
            })

        return memories

    # ── UPDATE (overwrite an existing memory) ─────────────────────────

    def update_memory(self, params: MemoryUpdate) -> str:
        """Update the text (and optionally category/entity_links) of an
        existing memory.

        How it works:
          1. Fetch the old vector to get its current metadata.
          2. Re-embed the new text.
          3. Upsert the vector with the same ID, merging old + new metadata.

        Args:
            params: MemoryUpdate with the memory_id and new values.

        Returns:
            A confirmation or error string.
        """
        # 1. Fetch the existing vector's metadata
        try:
            fetch_result = self.index.fetch(
                ids=[params.memory_id],
                namespace=self.user_id,
            )
        except Exception as e:
            return f"Error fetching memory: {e}"

        vectors = fetch_result.get("vectors", {})
        if params.memory_id not in vectors:
            return f"Memory '{params.memory_id}' not found."

        old_meta = vectors[params.memory_id].get("metadata", {})

        # 2. Re-embed the new text
        new_embedding = self._embed([params.text], input_type="passage")[0]

        # 3. Merge metadata — keep old values for fields not provided
        updated_meta = {
            "user_id": self.user_id,
            "text": params.text,
            "category": params.category if params.category else old_meta.get("category", ""),
            "entity_links": params.entity_links if params.entity_links else old_meta.get("entity_links", ""),
            "created_at": old_meta.get("created_at", ""),
            "updated_at": self._timestamp(),
        }

        try:
            self.index.upsert(
                vectors=[{
                    "id": params.memory_id,
                    "values": new_embedding,
                    "metadata": updated_meta,
                }],
                namespace=self.user_id,
            )
            return f"Memory '{params.memory_id}' updated successfully."
        except Exception as e:
            return f"Error updating memory: {e}"

    # ── DELETE ─────────────────────────────────────────────────────────

    def delete_memories(self, params: MemoryDelete) -> str:
        """Delete one or more memories by their IDs.

        Args:
            params: MemoryDelete with a list of memory IDs.

        Returns:
            A confirmation or error string.
        """
        if not params.memory_ids:
            return "No memory IDs provided."

        try:
            self.index.delete(
                ids=params.memory_ids,
                namespace=self.user_id,
            )
            return f"Deleted {len(params.memory_ids)} memory(ies)."
        except Exception as e:
            return f"Error deleting memories: {e}"

    # ── DELETE ALL (clear a user's entire memory) ─────────────────────

    def delete_all_memories(self) -> str:
        """Delete ALL memories for this user (wipes the namespace).

        Returns:
            A confirmation or error string.
        """
        try:
            self.index.delete(delete_all=True, namespace=self.user_id)
            return f"All memories for user '{self.user_id}' have been deleted."
        except Exception as e:
            return f"Error clearing memories: {e}"


# ──────────────────────────────────────────────────────────────────────
# Tool factory — creates LangChain tools bound to a specific user.
#
# Call `create_memory_tools(user_id)` once per user session. Each tool
# is bound to that user's namespace, so concurrent users each get their
# own isolated set of tools.
# ──────────────────────────────────────────────────────────────────────

def create_memory_tools(user_id: str) -> List[StructuredTool]:
    """Build the four semantic-memory LangChain tools for a given user.

    Args:
        user_id: The authenticated user's ID.

    Returns:
        A list of StructuredTool instances:
          - save_memories
          - search_memories
          - update_memory
          - delete_memories
    """
    mem = PineconeSemanticMemory(user_id=user_id)

    save_tool = StructuredTool.from_function(
        func=mem.save_memories,
        name="save_memories",
        description=(
            "Save new information to long-term memory. Use this when the user"
            " shares a preference, fact, goal, or constraint that should be"
            " remembered across sessions. Accepts a list of memories."
        ),
    )

    search_tool = StructuredTool.from_function(
        func=mem.search_memories,
        name="search_memories",
        description=(
            "Search long-term memory by semantic similarity. Use this to recall"
            " facts, preferences, goals, or constraints the user previously"
            " shared. Returns the most relevant matches with similarity scores."
        ),
    )

    update_tool = StructuredTool.from_function(
        func=mem.update_memory,
        name="update_memory",
        description=(
            "Update an existing memory. Use when the user corrects or refines"
            " something previously saved. Requires the memory's ID (from a"
            " prior search) and the new text."
        ),
    )

    delete_tool = StructuredTool.from_function(
        func=mem.delete_memories,
        name="delete_memories",
        description=(
            "Delete one or more memories by ID. Use when the user asks to"
            " forget something. Requires a list of memory IDs (from a prior"
            " search)."
        ),
    )

    return [save_tool, search_tool, update_tool, delete_tool]
