import uuid
from datetime import datetime
from pinecone import Pinecone
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from backend.core.config import PINECONE_API_KEY

# ──────────────────────────────────────────────────────────────────────
# Pydantic models — Schemas the AI agent must follow for episodic memory
# ──────────────────────────────────────────────────────────────────────

class EpisodeUpsert(BaseModel):
    """Schema for recording a past event or task (an episode)."""
    event_summary: str = Field(
        description="A summary of the event or task (e.g., 'Debugged a Redis connection issue')."
    )
    action_taken: str = Field(
        description="What action was taken during this event? (e.g., 'Rewrote the Redis client to use asyncio')."
    )
    outcome: str = Field(
        description="The final result (e.g., 'Performance improved, bugs resolved')."
    )


class EpisodeSearch(BaseModel):
    """Schema for searching past episodes by semantic similarity."""
    query: str = Field(
        description="The search query (e.g., 'Have we ever fixed a bug with Redis?')"
    )
    top_k: int = Field(
        default=5,
        description="Number of past episodes to return."
    )


# ──────────────────────────────────────────────────────────────────────
# Core class — Episodic Memory
# ──────────────────────────────────────────────────────────────────────

class PineconeEpisodicMemory:
    """Manages Episodic Memory (records of past events/tasks) for a user.

    Uses a separate Pinecone namespace (user_id + '_episodes') to ensure
    past events don't get mixed up with general semantic facts.
    """

    INDEX_NAME = "job-search-agent"
    EMBED_MODEL = "multilingual-e5-large"

    def __init__(self, user_id: str):
        self.user_id = user_id
        # We append '_episodes' so it stays separate from semantic memory!
        self.namespace = f"{user_id}_episodes"
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(self.INDEX_NAME)

    def _embed(self, texts: List[str], input_type: str = "passage") -> List[List[float]]:
        response = self.pc.inference.embed(
            model=self.EMBED_MODEL,
            inputs=texts,
            parameters={"input_type": input_type},
        )
        return [item.values for item in response]

    @staticmethod
    def _timestamp() -> int:
        return int(datetime.now().strftime("%Y%m%d%H%M%S"))

    def record_episode(self, episode: EpisodeUpsert) -> str:
        """Saves a new episodic memory (an experience) to the vector store."""

        # We embed a combination of the summary, action, and outcome so the whole
        # story is mathematically searchable.
        combined_text = f"Event: {episode.event_summary} | Action: {episode.action_taken} | Outcome: {episode.outcome}"
        embedding = self._embed([combined_text], input_type="passage")[0]

        vec_id = str(uuid.uuid4())

        vector = {
            "id": vec_id,
            "values": embedding,
            "metadata": {
                "user_id": self.user_id,
                "event_summary": episode.event_summary,
                "action_taken": episode.action_taken,
                "outcome": episode.outcome,
                "created_at": self._timestamp(),
            },
        }

        try:
            self.index.upsert(vectors=[vector], namespace=self.namespace)
            return f"Successfully recorded episode with ID: {vec_id}"
        except Exception as e:
            return f"Error recording episode: {e}"

    def search_past_episodes(self, params: EpisodeSearch) -> List[Dict[str, Any]]:
        """Search past events to remember what was previously done."""
        top_k = max(1, min(params.top_k, 20))
        query_embedding = self._embed([params.query], input_type="query")[0]

        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True,
            )
        except Exception as e:
            return [{"error": f"Search failed: {e}"}]

        episodes = []
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            episodes.append({
                "id": match["id"],
                "score": round(match["score"], 4),
                "event_summary": meta.get("event_summary", ""),
                "action_taken": meta.get("action_taken", ""),
                "outcome": meta.get("outcome", ""),
                "timestamp": meta.get("created_at", ""),
            })

        return episodes


# ──────────────────────────────────────────────────────────────────────
# Tool factory — Creates the LangChain tools for Episodic Memory
# ──────────────────────────────────────────────────────────────────────

def create_episodic_tools(user_id: str) -> List[StructuredTool]:
    """Build the episodic-memory LangChain tools for a given user."""
    mem = PineconeEpisodicMemory(user_id=user_id)

    record_tool = StructuredTool.from_function(
        func=mem.record_episode,
        name="record_episode",
        description=(
            "Record a past event, finished task, or significant action. Use this "
            "when you complete a task for the user, so you can remember what you "
            "did and what the outcome was for future reference."
        ),
    )

    search_tool = StructuredTool.from_function(
        func=mem.search_past_episodes,
        name="search_past_episodes",
        description=(
            "Search the user's past events and tasks. Use this to remember if you "
            "have previously solved a specific problem, what approach you took, "
            "and what the result was."
        ),
    )

    return [record_tool, search_tool]
