"""
Web Search Tool.

Provides web search capabilities with support for:
- Quick search: Single query, fast results
- Deep search: Multiple queries with LLM-generated sub-queries and summary synthesis

Supports multiple backends:
- Tavily (default): AI-optimized search API
- DuckDuckGo: Free fallback option
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from inception.tool.base import (
    Tool,
    ToolSpec,
    ToolResult,
    ParameterSpec,
    ParameterType,
    ReturnSpec,
)
from inception.provider.base import Message

if TYPE_CHECKING:
    from inception.config.settings import WebSearchConfig
    from inception.provider.base import BaseProvider

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    content: str
    score: float = 0.0
    published_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "score": self.score,
            "published_date": self.published_date,
        }


@dataclass
class SearchResponse:
    """Response from a search operation."""
    query: str
    mode: str
    total_results: int
    results: List[SearchResult] = field(default_factory=list)
    # Deep search specific
    sub_queries: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "query": self.query,
            "mode": self.mode,
            "total_results": self.total_results,
            "results": [r.to_dict() for r in self.results],
            "execution_time": self.execution_time,
        }
        if self.sub_queries:
            result["sub_queries"] = self.sub_queries
        if self.summary:
            result["summary"] = self.summary
        return result


class TavilyBackend:
    """Tavily search backend."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        include_content: bool = False,
        search_depth: str = "basic",
    ) -> List[SearchResult]:
        """
        Perform a search using Tavily API.

        Args:
            query: Search query
            max_results: Maximum number of results
            include_content: Whether to include full content
            search_depth: "basic" or "advanced"

        Returns:
            List of search results
        """
        import aiohttp

        url = f"{self.base_url}/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": include_content,
            "search_depth": search_depth,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Tavily API error: {response.status} - {error_text}")
                        return []

                    data = await response.json()
                    results = []
                    for item in data.get("results", []):
                        results.append(SearchResult(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            content=item.get("content", ""),
                            score=item.get("score", 0.0),
                            published_date=item.get("published_date"),
                        ))
                    return results

        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []


class DuckDuckGoBackend:
    """DuckDuckGo search backend (fallback)."""

    async def search(
        self,
        query: str,
        max_results: int = 5,
        region: str = "wt-wt",
        time_range: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Perform a search using DuckDuckGo.

        Args:
            query: Search query
            max_results: Maximum number of results
            region: Region code (default: worldwide)
            time_range: Time filter (d=day, w=week, m=month, y=year)

        Returns:
            List of search results
        """
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.error("duckduckgo-search not installed. Install with: uv pip install duckduckgo-search")
            return []

        try:
            # Run sync DuckDuckGo search in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._sync_search(query, max_results, region, time_range)
            )
            return results

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    def _sync_search(
        self,
        query: str,
        max_results: int,
        region: str,
        time_range: Optional[str],
    ) -> List[SearchResult]:
        """Synchronous search implementation."""
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            search_results = ddgs.text(
                query,
                region=region,
                timelimit=time_range,
                max_results=max_results,
            )
            for item in search_results:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", ""),
                    content=item.get("body", ""),
                    score=0.0,  # DuckDuckGo doesn't provide scores
                ))
        return results


class WebSearchTool(Tool):
    """
    Web search tool with quick and deep search modes.

    Quick search: Single query, returns top results
    Deep search: Generates sub-queries, searches in parallel, deduplicates, and synthesizes summary
    """

    def __init__(
        self,
        config: "WebSearchConfig",
        provider: Optional["BaseProvider"] = None,
    ):
        """
        Initialize the web search tool.

        Args:
            config: Web search configuration
            provider: LLM provider for deep search (sub-query generation and summary)
        """
        self._config = config
        self._provider = provider

        # Initialize backends
        self._tavily: Optional[TavilyBackend] = None
        self._duckduckgo: Optional[DuckDuckGoBackend] = None

        if config.tavily_api_key:
            self._tavily = TavilyBackend(config.tavily_api_key)
        self._duckduckgo = DuckDuckGoBackend()

        # Determine active backend
        if config.backend == "tavily" and self._tavily:
            self._active_backend = "tavily"
        else:
            self._active_backend = "duckduckgo"
            if config.backend == "tavily" and not self._tavily:
                logger.warning("Tavily API key not configured, falling back to DuckDuckGo")

        self._spec = ToolSpec(
            name="web_search",
            description=(
                "Search the web for information. Supports two modes:\n"
                "- quick: Fast single-query search for immediate results\n"
                "- deep: Multi-query search with AI-generated sub-queries and comprehensive summary\n\n"
                "Use 'quick' mode for simple factual queries.\n"
                "Use 'deep' mode for complex topics requiring comprehensive research."
            ),
            parameters={
                "query": ParameterSpec(
                    name="query",
                    type=ParameterType.STRING,
                    description="The search query",
                    required=True,
                ),
                "mode": ParameterSpec(
                    name="mode",
                    type=ParameterType.STRING,
                    description="Search mode: 'quick' for fast results, 'deep' for comprehensive research",
                    required=False,
                    default="quick",
                    enum=["quick", "deep"],
                ),
                "max_results": ParameterSpec(
                    name="max_results",
                    type=ParameterType.INTEGER,
                    description="Maximum number of results to return",
                    required=False,
                    default=5,
                ),
                "include_content": ParameterSpec(
                    name="include_content",
                    type=ParameterType.BOOLEAN,
                    description="Whether to include full page content (Tavily only)",
                    required=False,
                    default=False,
                ),
                "language": ParameterSpec(
                    name="language",
                    type=ParameterType.STRING,
                    description="Language code for search results (e.g., 'en', 'zh')",
                    required=False,
                ),
                "time_range": ParameterSpec(
                    name="time_range",
                    type=ParameterType.STRING,
                    description="Time filter: 'day', 'week', 'month', 'year' (DuckDuckGo only)",
                    required=False,
                    enum=["day", "week", "month", "year"],
                ),
            },
            returns=ReturnSpec(
                type=ParameterType.OBJECT,
                description="Search results with query, mode, total_results, results array, and optionally sub_queries and summary for deep search",
            ),
            category="search",
            tags=["web", "search", "research"],
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute web search."""
        query = kwargs.get("query", "")
        mode = kwargs.get("mode", "quick")
        max_results = kwargs.get("max_results", self._config.default_max_results)
        include_content = kwargs.get("include_content", False)
        language = kwargs.get("language", self._config.default_language)
        time_range = kwargs.get("time_range")

        if not query:
            return ToolResult.fail("No search query provided")

        start_time = time.time()

        try:
            if mode == "deep":
                response = await self._deep_search(
                    query=query,
                    max_results=max_results,
                    include_content=include_content,
                    language=language,
                    time_range=time_range,
                )
            else:
                response = await self._quick_search(
                    query=query,
                    max_results=max_results,
                    include_content=include_content,
                    language=language,
                    time_range=time_range,
                )

            response.execution_time = time.time() - start_time
            return ToolResult.ok(result=response.to_dict())

        except Exception as e:
            logger.exception(f"Web search failed: {e}")
            return ToolResult.fail(f"Search failed: {e}")

    async def _quick_search(
        self,
        query: str,
        max_results: int,
        include_content: bool,
        language: str,
        time_range: Optional[str],
    ) -> SearchResponse:
        """Perform a quick single-query search."""
        results = await self._single_search(
            query=query,
            max_results=max_results,
            include_content=include_content,
            time_range=time_range,
        )

        return SearchResponse(
            query=query,
            mode="quick",
            total_results=len(results),
            results=results,
        )

    async def _deep_search(
        self,
        query: str,
        max_results: int,
        include_content: bool,
        language: str,
        time_range: Optional[str],
    ) -> SearchResponse:
        """
        Perform a deep search with sub-queries and summary synthesis.

        Steps:
        1. Generate sub-queries using LLM
        2. Search all queries in parallel
        3. Deduplicate and rank results
        4. Synthesize comprehensive summary
        """
        # Use more results for deep search
        deep_max_results = self._config.deep_search_max_results

        # Step 1: Generate sub-queries
        sub_queries = await self._generate_sub_queries(query, language)
        all_queries = [query] + sub_queries

        # Step 2: Search all queries in parallel
        search_tasks = [
            self._single_search(
                query=q,
                max_results=deep_max_results,
                include_content=include_content,
                time_range=time_range,
            )
            for q in all_queries
        ]
        all_results = await asyncio.gather(*search_tasks)

        # Step 3: Flatten and deduplicate results
        unique_results = self._deduplicate_results(all_results)

        # Sort by score (if available) and limit results
        unique_results.sort(key=lambda x: x.score, reverse=True)
        unique_results = unique_results[:max_results]

        # Step 4: Synthesize summary
        summary = await self._synthesize_summary(query, unique_results, language)

        return SearchResponse(
            query=query,
            mode="deep",
            total_results=len(unique_results),
            results=unique_results,
            sub_queries=sub_queries,
            summary=summary,
        )

    async def _single_search(
        self,
        query: str,
        max_results: int,
        include_content: bool,
        time_range: Optional[str],
    ) -> List[SearchResult]:
        """Perform a single search using the active backend."""
        if self._active_backend == "tavily" and self._tavily:
            return await self._tavily.search(
                query=query,
                max_results=max_results,
                include_content=include_content,
                search_depth="advanced" if include_content else "basic",
            )
        else:
            # Map time_range to DuckDuckGo format
            ddg_time = None
            if time_range:
                time_map = {"day": "d", "week": "w", "month": "m", "year": "y"}
                ddg_time = time_map.get(time_range)

            return await self._duckduckgo.search(
                query=query,
                max_results=max_results,
                time_range=ddg_time,
            )

    async def _generate_sub_queries(self, query: str, language: str) -> List[str]:
        """Generate sub-queries for deep search using LLM."""
        if not self._provider:
            logger.warning("No LLM provider available for sub-query generation")
            return []

        system_prompt = (
            "You are a search query expansion assistant. Given a main search query, "
            "generate 3-5 related sub-queries that would help gather comprehensive information "
            "about the topic. Each sub-query should explore a different aspect or angle.\n\n"
            "Output only the sub-queries, one per line, without numbering or bullet points."
        )

        user_prompt = f"Main query: {query}\n\nLanguage: {language}\n\nGenerate sub-queries:"

        try:
            response = await self._provider.complete(
                messages=[
                    Message.system(system_prompt),
                    Message.user(user_prompt),
                ],
                temperature=0.7,
                max_tokens=500,
            )

            # Parse sub-queries from response
            lines = response.content.strip().split("\n")
            sub_queries = [
                line.strip().lstrip("0123456789.-) ")
                for line in lines
                if line.strip() and len(line.strip()) > 3
            ]
            return sub_queries[:5]  # Limit to 5 sub-queries

        except Exception as e:
            logger.error(f"Failed to generate sub-queries: {e}")
            return []

    def _deduplicate_results(
        self,
        all_results: List[List[SearchResult]],
    ) -> List[SearchResult]:
        """Deduplicate results from multiple searches based on URL."""
        seen_urls = set()
        unique_results = []

        for results in all_results:
            for result in results:
                # Create URL hash for deduplication
                url_hash = hashlib.md5(result.url.encode()).hexdigest()
                if url_hash not in seen_urls:
                    seen_urls.add(url_hash)
                    unique_results.append(result)

        return unique_results

    async def _synthesize_summary(
        self,
        query: str,
        results: List[SearchResult],
        language: str,
    ) -> Optional[str]:
        """Synthesize a comprehensive summary from search results."""
        if not self._provider:
            logger.warning("No LLM provider available for summary synthesis")
            return None

        if not results:
            return None

        # Build context from search results
        context_parts = []
        for i, result in enumerate(results[:10], 1):  # Limit to top 10 results
            context_parts.append(
                f"[{i}] {result.title}\n"
                f"URL: {result.url}\n"
                f"Content: {result.content[:500]}..."
            )

        context = "\n\n".join(context_parts)

        system_prompt = (
            "You are a research assistant. Based on the search results provided, "
            "synthesize a comprehensive and well-organized summary that answers the user's query. "
            "Include key findings, cite sources by their number [1], [2], etc., "
            "and highlight any conflicting information if present."
        )

        user_prompt = (
            f"Query: {query}\n\n"
            f"Language for response: {language}\n\n"
            f"Search Results:\n{context}\n\n"
            "Please provide a comprehensive summary:"
        )

        try:
            response = await self._provider.complete(
                messages=[
                    Message.system(system_prompt),
                    Message.user(user_prompt),
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            return response.content

        except Exception as e:
            logger.error(f"Failed to synthesize summary: {e}")
            return None
