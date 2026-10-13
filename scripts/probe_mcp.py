"""Drive the real MCP server over stdio, exactly as Claude or Cursor would.

    python scripts/probe_mcp.py [index_dir]

Proves the whole path: the process starts, both tools are registered, a query crosses
modalities, and a query nothing resembles comes back empty rather than as a weak guess.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

QUERIES = [
    "where they talk about the flight delay",
    "a penguin",
    "INV-20931",
    "ZQ-7734",
    "quantum blockchain yoga",
]


async def run(index_dir: str) -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mimir", "--index-dir", index_dir, "mcp"],
        env={**os.environ},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("tools:", [tool.name for tool in tools.tools])
            for query in QUERIES:
                result = await session.call_tool("mimir_search", {"query": query, "limit": 2})
                hits = (result.structured_content or {}).get("result", [])
                print(f"\n{query!r}")
                if not hits:
                    print("   nothing found")
                for hit in hits:
                    where = f" @{hit['loc']:.0f}" if "loc" in hit else ""
                    print(f"   {hit['score']:5.2f}  {hit['kind']:<6} {hit['path']}{where}")
    return 0


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / ".mimir")
    sys.exit(asyncio.run(run(directory)))
