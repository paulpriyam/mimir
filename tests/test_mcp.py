"""The MCP surface: what a client (Claude, Cursor) can ask for. The transport is thin; the
logic worth testing is MimirSearch, so it is tested with the fake embedder and no model."""

from pathlib import Path

import pytest

from mimir.cli import main
from mimir.config import MAIN_DB
from mimir.embed import FakeEmbedder
from mimir.mcp_server import MimirSearch, hit_to_dict
from mimir.store import Store

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def search(tmp_path):
    index_dir = tmp_path / "index"
    assert main(["--fake", "--index-dir", str(index_dir), "index", str(FIXTURES)]) == 0
    searcher = MimirSearch(index_dir, embedder=FakeEmbedder())
    yield searcher
    searcher.close()


def test_search_returns_relative_paths_and_a_location(search):
    results = search.search("INV-20931", limit=3)
    assert results
    assert results[0]["path"] == "notes/invoice.txt"
    assert Path(results[0]["absolute_path"]).is_absolute()
    assert results[0]["kind"] == "doc"


def test_search_is_empty_rather_than_a_weak_guess(search):
    assert search.search("quantum blockchain yoga") == []


def test_read_returns_the_indexed_text(search):
    text = search.read(str(FIXTURES / "notes" / "lease.txt"))
    assert "pets" in text
    assert search.read("/nope/missing.txt") == ""


def test_hit_to_dict_keeps_a_page_or_moment():
    from mimir.search import Hit

    hit = Hit(path="/x/a.pdf", kind="pdf", score=1.5, z=3.0, cosine=0.8, keyword_rank=None,
              segment_id=1, loc=6.0, loc_end=None, excerpt="fire drill", modified=0.0, size=1)
    result = hit_to_dict(hit, Path("/x"))
    assert result["loc"] == 6.0 and result["path"] == "a.pdf"


def test_server_registers_both_tools(tmp_path):
    pytest.importorskip("mcp")
    from mimir.mcp_server import build_server

    index_dir = tmp_path / "index"
    main(["--fake", "--index-dir", str(index_dir), "index", str(FIXTURES)])
    server, search = build_server(index_dir)
    try:
        import asyncio

        tools = asyncio.run(server.list_tools())
        assert {tool.name for tool in tools} == {"mimir_search", "mimir_read"}
    finally:
        search.close()
