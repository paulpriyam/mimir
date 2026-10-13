"""The web UI's API surface. The page is served from the same MimirSearch the MCP tools use,
so these tests cover the JSON contract the page depends on."""

from pathlib import Path

import pytest

from mimir.cli import main
from mimir.embed import FakeEmbedder
from mimir.mcp_server import MimirSearch
from mimir.ui import create_app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client(tmp_path):
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient

    index_dir = tmp_path / "index"
    assert main(["--fake", "--index-dir", str(index_dir), "index", str(FIXTURES)]) == 0
    search = MimirSearch(index_dir, embedder=FakeEmbedder())
    with TestClient(create_app(search)) as test_client:
        yield test_client
    search.close()


def test_serves_the_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "mimir" in response.text
    assert "/api/search" in response.text


def test_info_reports_the_index(client):
    payload = client.get("/api/info").json()
    assert payload["files"] == 4
    assert payload["segments"] >= 4
    assert payload["deep_available"] in (True, False)


def test_search_returns_results_with_an_absolute_path(client):
    results = client.get("/api/search", params={"q": "INV-20931"}).json()["results"]
    assert results
    assert results[0]["path"] == "notes/invoice.txt"
    assert Path(results[0]["absolute_path"]).is_absolute()


def test_an_empty_query_returns_nothing_rather_than_everything(client):
    assert client.get("/api/search", params={"q": "   "}).json()["results"] == []


def test_a_query_nothing_matches_returns_nothing(client):
    assert client.get("/api/search", params={"q": "quantum blockchain yoga"}).json()["results"] == []


def test_open_refuses_a_path_that_does_not_exist(client):
    # Deliberately a missing path: a real one would launch an app during the test run.
    response = client.post("/api/open", params={"path": "/nope/missing.pdf"})
    assert response.json()["ok"] is False
