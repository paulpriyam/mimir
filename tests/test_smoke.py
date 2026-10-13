from pathlib import Path

from mimir.cli import main
from mimir.config import MAIN_DB
from mimir.embed import FakeEmbedder
from mimir.evals import TagRow, evaluate
from mimir.search import Searcher
from mimir.store import Store

FIXTURES = Path(__file__).parent / "fixtures"
EVAL_FILE = Path(__file__).parents[1] / "evals" / "smoke.json"


def build_index(tmp_path: Path) -> Store:
    index_dir = tmp_path / "index"
    assert main(["--fake", "--index-dir", str(index_dir), "index", str(FIXTURES)]) == 0
    return Store(index_dir / MAIN_DB)


def top_hit(searcher: Searcher, query: str) -> str | None:
    vector = FakeEmbedder().embed_text([query])[0]
    hits = searcher.search(query, query_vector=vector, limit=5)
    return Path(hits[0].path).name if hits else None


def test_index_stores_every_file(tmp_path):
    store = build_index(tmp_path)
    files, segments = store.count()
    store.close()
    assert files == 4
    assert segments >= 4


def test_exact_lookup_and_word_search(tmp_path):
    store = build_index(tmp_path)
    searcher = Searcher(store)
    assert top_hit(searcher, "INV-20931") == "invoice.txt"
    assert top_hit(searcher, "exponential backoff") == "retry.txt"
    store.close()


def test_eval_harness_reports_positives_and_negatives(tmp_path):
    store = build_index(tmp_path)
    report = evaluate(store, FakeEmbedder(), EVAL_FILE)
    store.close()
    assert report.rows["exact"].at1 == 1
    assert report.rows["word"].at1 == 1
    assert report.rows["ALL"].false_positives == 0


def test_eval_harness_separates_empty_misses_from_wrong_ones():
    # A miss where nothing came back is a gate or retrieval problem; a miss where something
    # else came first is a ranking problem. They are counted apart because the fixes differ.
    row = TagRow()
    row.add(None, negative=False, returned_any=False)   # found nothing
    row.add(None, negative=False, returned_any=True)    # found the wrong thing
    row.add(1, negative=False, returned_any=True)       # correct
    assert row.empty_misses == 1
    assert row.wrong_misses == 1
    assert row.at1 == 1
    assert row.positives == 3
