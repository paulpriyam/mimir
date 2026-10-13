import numpy as np

from mimir.vectors import cosine_scores, meaning, normalize, rrf


def test_normalize_unit_rows():
    matrix = normalize(np.array([[3.0, 4.0], [0.0, 0.0]], np.float32))
    assert np.allclose(matrix[0], [0.6, 0.8])
    assert np.allclose(matrix[1], [0.0, 0.0])  # zero rows are left alone


def test_cosine_ranks_the_aligned_row_first():
    matrix = normalize(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], np.float32))
    scores = cosine_scores(matrix, np.array([1.0, 0.0], np.float32))
    assert scores.argmax() == 0


def test_meaning_puts_each_modality_on_one_scale():
    # text scores sit higher than image scores; after `meaning` the standout is the image.
    scores = np.array([0.30, 0.28, 0.26, 0.10, 0.80], np.float32)
    modality = np.array(["text", "text", "text", "image", "image"], object)
    z = meaning(scores, modality)
    assert z.argmax() == 4


def test_rrf_prefers_items_ranked_high_in_either_list():
    fused = rrf([["a", "b"], ["b", "a"]])
    assert fused["a"] == fused["b"]
    assert fused["a"] > rrf([["a", "b"], ["c", "d"]])["a"]
