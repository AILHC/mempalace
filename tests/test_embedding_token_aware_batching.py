import pytest

np = pytest.importorskip("numpy")

import mempalace.embedding as embedding  # noqa: E402


class _Enc:
    def __init__(self, ids):
        self.ids = list(ids)
        self.attention_mask = [1] * len(self.ids)


class _TokenAwareTokenizer:
    def __init__(self):
        self.calls = []

    def encode_batch(self, texts):
        self.calls.append(list(texts))
        encs = []
        for text in texts:
            marker = int(text.rsplit("#", 1)[1])
            # Lengths intentionally differ from input order, and marker stays
            # in the first token so the fake session can prove output order.
            encs.append(_Enc([marker] * marker))
        return encs


class _RecordingSession:
    def __init__(self):
        self.batch_markers = []

    def run(self, _output_names, feed):
        markers = [int(row[0]) for row in feed["input_ids"]]
        self.batch_markers.append(markers)
        sent = np.zeros((len(markers), embedding._EMBEDDINGGEMMA_DIM), dtype=np.float32)
        for row, marker in enumerate(markers):
            sent[row, marker] = 1.0
        last_hidden = np.zeros(
            (len(markers), feed["input_ids"].shape[1], embedding._EMBEDDINGGEMMA_DIM),
            dtype=np.float32,
        )
        return [last_hidden, sent]


class _OomOnceSession(_RecordingSession):
    def __init__(self, fail_on_batch_size):
        super().__init__()
        self.fail_on_batch_size = fail_on_batch_size
        self.failed = False

    def run(self, output_names, feed):
        if not self.failed and feed["input_ids"].shape[0] == self.fail_on_batch_size:
            self.failed = True
            raise RuntimeError("HRESULT 8007000E: out of memory")
        return super().run(output_names, feed)


def _install_loaded_ef(monkeypatch, ef, tokenizer, session):
    ef._tokenizer = tokenizer
    ef._session = session
    ef._np = np
    ef._output_idx = 1

    def already_loaded(self):
        return None

    monkeypatch.setattr(embedding.EmbeddinggemmaONNX, "_lazy_load", already_loaded)


def _marker_values(vectors):
    return [int(np.argmax(row)) for row in vectors]


def test_plan_token_batches_sorts_by_length_and_respects_limits():
    batches = embedding._plan_token_batches([9, 2, 5, 4, 3], max_batch_size=2, token_budget=10)

    assert batches == [[1, 4], [3, 2], [0]]


def test_plan_token_batches_respects_token_budget_before_item_limit():
    batches = embedding._plan_token_batches([4, 4, 4, 4], max_batch_size=4, token_budget=12)

    assert batches == [[0, 1, 2], [3]]


def test_plan_token_batches_allows_single_input_over_budget():
    batches = embedding._plan_token_batches([100, 2, 3], max_batch_size=8, token_budget=16)

    assert batches == [[1, 2], [0]]


def test_embeddinggemma_tokenizes_once_packs_by_length_and_restores_input_order(monkeypatch):
    tokenizer = _TokenAwareTokenizer()
    session = _RecordingSession()
    ef = embedding.EmbeddinggemmaONNX(
        preferred_providers=["CPUExecutionProvider"],
        batch_size=3,
        token_budget=12,
    )
    _install_loaded_ef(monkeypatch, ef, tokenizer, session)

    vectors = ef(["doc #5", "doc #1", "doc #4", "doc #2", "doc #3"])

    assert len(tokenizer.calls) == 1
    assert session.batch_markers == [[1, 2, 3], [4, 5]]
    assert _marker_values(vectors) == [5, 1, 4, 2, 3]


def test_embeddinggemma_order_with_repeated_empty_and_mixed_lengths(monkeypatch):
    class MixedTokenizer(_TokenAwareTokenizer):
        def encode_batch(self, texts):
            self.calls.append(list(texts))
            mapping = {
                "task: sentence similarity | query: ": [10],
                "task: sentence similarity | query: repeat": [20, 20],
                "task: sentence similarity | query: long #6": [60] * 6,
                "task: sentence similarity | query: short #1": [30],
                "task: sentence similarity | query: repeat again": [40, 40],
            }
            return [_Enc(mapping[text]) for text in texts]

    class MarkerSession(_RecordingSession):
        def run(self, _output_names, feed):
            markers = [int(row[0]) for row in feed["input_ids"]]
            self.batch_markers.append(markers)
            sent = np.zeros((len(markers), embedding._EMBEDDINGGEMMA_DIM), dtype=np.float32)
            for row, marker in enumerate(markers):
                sent[row, marker // 10] = 1.0
            last_hidden = np.zeros(
                (len(markers), feed["input_ids"].shape[1], embedding._EMBEDDINGGEMMA_DIM),
                dtype=np.float32,
            )
            return [last_hidden, sent]

    tokenizer = MixedTokenizer()
    session = MarkerSession()
    ef = embedding.EmbeddinggemmaONNX(batch_size=2, token_budget=8)
    _install_loaded_ef(monkeypatch, ef, tokenizer, session)

    vectors = ef(["repeat", "", "long #6", "short #1", "repeat again"])

    assert session.batch_markers == [[10, 30], [20, 40], [60]]
    assert [int(np.argmax(row)) for row in vectors] == [2, 1, 6, 3, 4]


def test_directml_oom_halves_effective_budget_and_replans_pending_inputs(monkeypatch):
    tokenizer = _TokenAwareTokenizer()
    session = _OomOnceSession(fail_on_batch_size=4)
    ef = embedding.EmbeddinggemmaONNX(
        preferred_providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        batch_size=4,
        token_budget=16,
    )
    _install_loaded_ef(monkeypatch, ef, tokenizer, session)
    monkeypatch.setattr(ef, "_reset_session", lambda: None)

    vectors = ef(["doc #1", "doc #2", "doc #3", "doc #4"])

    assert ef._token_budget == 16
    assert ef._effective_token_budget == 8
    assert session.batch_markers == [[1, 2], [3, 4]]
    assert len(vectors) == 4


def test_non_directml_oom_is_not_retried(monkeypatch):
    tokenizer = _TokenAwareTokenizer()
    session = _OomOnceSession(fail_on_batch_size=2)
    ef = embedding.EmbeddinggemmaONNX(
        preferred_providers=["CPUExecutionProvider"],
        batch_size=2,
        token_budget=8,
    )
    _install_loaded_ef(monkeypatch, ef, tokenizer, session)

    with pytest.raises(RuntimeError, match="8007000E"):
        ef(["doc #1", "doc #2"])

    assert ef._effective_token_budget == 8


def test_single_item_directml_oom_raises_original_error(monkeypatch):
    tokenizer = _TokenAwareTokenizer()
    session = _OomOnceSession(fail_on_batch_size=1)
    ef = embedding.EmbeddinggemmaONNX(
        preferred_providers=["DmlExecutionProvider"],
        batch_size=2,
        token_budget=1,
    )
    _install_loaded_ef(monkeypatch, ef, tokenizer, session)

    with pytest.raises(RuntimeError, match="8007000E"):
        ef(["doc #4"])

    assert ef._effective_token_budget == 1


def test_unicode_decode_error_is_not_directml_oom():
    exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "out of memory")

    assert not embedding._is_directml_oom(exc, ["CPUExecutionProvider"])


def test_directml_unicode_decode_error_is_retryable_for_multi_item_batch(monkeypatch):
    class UnicodeDecodeThenRecordSession(_RecordingSession):
        def __init__(self):
            super().__init__()
            self.failed = False

        def run(self, output_names, feed):
            if not self.failed and feed["input_ids"].shape[0] == 2:
                self.failed = True
                raise UnicodeDecodeError("utf-8", b"\xc4", 0, 1, "invalid continuation byte")
            return super().run(output_names, feed)

    tokenizer = _TokenAwareTokenizer()
    session = UnicodeDecodeThenRecordSession()
    ef = embedding.EmbeddinggemmaONNX(
        preferred_providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        batch_size=2,
        token_budget=8,
    )
    _install_loaded_ef(monkeypatch, ef, tokenizer, session)
    monkeypatch.setattr(ef, "_reset_session", lambda: None)

    vectors = ef(["doc #3", "doc #3"])

    assert ef._effective_token_budget == 4
    assert session.batch_markers == [[3], [3]]
    assert len(vectors) == 2
