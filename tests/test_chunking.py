from app.core.chunking import chunk_text


def test_empty_text_yields_no_chunks():
    assert chunk_text("", chunk_tokens=50, overlap=10) == []
    assert chunk_text("   \n\n  ", chunk_tokens=50, overlap=10) == []


def test_short_text_single_chunk():
    chunks = chunk_text("hello world foo bar", chunk_tokens=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].text == "hello world foo bar"


def test_overlap_preserves_boundary_tokens():
    words = " ".join(str(i) for i in range(100))
    chunks = chunk_text(words, chunk_tokens=30, overlap=10)
    assert len(chunks) > 1
    # Consecutive chunks share `overlap` tokens.
    first_tail = chunks[0].text.split()[-10:]
    second_head = chunks[1].text.split()[:10]
    assert first_tail == second_head


def test_no_chunk_exceeds_budget():
    words = " ".join(str(i) for i in range(500))
    chunks = chunk_text(words, chunk_tokens=64, overlap=16)
    assert all(c.token_estimate <= 64 for c in chunks)


def test_indices_are_sequential():
    words = " ".join(str(i) for i in range(200))
    chunks = chunk_text(words, chunk_tokens=25, overlap=5)
    assert [c.index for c in chunks] == list(range(len(chunks)))
