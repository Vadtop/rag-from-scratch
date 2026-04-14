import pytest
import json


def test_vector_store_add_and_count():
    import chromadb
    from vector_store import VectorStore

    client = chromadb.Client()
    collection = client.create_collection("test_count_vs")

    with pytest.MonkeyPatch.context() as m:
        m.setattr(VectorStore, "__init__", lambda self, name="test": None)
        store = VectorStore.__new__(VectorStore)
        store.client = client
        store.collection = collection

    store.add_chunk(
        "chunk_1",
        "test content",
        [0.1] * 384,
        {"source": "test.txt", "chunk_id": 0, "total_chunks": 1},
    )
    assert store.count() == 1


def test_vector_store_search():
    import chromadb
    from vector_store import VectorStore

    client = chromadb.Client()
    collection = client.create_collection("test_search_vs")

    with pytest.MonkeyPatch.context() as m:
        m.setattr(VectorStore, "__init__", lambda self, name="test": None)
        store = VectorStore.__new__(VectorStore)
        store.client = client
        store.collection = collection

    store.add_chunk(
        "c1",
        "RAG uses vector databases",
        [0.5] * 384,
        {"source": "doc1.txt", "chunk_id": 0, "total_chunks": 1},
    )
    store.add_chunk(
        "c2",
        "Python is a programming language",
        [0.1] * 384,
        {"source": "doc2.txt", "chunk_id": 0, "total_chunks": 1},
    )

    results = store.search([0.5] * 384, top_k=1)
    assert len(results["documents"][0]) == 1
    assert "RAG" in results["documents"][0][0]


def test_vector_store_clear():
    import chromadb
    from vector_store import VectorStore

    client = chromadb.Client()
    collection = client.create_collection("test_clear_vs2")

    with pytest.MonkeyPatch.context() as m:
        m.setattr(VectorStore, "__init__", lambda self, name="test": None)
        store = VectorStore.__new__(VectorStore)
        store.client = client
        store.collection = collection

    store.add_chunk(
        "chunk_1",
        "test",
        [0.1] * 384,
        {"source": "test.txt", "chunk_id": 0, "total_chunks": 1},
    )
    assert store.count() == 1
    store.clear()
    assert store.count() == 0


def test_vector_store_get_all_sources():
    import chromadb
    from vector_store import VectorStore

    client = chromadb.Client()
    collection = client.create_collection("test_sources_vs2")

    with pytest.MonkeyPatch.context() as m:
        m.setattr(VectorStore, "__init__", lambda self, name="test": None)
        store = VectorStore.__new__(VectorStore)
        store.client = client
        store.collection = collection

    store.add_chunk(
        "c1",
        "text1",
        [0.1] * 384,
        {"source": "a.txt", "chunk_id": 0, "total_chunks": 2},
    )
    store.add_chunk(
        "c2",
        "text2",
        [0.2] * 384,
        {"source": "b.txt", "chunk_id": 1, "total_chunks": 2},
    )

    sources = store.get_all_sources()
    assert set(sources) == {"a.txt", "b.txt"}


def test_format_schema():
    from huggingface_rag import _format_schema

    schema = {
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "score": {"type": "number"},
            "active": {"type": "boolean"},
            "tags": {"type": "array"},
        }
    }
    result = _format_schema(schema)
    parsed = json.loads(result)
    assert parsed["name"] == ""
    assert parsed["age"] == 0
    assert parsed["score"] == 0.0
    assert parsed["active"] is False
    assert parsed["tags"] == []


def test_parse_json_valid():
    from huggingface_rag import _parse_json_response

    result = _parse_json_response('{"name": "test", "value": 42}')
    assert result["name"] == "test"
    assert result["value"] == 42


def test_parse_json_with_noise():
    from huggingface_rag import _parse_json_response

    result = _parse_json_response(
        'Here is the result: {"name": "test", "value": 42} done.'
    )
    assert result["name"] == "test"


def test_parse_json_invalid():
    from huggingface_rag import _parse_json_response

    result = _parse_json_response("no json here at all")
    assert "parse_error" in result


def test_parse_json_with_markdown():
    from huggingface_rag import _parse_json_response

    result = _parse_json_response('```json\n{"key": "val"}\n```')
    assert result["key"] == "val"
