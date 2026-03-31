import pytest
from pathlib import Path
from src.rag.ingestion.loader import DocumentLoader


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def test_load_txt_returns_documents():
    loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
    docs = loader.load(str(EXAMPLES_DIR / "sample.txt"))
    assert len(docs) > 0
    assert all(hasattr(d, "page_content") for d in docs)
    assert all(len(d.page_content) > 0 for d in docs)


def test_load_txt_chunks_respect_size():
    loader = DocumentLoader(chunk_size=200, chunk_overlap=20)
    docs = loader.load(str(EXAMPLES_DIR / "sample.txt"))
    for doc in docs:
        assert len(doc.page_content) <= 300  # margem para overlap


def test_load_nonexistent_file_raises():
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("/tmp/arquivo_que_nao_existe_xyz.txt")
