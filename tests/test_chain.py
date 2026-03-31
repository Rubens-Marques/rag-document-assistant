import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.rag.chain.qa_chain import QAChain


EXAMPLES_DIR = str(Path(__file__).parent.parent / "examples")


def test_qa_chain_initializes():
    with patch("src.rag.chain.qa_chain.OpenAIEmbeddings"), \
         patch("src.rag.chain.qa_chain.ChatAnthropic"):
        chain = QAChain(api_key="test", vectorstore_path="/tmp/test_vs")
    assert chain is not None


def test_qa_chain_index_document():
    mock_store = MagicMock()
    mock_store.build = MagicMock()
    with patch("src.rag.chain.qa_chain.OpenAIEmbeddings"), \
         patch("src.rag.chain.qa_chain.ChatAnthropic"), \
         patch("src.rag.chain.qa_chain.VectorStore", return_value=mock_store), \
         patch("src.rag.chain.qa_chain.RetrievalQA"):
        chain = QAChain(api_key="test", vectorstore_path="/tmp/test_vs")
        chain.index(str(Path(EXAMPLES_DIR) / "sample.txt"))
    mock_store.build.assert_called_once()
