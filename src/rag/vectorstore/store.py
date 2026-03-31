from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class VectorStore:
    def __init__(self, embeddings, persist_path: str):
        self.embeddings = embeddings
        self.persist_path = Path(persist_path)
        self._store: FAISS | None = None

    def build(self, documents: list[Document]) -> None:
        self._store = FAISS.from_documents(documents, self.embeddings)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self.persist_path))

    def load(self) -> bool:
        if self.persist_path.exists():
            self._store = FAISS.load_local(
                str(self.persist_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            return True
        return False

    def as_retriever(self, k: int = 5):
        if not self._store:
            raise RuntimeError("VectorStore não inicializado. Chame build() ou load() primeiro.")
        return self._store.as_retriever(search_kwargs={"k": k})
