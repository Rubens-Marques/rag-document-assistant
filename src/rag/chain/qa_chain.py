import os
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.rag.ingestion.loader import DocumentLoader
from src.rag.vectorstore.store import VectorStore


PROMPT_TEMPLATE = """Use o contexto abaixo para responder à pergunta do usuário.
Se não encontrar a resposta no contexto, diga explicitamente que não sabe.
Responda sempre em português.

Contexto:
{context}

Pergunta: {question}

Resposta:"""


class QAChain:
    def __init__(
        self,
        api_key: str,
        vectorstore_path: str,
        llm_model: str = "claude-3-5-haiku-20241022",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_chunks: int = 5,
    ):
        self.loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatAnthropic(model=llm_model, anthropic_api_key=api_key)
        self.vectorstore = VectorStore(self.embeddings, vectorstore_path)
        self.max_chunks = max_chunks
        self._chain = None

    def index(self, file_path: str) -> int:
        docs = self.loader.load(file_path)
        self.vectorstore.build(docs)
        self._build_chain()
        return len(docs)

    def load_existing(self) -> bool:
        loaded = self.vectorstore.load()
        if loaded:
            self._build_chain()
        return loaded

    def ask(self, question: str) -> str:
        if not self._chain:
            raise RuntimeError("Nenhum documento indexado. Chame index() ou load_existing() primeiro.")
        result = self._chain.invoke({"query": question})
        return result["result"]

    def _build_chain(self):
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )
        self._chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(k=self.max_chunks),
            chain_type_kwargs={"prompt": prompt},
        )
