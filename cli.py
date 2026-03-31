#!/usr/bin/env python3
"""CLI para interagir com o RAG Document Assistant."""
import os
import sys
from dotenv import load_dotenv
from src.rag.chain.qa_chain import QAChain

load_dotenv()


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    vectorstore_path = os.getenv("VECTORSTORE_PATH", "./data/vectorstore")

    if not api_key:
        print("Erro: ANTHROPIC_API_KEY não configurada no .env")
        sys.exit(1)

    chain = QAChain(api_key=api_key, vectorstore_path=vectorstore_path)

    if len(sys.argv) == 3 and sys.argv[1] == "index":
        file_path = sys.argv[2]
        print(f"Indexando {file_path}...")
        n = chain.index(file_path)
        print(f"✓ {n} chunks indexados com sucesso.")
        return

    if not chain.load_existing():
        print("Nenhum documento indexado. Use: python cli.py index <arquivo>")
        sys.exit(1)

    print("RAG Document Assistant — Digite 'sair' para encerrar\n")
    while True:
        question = input("Pergunta: ").strip()
        if question.lower() in ("sair", "exit", "quit"):
            break
        if not question:
            continue
        print(f"\nResposta: {chain.ask(question)}\n")


if __name__ == "__main__":
    main()
