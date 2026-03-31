# rag-document-assistant

> Faça perguntas em linguagem natural sobre seus documentos internos usando RAG + Claude.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat)
![Claude](https://img.shields.io/badge/Claude_API-Anthropic-8B5CF6?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

## Sobre

Sistema RAG (Retrieval-Augmented Generation) que permite conversar com documentos internos — manuais, contratos, relatórios, políticas. O modelo não alucina: só responde com base no que está nos documentos.

**Problema resolvido:** Encontrar informação em documentos internos é lento e manual. Com este assistente, qualquer pessoa pode perguntar em linguagem natural e obter a resposta com a fonte.

## Como funciona

```
Documento PDF/TXT
      │
      ▼
 [Chunking]           Divide em pedaços menores
      │
      ▼
 [Embeddings]         Transforma em vetores semânticos
      │
      ▼
 [FAISS Index]        Armazena localmente
      │
 Pergunta do usuário
      │
      ▼
 [Busca semântica]    Encontra os chunks mais relevantes
      │
      ▼
 [Claude API]         Gera resposta baseada no contexto
      │
      ▼
    Resposta
```

## Stack

- **LangChain** — orquestração do pipeline RAG
- **Claude API (Anthropic)** — geração de respostas
- **FAISS** — busca vetorial local (sem servidor externo)
- **OpenAI Embeddings** — vetorização de texto
- **PyPDF** — leitura de PDFs

## Instalação

```bash
git clone https://github.com/Rubens-Marques/rag-document-assistant
cd rag-document-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # adicionar ANTHROPIC_API_KEY e OPENAI_API_KEY
```

## Como usar

```bash
# Indexar um documento
python cli.py index examples/sample.txt

# Chat interativo
python cli.py

# Via API (opcional)
uvicorn src.rag.main:app --reload
```

## Testes

```bash
pytest tests/ -v
```

## Licença

MIT © Rubens Marques
