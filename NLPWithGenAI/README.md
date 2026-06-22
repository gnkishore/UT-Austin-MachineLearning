# NLP with GenAI — Medical-Diagnosis RAG Assistant

Notebook: `Full_Code_NLP_RAG_Project_Notebook_.ipynb`
Data: `data/medical_diagnosis_manual.pdf` (Merck Manual, ~20 MB, 4,000+ pages, 23 sections)

## Problem

Healthcare professionals face information overload at the point of care. The project builds a **Retrieval-Augmented Generation (RAG) assistant** over the Merck Manual that doctors, nurses, and clinical staff can query for diagnostic and treatment guidance — grounding LLM answers in an authoritative medical reference instead of relying on the LLM's parametric memory.

**Users:** healthcare professionals.
**Use cases:** diagnostic assistance, drug-information lookup, treatment-plan suggestions, specialty knowledge, critical-care protocols.

## Data

- **Source PDF:** `medical_diagnosis_manual.pdf` (Merck Manual) — ~20.1 MB, 4,000+ pages organized into 23 sections.
- **Loader:** LangChain `PyMuPDFLoader` (PyMuPDF 1.26.5).

## RAG pipeline architecture

| Stage | Choice |
|---|---|
| PDF loader | LangChain `PyMuPDFLoader` |
| Chunking | `RecursiveCharacterTextSplitter` (chunk size / overlap configured in notebook) |
| Embeddings | `SentenceTransformerEmbeddings` (sentence-transformers 5.1.1) |
| Vector store | Chroma (chromadb 1.1.1) |
| Retriever | Chroma `.as_retriever()`, **top-k = 3** |
| LLM | **Mistral 7B**, `gguf` format, downloaded via `hf_hub_download` |
| LLM runtime | `llama-cpp-python==0.2.28` |
| Inference params | `max_tokens=128`, `temperature=0`, `top_p=0.95`, `top_k=50` |
| Prompt template | System message + user-message template with `{context}` and `{question}` placeholders |

## Sample queries

The notebook fires five clinical queries at the assistant and compares three configurations: (a) LLM only, (b) LLM with prompt engineering, (c) LLM + RAG retrieval.

1. **Sepsis** — "What is the protocol for managing sepsis in a critical care unit?"
2. **Appendicitis** — symptoms, medical vs. surgical management.
3. **Alopecia** — treatments and causes of sudden patchy hair loss.
4. **Brain injury** — recommended treatments for physical brain tissue injury.
5. **Fractured leg** — precautions and treatment during a hiking trip.

## Evaluation

LLM-as-a-judge — the same Mistral 7B rates its own answers on two axes:

- **Groundedness** — is the answer supported by the retrieved context?
- **Relevance** — does the answer address the question?

Eval prompts (`groundedness_rater_system_message`, `relevance_rater_system_message`, `user_message_template`) are set up in the notebook. No external ground-truth dataset is used; scoring is purely model-judged.

## Conclusions

- The pipeline demonstrates the end-to-end RAG pattern over a real medical reference: load → chunk → embed → store → retrieve → generate → self-evaluate.
- Comparing the three configurations (no RAG / prompt-engineered / RAG) illustrates how retrieval reduces hallucination on domain-specific clinical questions.

## Recommendations / limitations

- **Top-k = 3 is tight** for multi-faceted clinical questions; consider sweeping k and/or adding a reranker.
- **LLM-as-judge is a weak proxy** for clinical correctness — pair with expert review before any real clinical use.
- **Chunk-size tuning** matters for long protocol passages (sepsis, brain injury) that may span chunk boundaries.
- **No fine-tuning of the LLM** — the assistant is purely retrieval-grounded, which is appropriate for safety but limits stylistic control.
- This is a **proof-of-concept**, not a clinical tool: deployment would require IRB / compliance review, citation surfacing back to the manual, and uncertainty disclosure to users.
