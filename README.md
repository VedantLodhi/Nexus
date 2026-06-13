# 🧠 NexusOS — Cognitive Digital Twin Operating System

> An AI-powered cognitive memory system that stores, retrieves, reasons over, and resolves contradictions in user knowledge using semantic memory, vector search, and knowledge graphs.

---

## 🚀 Overview

NexusOS is a next-generation Cognitive Digital Twin designed to mimic how human memory works.

The system combines:

* Semantic Memory Storage
* Vector Embeddings
* Knowledge Graphs
* Hybrid Retrieval
* Contradiction Detection
* Memory Resolution Workflows

By integrating PostgreSQL, Neo4j, Qdrant, and Local LLMs through Ollama, NexusOS creates a persistent AI memory layer capable of understanding relationships, recalling information, and detecting inconsistencies over time.

---

# 🏗️ Architecture

```text
User Input
     │
     ▼
Memory Agent
     │
     ▼
Entity Extraction (Ollama)
     │
     ▼
┌─────────────────┐
│   PostgreSQL    │
│ Memory Metadata │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│     Qdrant      │
│ Vector Memory   │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│      Neo4j      │
│ Knowledge Graph │
└─────────────────┘
     │
     ▼
Hybrid Retrieval Engine
     │
     ▼
Contradiction Engine
     │
     ▼
Resolution Engine
```

---

# ⚡ Tech Stack

## Backend

* FastAPI
* Python 3.12
* AsyncIO
* HTTPX
* Pydantic

## Databases

### PostgreSQL

Stores:

* User Metadata
* Memory Records
* Recall Logs
* Contradiction Records

### Neo4j

Stores:

* Knowledge Graph
* Entity Relationships
* Memory Connections

### Qdrant

Stores:

* Vector Embeddings
* Semantic Memory
* Similarity Search Indexes

## AI Layer

### Ollama

Models:

* nomic-embed-text
* qwen2.5:1.5b
* qwen3:8b

Capabilities:

* Embedding Generation
* Entity Extraction
* Reasoning
* Contradiction Analysis

## Frontend

* Next.js 15
* TypeScript
* Tailwind CSS
* shadcn/ui

---

# ✨ Implemented Features

### Cognitive Memory Engine

* Store user memories
* Semantic embedding generation
* Persistent memory storage

### Knowledge Graph

* Automatic entity extraction
* Relationship generation
* Neo4j graph construction

### Hybrid Retrieval

* Vector similarity search
* Graph-based retrieval
* Context enrichment

### Contradiction Detection

* Memory conflict discovery
* Severity scoring
* Resolution workflow

### Infrastructure

* Dockerized deployment
* Multi-database architecture
* Health monitoring endpoints

---

# 🐳 Infrastructure Setup

## Start Database Cluster

```bash
docker compose up -d
```

Verify services:

```bash
docker ps
```

---

## Service Endpoints

### PostgreSQL

```text
Host: localhost
Port: 5432
Database: nexus_db
Username: postgres
```

### Neo4j

```text
Console: http://localhost:7474
Bolt: localhost:7687
Username: neo4j
Password: neo4j_secure_password
```

### Qdrant

```text
Dashboard: http://localhost:6333/dashboard
REST API: localhost:6333
gRPC: localhost:6334
```

---

# 🔧 Backend Setup

```bash
cd backend
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Backend

```bash
python -m uvicorn main:app --reload
```

Backend URL:

```text
http://localhost:8000
```

---

# ❤️ Health Check

Verify all database integrations:

```text
http://localhost:8000/api/v1/health
```

Expected checks:

* PostgreSQL Connectivity
* Neo4j Connectivity
* Qdrant Connectivity

---

# 🎨 Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:3000
```

---

# 📈 Project Status

| Module                  | Status         |
| ----------------------- | -------------- |
| Infrastructure          | ✅ Complete     |
| Memory Engine           | ✅ Complete     |
| Knowledge Graph         | ✅ Complete     |
| Vector Search           | ✅ Complete     |
| Hybrid Retrieval        | ✅ Complete     |
| Contradiction Detection | ✅ Complete     |
| Frontend Integration    | 🚧 In Progress |
| Cognitive Dashboard     | 🚧 Planned     |
| Graph Visualization     | 🚧 Planned     |

---

# 🎯 Vision

NexusOS aims to become a fully-fledged Cognitive Digital Twin capable of remembering, reasoning, learning, and evolving with the user over time through persistent memory and graph-based intelligence.
