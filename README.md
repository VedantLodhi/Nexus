# NexusOS MVP - Infrastructure Core (Week 1)

This repository contains the initialized, simplified infrastructure configuration for **NexusOS**, a Cognitive Digital Twin Operating System.

---

## Prerequisites
Ensure the following tools are installed on your host system:
* **Docker** & **Docker Compose**
* **Python 3.10+** (with `pip` and virtual environment support)
* **Node.js 18+** & **npm**

---

## Database Container Mesh Setup
To spin up the multi-database cluster (PostgreSQL, Qdrant, and Neo4j):

```bash
# Start all containers in detached mode
docker compose up -d

# Verify all containers are healthy
docker compose ps
```

### Exponent Database Endpoints
* **PostgreSQL**: `localhost:5432` (Username: `postgres`, DB: `nexus_db`)
* **Neo4j Console**: `http://localhost:7474` (Bolt: `localhost:7687`, Auth: `neo4j/neo4j_secure_password`)
* **Qdrant Console**: `http://localhost:6333` (gRPC: `localhost:6334`)

---

## FastAPI Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a python virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   * **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   * **Linux/macOS**:
     ```bash
     source venv/bin/activate
     ```

4. Install the backend package dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the FastAPI application locally:
   ```bash
   uvicorn main:app --reload
   ```

The backend API is now running locally at `http://localhost:8000`.

### Health Diagnostics Endpoint
Validate database connections by opening `http://localhost:8000/api/v1/health` in your browser. This endpoint runs active check queries across all three databases.

---

## Next.js Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install the client packages:
   ```bash
   npm install
   ```

3. Start the Next.js local development server:
   ```bash
   npm run dev
   ```

The client application is now running locally at `http://localhost:3000`.
