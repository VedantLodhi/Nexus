import logging
import asyncpg
from app.config import settings

logger = logging.getLogger("nexus.postgres")

class PostgresManager:
    def __init__(self):
        self.pool: asyncpg.Pool | None = None

    async def connect(self):
        """Initialize Postgres Connection Pool and build default tables."""
        if self.pool is not None:
            return

        dsn = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        logger.info(f"Connecting to PostgreSQL at {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")
        
        try:
            self.pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=2,
                max_size=10
            )
            logger.info("PostgreSQL connection pool established.")
            await self.initialize_tables()
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise e

    async def disconnect(self):
        """Close Postgres Connection Pool."""
        if self.pool is None:
            return

        logger.info("Closing PostgreSQL connection pool...")
        await self.pool.close()
        self.pool = None
        logger.info("PostgreSQL connection pool closed.")

    async def initialize_tables(self):
        """Creates mandatory system schemas on startup if they do not exist."""
        if self.pool is None:
            raise RuntimeError("Postgres connection pool not initialized.")

        create_tables_sql = """
        -- Dynamic user cognitive profiles
        CREATE TABLE IF NOT EXISTS cognitive_profiles (
            user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            forgetting_rate DOUBLE PRECISION DEFAULT 0.05,
            epistemic_threshold DOUBLE PRECISION DEFAULT 0.60,
            lexical_style JSONB NOT NULL DEFAULT '{}'::jsonb,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Event store logging raw user interactions (Episodic Logs)
        CREATE TABLE IF NOT EXISTS cognitive_events (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES cognitive_profiles(user_id) ON DELETE CASCADE,
            event_type VARCHAR(50) NOT NULL,
            statement TEXT NOT NULL,
            metadata JSONB,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Historical Contradiction telemetry
        CREATE TABLE IF NOT EXISTS contradictions_history (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES cognitive_profiles(user_id) ON DELETE CASCADE,
            concept_a_id VARCHAR(255) NOT NULL,
            concept_b_id VARCHAR(255) NOT NULL,
            severity VARCHAR(50) NOT NULL,
            resolution_status VARCHAR(50) DEFAULT 'ACTIVE',
            resolution_details TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP WITH TIME ZONE
        );

        -- Recall & Access Metric Logs (Access Tracking)
        CREATE TABLE IF NOT EXISTS memory_access_logs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            belief_id UUID NOT NULL REFERENCES cognitive_events(id) ON DELETE CASCADE,
            accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
            retrieved_score DOUBLE PRECISION NOT NULL,
            reinforced_strength DOUBLE PRECISION NOT NULL
        );
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(create_tables_sql)
        logger.info("PostgreSQL database schemas verified/initialized.")


    async def get_or_create_user(self) -> str:
        """Retrieves or creates a default user profile to support single-user local MVP operation."""
        if self.pool is None:
            raise RuntimeError("Postgres connection pool not initialized.")

        async with self.pool.acquire() as conn:
            user_id = await conn.fetchval("SELECT user_id FROM cognitive_profiles LIMIT 1")
            if not user_id:
                user_id = await conn.fetchval(
                    "INSERT INTO cognitive_profiles DEFAULT VALUES RETURNING user_id"
                )
            return str(user_id)

    async def create_episodic_log(self, user_id: str, event_type: str, statement: str, metadata: dict = None) -> str:
        """Logs an interaction or statement in the episodic table."""
        if self.pool is None:
            raise RuntimeError("Postgres connection pool not initialized.")

        import json
        metadata_json = json.dumps(metadata) if metadata else None
        
        async with self.pool.acquire() as conn:
            log_id = await conn.fetchval(
                """
                INSERT INTO cognitive_events (user_id, event_type, statement, metadata)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                user_id, event_type, statement, metadata_json
            )
            return str(log_id)

    async def get_memory_metadata(self, belief_ids: list[str]) -> dict[str, dict]:
        """Fetches access metadata (frequency and elapsed time since last event) for ranking."""
        if self.pool is None:
            raise RuntimeError("Postgres connection pool not initialized.")
        if not belief_ids:
            return {}

        results = {}
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, timestamp, 
                       (EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - timestamp))) AS elapsed_seconds,
                       (SELECT COUNT(*) FROM memory_access_logs WHERE belief_id = id) AS access_frequency
                FROM cognitive_events
                WHERE id = ANY($1::uuid[])
                """,
                belief_ids
            )
            for row in rows:
                results[str(row["id"])] = {
                    "timestamp": row["timestamp"],
                    "elapsed_seconds": float(row["elapsed_seconds"]),
                    "frequency": int(row["access_frequency"]) + 1 # Add 1 to represent initial ingestion
                }
        return results


    async def delete_episodic_log(self, log_id: str):
        """Deletes a raw episodic log by ID (used for transaction rollbacks)."""
        if self.pool is None:
            return
        
        import uuid
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM cognitive_events WHERE id = $1", uuid.UUID(log_id))
            logger.info(f"PostgreSQL raw log '{log_id}' deleted for rollback.")

    async def log_memory_recall(self, belief_id: str, retrieved_score: float, reinforced_strength: float):
        """Logs a recall access event to PostgreSQL for tracking decay schedules."""
        if self.pool is None:
            return

        import uuid
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory_access_logs (belief_id, retrieved_score, reinforced_strength)
                VALUES ($1, $2, $3)
                """,
                uuid.UUID(belief_id), retrieved_score, reinforced_strength
            )
            logger.info(f"Logged memory recall event in Postgres for belief: {belief_id}")

    async def create_contradiction_log(self, user_id: str, concept_a_id: str, concept_b_id: str, severity: str) -> str:
        """Logs a contradiction event in PostgreSQL."""
        if self.pool is None:
            raise RuntimeError("Postgres connection pool not initialized.")

        import uuid
        async with self.pool.acquire() as conn:
            log_id = await conn.fetchval(
                """
                INSERT INTO contradictions_history (user_id, concept_a_id, concept_b_id, severity, resolution_status)
                VALUES ($1, $2, $3, $4, 'ACTIVE')
                RETURNING id
                """,
                uuid.UUID(user_id), concept_a_id, concept_b_id, severity
            )
            return str(log_id)

    async def resolve_contradiction_log(self, contradiction_id: str, details: str):
        """Updates a contradiction log to mark it resolved."""
        if self.pool is None:
            return

        import uuid
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE contradictions_history
                SET resolution_status = 'RESOLVED', resolution_details = $1, resolved_at = CURRENT_TIMESTAMP
                WHERE id = $2
                """,
                details, uuid.UUID(contradiction_id)
            )
            logger.info(f"Updated PostgreSQL contradiction log '{contradiction_id}' to RESOLVED.")

# Singleton instance
postgres_db = PostgresManager()




