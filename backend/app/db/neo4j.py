import logging
from neo4j import AsyncGraphDatabase, AsyncDriver
from app.config import settings

logger = logging.getLogger("nexus.neo4j")

class Neo4jManager:
    def __init__(self):
        self.driver: AsyncDriver | None = None

    async def connect(self):
        """Initialize Neo4j Async Driver and verify index structures."""
        if self.driver is not None:
            return

        logger.info(f"Connecting to Neo4j at {settings.NEO4J_URI}")
        try:
            self.driver = AsyncGraphDatabase.driver(
                uri=settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
            )
            await self.driver.verify_connectivity()
            logger.info("Neo4j database connection verified.")
            await self.initialize_indexes()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise e

    async def disconnect(self):
        """Close Neo4j Driver Connection."""
        if self.driver is None:
            return

        logger.info("Closing Neo4j connection driver...")
        await self.driver.close()
        self.driver = None
        logger.info("Neo4j connection driver closed.")

    async def initialize_indexes(self):
        """Ensures uniqueness constraints exist on concepts and contradictions."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized.")

        # Constraints prevent duplicate entities and optimize index traversals
        queries = [
            "CREATE CONSTRAINT unique_concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT unique_contradiction_id IF NOT EXISTS FOR (cr:Contradiction) REQUIRE cr.id IS UNIQUE"
        ]
        
        async with self.driver.session() as session:
            for query in queries:
                try:
                    await session.run(query)
                except Exception as e:
                    logger.warning(f"Failed to verify constraint index in Neo4j: {e}")
        logger.info("Neo4j uniqueness constraint indexes initialized.")

    async def create_concept_node(self, concept_id: str, name: str, user_id: str, confidence: float = 1.0):
        """Creates a Concept node and links it to the User profile in Neo4j."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized.")

        query = """
        MERGE (u:User {id: $user_id})
        MERGE (c:Concept {id: $concept_id})
        ON CREATE SET c.name = $name, c.version = 1, c.valid_from = timestamp(), c.status = 'ACTIVE'
        MERGE (u)-[r:HELD_STATE]->(c)
        ON CREATE SET r.confidence = $confidence, r.created_at = timestamp()
        """
        
        async with self.driver.session() as session:
            await session.run(query, user_id=user_id, concept_id=concept_id, name=name, confidence=confidence)
            logger.debug(f"Created Concept node '{name}' in Neo4j linked to User '{user_id}'")

    async def create_memory_node(self, belief_id: str, statement: str, user_id: str):
        """Creates the parent Memory node and links it to the User profile."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized.")

        query = """
        MERGE (u:User {id: $user_id})
        MERGE (m:Memory {id: $belief_id})
        ON CREATE SET 
            m.statement = $statement, 
            m.timestamp = timestamp(), 
            m.status = 'ACTIVE'
        MERGE (u)-[r:RECALLS]->(m)
        ON CREATE SET r.confidence = 1.0, r.last_accessed = timestamp()
        """
        async with self.driver.session() as session:
            await session.run(query, user_id=user_id, belief_id=belief_id, statement=statement)
            logger.info(f"Created Memory node '{belief_id}' in Neo4j linked to User '{user_id}'")

    async def create_semantic_entities(self, belief_id: str, nodes: list[dict], relationships: list[dict]):
        """Creates extracted nodes and maps relationships linked to the Memory node."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized.")

        async with self.driver.session() as session:
            # 1. Merge the semantic entities
            for node in nodes:
                name = node.get("name")
                label = node.get("label")
                if not name or not label:
                    continue
                
                # Dynamic labels are resolved explicitly to avoid Cypher injection
                if label == "Project":
                    query = """
                    MATCH (m:Memory {id: $belief_id})
                    MERGE (e:Project {name: $name})
                    ON CREATE SET e.id = apoc.create.uuid(), e.created_at = timestamp()
                    MERGE (m)-[:EXTRACTED_ENTITY]->(e)
                    """
                elif label == "Technology":
                    query = """
                    MATCH (m:Memory {id: $belief_id})
                    MERGE (e:Technology {name: $name})
                    ON CREATE SET e.id = apoc.create.uuid(), e.created_at = timestamp()
                    MERGE (m)-[:EXTRACTED_ENTITY]->(e)
                    """
                else:
                    query = """
                    MATCH (m:Memory {id: $belief_id})
                    MERGE (e:Concept {name: $name})
                    ON CREATE SET e.id = apoc.create.uuid(), e.created_at = timestamp()
                    MERGE (m)-[:EXTRACTED_ENTITY]->(e)
                    """
                await session.run(query, belief_id=belief_id, name=name)

            # 2. Merge semantic relationships
            for rel in relationships:
                source = rel.get("source")
                target = rel.get("target")
                rel_type = rel.get("type")
                if not source or not target or not rel_type:
                    continue

                if rel_type == "BUILT_WITH":
                    query = """
                    MATCH (m:Memory {id: $belief_id})
                    MATCH (source {name: $source})<-[:EXTRACTED_ENTITY]-(m)
                    MATCH (target {name: $target})<-[:EXTRACTED_ENTITY]-(m)
                    MERGE (source)-[r:BUILT_WITH]->(target)
                    ON CREATE SET r.confidence = 1.0, r.created_at = timestamp()
                    """
                elif rel_type == "USES":
                    query = """
                    MATCH (m:Memory {id: $belief_id})
                    MATCH (source {name: $source})<-[:EXTRACTED_ENTITY]-(m)
                    MATCH (target {name: $target})<-[:EXTRACTED_ENTITY]-(m)
                    MERGE (source)-[r:USES]->(target)
                    ON CREATE SET r.confidence = 1.0, r.created_at = timestamp()
                    """
                else:
                    query = """
                    MATCH (m:Memory {id: $belief_id})
                    MATCH (source {name: $source})<-[:EXTRACTED_ENTITY]-(m)
                    MATCH (target {name: $target})<-[:EXTRACTED_ENTITY]-(m)
                    MERGE (source)-[r:RELATED_TO]->(target)
                    ON CREATE SET r.confidence = 1.0, r.created_at = timestamp()
                    """
                await session.run(query, belief_id=belief_id, source=source, target=target)
        logger.info(f"Entities and relations processed for Memory '{belief_id}' in Neo4j.")

    async def delete_memory_nodes(self, belief_id: str):
        """Detaches and deletes the parent Memory node during transaction rollback."""
        if self.driver is None:
            return

        query = """
        MATCH (m:Memory {id: $belief_id})
        DETACH DELETE m
        """
        async with self.driver.session() as session:
            await session.run(query, belief_id=belief_id)
            logger.info(f"Neo4j Memory node '{belief_id}' deleted for rollback.")

    async def get_adjacent_neighbors(self, concept_ids: list[str]) -> dict[str, list[str]]:
        """Traverses Neo4j to fetch connected concepts (up to 2 hops) for context expansion."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized.")
        if not concept_ids:
            return {}

        query = """
        MATCH (c:Concept) WHERE c.id IN $concept_ids
        OPTIONAL MATCH (c)-[*1..2]-(neighbor:Concept)
        WHERE neighbor.status = 'ACTIVE' AND neighbor.id <> c.id
        RETURN c.id AS source_id, collect(DISTINCT neighbor.name) AS neighbors
        """
        
        results = {}
        async with self.driver.session() as session:
            result = await session.run(query, concept_ids=concept_ids)
            async for record in result:
                results[record["source_id"]] = record["neighbors"]
        return results

    async def get_concept_centrality(self, concept_id: str) -> float:
        """Returns normalized degree centrality of a concept node in the graph."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized.")

        query = """
        MATCH (c:Concept {id: $concept_id})
        OPTIONAL MATCH (c)-[r]-()
        RETURN count(r) AS degree
        """
        
        async with self.driver.session() as session:
            result = await session.run(query, concept_id=concept_id)
            record = await result.single()
            if record:
                degree = record["degree"]
                # Normalize degree centrality for simple V1 (e.g. division by max degrees or simple sigmoid logic)
                # Max nodes is small in MVP, simple min-max scaling or sigmoidal normalization:
                import math
                return float(1.0 / (1.0 + math.exp(-degree / 2.0)) - 0.5) * 2.0
            return 0.0

    async def retrieve_memory_graph_context(self, belief_ids: list[str]) -> dict[str, dict]:
        """Traverses the Memory-Centric Graph layer to return primary entities and connected neighbors."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized.")
        if not belief_ids:
            return {}

        query = """
        MATCH (m:Memory) WHERE m.id IN $belief_ids AND m.status = 'ACTIVE'
        MATCH (m)-[:EXTRACTED_ENTITY]->(entity)
        OPTIONAL MATCH (entity)-[*1..2]-(neighbor)
        WHERE NOT neighbor:Memory AND (neighbor.status IS NULL OR neighbor.status = 'ACTIVE') AND neighbor <> entity
        RETURN 
            m.id AS belief_id,
            collect(DISTINCT {name: entity.name, label: labels(entity)[0]}) AS entities,
            collect(DISTINCT neighbor.name) AS neighbors
        """
        
        results = {}
        async with self.driver.session() as session:
            res = await session.run(query, belief_ids=belief_ids)
            async for record in res:
                results[record["belief_id"]] = {
                    "entities": record["entities"],
                    "neighbors": record["neighbors"]
                }
        return results

    async def create_contradiction_node(self, contradiction_id: str, category: str, severity: str, belief_ids: list[str]):
        """Creates a Contradiction junction node linking conflicting Memory nodes."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized.")

        query = """
        MERGE (cr:Contradiction {id: $contradiction_id})
        ON CREATE SET 
            cr.category = $category,
            cr.severity = $severity,
            cr.status = 'ACTIVE',
            cr.created_at = timestamp()
        WITH cr
        UNWIND $belief_ids AS b_id
        MATCH (m:Memory {id: b_id})
        MERGE (m)-[:CONFLICTS_WITH {confidence: 1.0}]->(cr)
        """
        async with self.driver.session() as session:
            await session.run(
                query, 
                contradiction_id=contradiction_id, 
                category=category, 
                severity=severity, 
                belief_ids=belief_ids
            )
            logger.info(f"Created Contradiction node '{contradiction_id}' linking memories: {belief_ids}")

    async def decay_recall_confidence(self, belief_ids: list[str], severity_score: float):
        """Reduces the User RECALLS Memory confidence edges based on conflict severity."""
        if self.driver is None:
            return

        query = """
        UNWIND $belief_ids AS b_id
        MATCH (u:User)-[r:RECALLS]->(m:Memory {id: b_id})
        SET r.confidence = r.confidence * (1.0 - 0.5 * $severity_score)
        """
        async with self.driver.session() as session:
            await session.run(query, belief_ids=belief_ids, severity_score=severity_score)
            logger.info(f"Decayed recall confidence for memories: {belief_ids}")

    async def resolve_contradiction_graph(
        self, contradiction_id: str, keep_belief_id: str, supersede_belief_id: str, transition_trigger: str
    ):
        """Resolves a contradiction in the graph by setting statuses and creating SUPERSEDES link."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized.")

        query = """
        // 1. Resolve contradiction node status
        MATCH (cr:Contradiction {id: $contradiction_id})
        SET cr.status = 'RESOLVED'
        
        // 2. Set superseded memory node status and expiration
        WITH cr
        MATCH (m_old:Memory {id: $supersede_belief_id})
        SET m_old.status = 'SUPERSEDED', m_old.valid_to = timestamp()
        
        // 3. Link new memory node to old memory node
        WITH cr, m_old
        MATCH (m_new:Memory {id: $keep_belief_id})
        MERGE (m_new)-[s:SUPERSEDES]->(m_old)
        ON CREATE SET s.transition_trigger = $transition_trigger, s.timestamp = timestamp()
        """
        async with self.driver.session() as session:
            await session.run(
                query, 
                contradiction_id=contradiction_id, 
                keep_belief_id=keep_belief_id, 
                supersede_belief_id=supersede_belief_id, 
                transition_trigger=transition_trigger
            )
            logger.info(f"Resolved contradiction graph: {contradiction_id}")

# Singleton instance
neo4j_db = Neo4jManager()


