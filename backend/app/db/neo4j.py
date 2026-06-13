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

    async def create_semantic_entities(self, belief_id: str, nodes: list[dict], relationships: list[dict]) -> dict:
        """Creates extracted nodes and maps relationships linked to the Memory node."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized.")

        nodes_processed = 0
        nodes_created_db = 0
        relationships_processed = 0
        relationships_created_db = 0

        label_map = {
            "Language": "Technology",
            "Framework": "Technology",
            "Library": "Technology",
            "Database": "Technology",
            "Tool": "Technology",
            "Platform": "Technology",

            "Technology": "Technology",
            "Project": "Project",
            "Skill": "Skill",
            "Career": "Career",
            "Company": "Company",
            "Person": "Person",
            "Concept": "Concept"
        }

        async with self.driver.session() as session:
            # 1. Merge the semantic entities
            for node in nodes:
                name = node.get("name")
                raw_label = node.get("label", "Concept")
                label = label_map.get(raw_label, "Concept")

                if not name:
                    continue
                
                # Normalize label to Concept if not in valid ontology values
                if label not in label_map.values():
                    logger.warning(f"Label '{label}' is not in valid ontology. Mapping to 'Concept'.")
                    label = "Concept"

                # Dynamic labels are resolved explicitly to avoid Cypher injection
                query = f"""
                MATCH (m:Memory {{id: $belief_id}})
                MERGE (e:{label} {{name: $name}})
                ON CREATE SET e.id = apoc.create.uuid(), e.created_at = timestamp()
                MERGE (m)-[:EXTRACTED_ENTITY]->(e)
                """
                res = await session.run(query, belief_id=belief_id, name=name)
                summary = await res.consume()
                nodes_processed += 1
                nodes_created_db += summary.counters.nodes_created

            # 2. Merge semantic relationships
            for rel in relationships:
                source = rel.get("source")
                target = rel.get("target")
                rel_type = rel.get("type")
                if not source or not target or not rel_type:
                    continue

                # Verify endpoints exist and are linked to the memory node
                check_query = """
                MATCH (m:Memory {id: $belief_id})
                OPTIONAL MATCH (m)-[:EXTRACTED_ENTITY]->(s {name: $source})
                OPTIONAL MATCH (m)-[:EXTRACTED_ENTITY]->(t {name: $target})
                RETURN s IS NOT NULL AS source_exists, t IS NOT NULL AS target_exists
                """
                check_res = await session.run(check_query, belief_id=belief_id, source=source, target=target)
                check_record = await check_res.single()
                
                source_exists = False
                target_exists = False
                if check_record:
                    source_exists = check_record["source_exists"]
                    target_exists = check_record["target_exists"]

                if not source_exists or not target_exists:
                    logger.warning(
                        f"MISSING ENDPOINTS: Cannot create relationship '{rel_type}' "
                        f"between '{source}' (exists: {source_exists}) and '{target}' (exists: {target_exists}) "
                        f"for memory '{belief_id}'."
                    )
                    continue

                # Clean and sanitize relationship type to prevent Cypher injection
                import re
                clean_rel_type = re.sub(r'[^a-zA-Z0-9_]', '', rel_type).upper()
                if not clean_rel_type or not clean_rel_type[0].isalpha():
                    clean_rel_type = "RELATED_TO"

                query = f"""
                MATCH (m:Memory {{id: $belief_id}})
                MATCH (source {{name: $source}})<-[:EXTRACTED_ENTITY]-(m)
                MATCH (target {{name: $target}})<-[:EXTRACTED_ENTITY]-(m)
                MERGE (source)-[r:{clean_rel_type}]->(target)
                ON CREATE SET r.confidence = 1.0, r.created_at = timestamp()
                """
                res = await session.run(query, belief_id=belief_id, source=source, target=target)
                summary = await res.consume()
                relationships_processed += 1
                relationships_created_db += summary.counters.relationships_created

        logger.info(
            f"Entities and relations processed for Memory '{belief_id}' in Neo4j. "
            f"Nodes: {nodes_processed} processed ({nodes_created_db} created in DB). "
            f"Relationships: {relationships_processed} processed ({relationships_created_db} created in DB)."
        )
        return {
            "nodes_processed": nodes_processed,
            "nodes_created_db": nodes_created_db,
            "relationships_processed": relationships_processed,
            "relationships_created_db": relationships_created_db
        }

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
        """Returns normalized degree centrality of any node in the graph."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized.")

        query = """
        MATCH (c {id: $concept_id})
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

    async def update_memory_access(self, belief_id: str, user_id: str):
        """Reinforces recall confidence and updates last_accessed timestamp for an existing memory in Neo4j."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized.")

        query = """
        MATCH (u:User {id: $user_id})-[r:RECALLS]->(m:Memory {id: $belief_id})
        SET r.last_accessed = timestamp(),
            r.confidence = CASE WHEN r.confidence + 0.1 > 1.0 THEN 1.0 ELSE r.confidence + 0.1 END
        """
        async with self.driver.session() as session:
            await session.run(query, user_id=user_id, belief_id=belief_id)
            logger.info(f"Reinforced recall confidence and timestamp for Memory node '{belief_id}' in Neo4j.")

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

    async def get_all_projects(self) -> list[dict]:
        """Queries Neo4j for all Project nodes ordered by created_at DESC."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized.")
        query = """
        MATCH (p:Project)
        RETURN p.name AS name, p.id AS id, p.created_at AS created_at
        ORDER BY p.created_at DESC
        """
        async with self.driver.session() as session:
            result = await session.run(query)
            return [
                {
                    "name": record["name"],
                    "id": record["id"],
                    "created_at": record["created_at"]
                }
                async for record in result
            ]

    async def get_all_technologies(self) -> list[dict]:
        """Queries Neo4j for all Technology nodes ordered by name."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized.")
        query = """
        MATCH (t:Technology)
        RETURN t.name AS name, t.id AS id
        ORDER BY t.name
        """
        async with self.driver.session() as session:
            result = await session.run(query)
            return [
                {
                    "name": record["name"],
                    "id": record["id"]
                }
                async for record in result
            ]

    async def get_all_careers(self) -> list[dict]:
        """Queries Neo4j for all Career nodes ordered by name."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized.")
        query = """
        MATCH (c:Career)
        RETURN c.name AS name, c.id AS id
        ORDER BY c.name
        """
        async with self.driver.session() as session:
            result = await session.run(query)
            return [
                {
                    "name": record["name"],
                    "id": record["id"]
                }
                async for record in result
            ]

    async def get_all_contradictions(self) -> list[dict]:
        """Queries Neo4j for all Contradiction nodes ordered by created_at DESC."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized.")
        query = """
        MATCH (c:Contradiction)
        RETURN c.id AS id, c.category AS category, c.status AS status, c.severity AS severity, c.created_at AS created_at
        ORDER BY c.created_at DESC
        """
        async with self.driver.session() as session:
            result = await session.run(query)
            return [
                {
                    "id": record["id"],
                    "category": record["category"],
                    "status": record["status"],
                    "severity": record["severity"],
                    "created_at": record["created_at"]
                }
                async for record in result
            ]

    async def get_dashboard_graph(self) -> dict:
        """Queries Neo4j for connected nodes and edges in a format suitable for graph rendering."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized.")
        query = """
        MATCH (n)-[r]->(m)
        RETURN n, r, m
        """
        nodes_map = {}
        edges = []
        async with self.driver.session() as session:
            result = await session.run(query)
            async for record in result:
                n = record["n"]
                m = record["m"]
                r = record["r"]

                n_id = n.get("id") or n.get("name") or str(n.element_id)
                n_name = n.get("name") or n.get("statement") or n_id
                n_labels = list(n.labels)
                n_label = n_labels[0] if n_labels else "Concept"

                m_id = m.get("id") or m.get("name") or str(m.element_id)
                m_name = m.get("name") or m.get("statement") or m_id
                m_labels = list(m.labels)
                m_label = m_labels[0] if m_labels else "Concept"

                nodes_map[n_id] = {
                    "id": n_id,
                    "name": n_name,
                    "label": n_label
                }
                nodes_map[m_id] = {
                    "id": m_id,
                    "name": m_name,
                    "label": m_label
                }

                edges.append({
                    "source": n_id,
                    "target": m_id,
                    "type": r.type
                })

        return {
            "nodes": list(nodes_map.values()),
            "edges": edges
        }

    async def get_dashboard_stats(self) -> dict:
        """Returns counts for Project, Technology, Career, and Contradiction nodes in Neo4j."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized.")
        query = """
        MATCH (p:Project) WITH count(p) AS projects
        MATCH (t:Technology) WITH projects, count(t) AS technologies
        MATCH (c:Career) WITH projects, technologies, count(c) AS careers
        MATCH (cr:Contradiction)
        RETURN projects, technologies, careers, count(cr) AS contradictions
        """
        async with self.driver.session() as session:
            result = await session.run(query)
            record = await result.single()
            if record:
                return {
                    "projects": record["projects"],
                    "technologies": record["technologies"],
                    "careers": record["careers"],
                    "contradictions": record["contradictions"]
                }
            return {
                "projects": 0,
                "technologies": 0,
                "careers": 0,
                "contradictions": 0
            }

# Singleton instance
neo4j_db = Neo4jManager()
