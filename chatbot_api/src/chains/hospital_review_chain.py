from dotenv import dotenv_values
import logging 

from langchain.vectorstores.neo4j_vector import Neo4jVector
from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings

# Initialise Logger
logging.basicConfig(level=logging.INFO, format="[{asctime}] - {funcName} - {message}", style='{')
logger = logging.getLogger("LLM_RAG_NEO4J")

# Get Configurations
config = dotenv_values(".env") 
print(f"{config=}")

# LLM Model name
HOSPITAL_QA_MODEL = config.get("HOSPITAL_QA_MODEL")
print(f"{HOSPITAL_QA_MODEL=}")
print(f"{config.get('NEO4J_URI')=}")

driver = GraphDatabase.driver(
        config.get("NEO4J_URI"), 
        auth=(config.get("NEO4J_USERNAME"), config.get("NEO4J_PASSWORD"))
        ) 

with driver.session(database="neo4j") as session:
    print(driver.verify_connectivity())
    query = f"""MATCH (p:Patient)
                RETURN p LIMIT 5;"""
    
# Create Chain
# neo4j_vector_index = Neo4jVector.from_existing_graph(
#     embedding=OpenAIEmbeddings(),
#     url=config.get("NEO4J_URI"),
#     username=config.get("NEO4J_USERNAME"),
#     password=config.get("NEO4J_PASSWORD"),
#     index_name="reviews",
#     node_label="Review",
#     text_node_properties=[
#         "physician_name",
#         "patient_name",
#         "text",
#         "hospital_name",
#     ],
#     embedding_node_property="embedding",
# )



