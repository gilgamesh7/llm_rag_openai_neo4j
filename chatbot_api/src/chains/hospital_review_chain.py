import os
from dotenv import dotenv_values
import logging 

from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI
)
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.chains import RetrievalQA

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

# TEST CONNECTIVITY
# from neo4j import GraphDatabase 
# driver = GraphDatabase.driver(
#         config.get("NEO4J_URI"), 
#         auth=(config.get("NEO4J_USERNAME"), 
#               config.get("NEO4J_PASSWORD"))
#         ) 

# with driver.session(database="neo4j") as session:
#     print(driver.verify_connectivity())
#     query = f"""MATCH (p:Patient)
#                 RETURN p LIMIT 5;"""
#     summary = driver.execute_query(query)
#     print(summary)
    
# Create Chain
os.environ["NEO4J_URI"] = config.get("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = config.get("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = config.get("NEO4J_PASSWORD")
neo4j_vector_index = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(), # The model used to create the embeddings
    index_name="reviews",
    node_label="Review", # Node to create embeddings for
    text_node_properties=[ # node properties to include in the embedding
        "physician_name",
        "patient_name",
        "text",
        "hospital_name",
    ],
    embedding_node_property="embedding",
)

# Create Prompt template
review_template = """Your job is to use patient
reviews to answer questions about their experience at a hospital. Use
the following context to answer questions. Be as detailed as possible, but
don't make up any information that's not from the context. If you don't know
an answer, say you don't know.
{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template
        )
    )   

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}"
        )
) 

messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["context","question"],
    messages=messages
)


reviews_vector_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name=HOSPITAL_QA_MODEL, temperature=0), 
    chain_type="stuff", 
    retriever=neo4j_vector_index.as_retriever(k=12),
    return_source_documents=True
    )

reviews_vector_chain.combine_documents_chain.llm_chain.prompt = review_prompt_template




