import dotenv
import logging 

from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()

# Initialise Logger
logging.basicConfig(level=logging.INFO, format="[{asctime}] - {funcName} - {message}", style='{')
logger = logging.getLogger("LLM_RAG_NEO4J")

# path to the CSV & stored ChromaDB data
REVIEWS_CSV_PATH = "data/reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

#  load the reviews 
loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
reviews = loader.load()

# create a ChromaDB instance from reviews using the default OpenAI embedding model
reviews_vector_db = Chroma.from_documents(
    reviews,
    OpenAIEmbeddings(),
    persist_directory=REVIEWS_CHROMA_PATH
)

# create a new Chroma instance pointing to your vector database
reviews_vector_db_instance = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=OpenAIEmbeddings()
)

question = 'Has anyone complained about communication with hospital staff?'

relevant_docs = reviews_vector_db_instance.similarity_search(question, k=3)

logger.info(f"\n{relevant_docs[0].page_content} \n{relevant_docs[1].page_content} \n{relevant_docs[2].page_content}")


