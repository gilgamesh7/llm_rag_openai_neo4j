import dotenv
import logging 

from langchain_openai import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)


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

# create a embeddings  from reviews using the default OpenAI embedding model
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

# TO TEST RETRIEVAL ONLY
# question = 'Has anyone complained about communication with hospital staff?'

# relevant_docs = reviews_vector_db_instance.similarity_search(question, k=3)

# logger.info(f"\n{relevant_docs[0].page_content} \n{relevant_docs[1].page_content} \n{relevant_docs[2].page_content}")

# Create prompt review template using ICEL
review_system_template_str = """Your job is to use patient
    reviews to answer questions about their experience at a
    hospital. Use the following context to answer questions.
    Be as detailed as possible, but don't make up any information
    that's not from the context. If you don't know an answer, say
    you don't know.

    {context}
    """

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_system_template_str
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

# OpenAI chat model to use
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Retriever object with the relevant documents from ChromaDB
reviews_retriever = reviews_vector_db_instance.as_retriever(k=10)

review_chain = (
    {"context": reviews_retriever, "question":RunnablePassthrough()}
    | review_prompt_template
    | chat_model
    | StrOutputParser()
)

question = 'Has anyone complained about communication with hospital staff?'

answer = review_chain.invoke(question)
logger.info(f'\n {answer} \n')