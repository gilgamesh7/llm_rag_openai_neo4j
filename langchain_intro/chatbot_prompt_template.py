import dotenv
import logging 

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate



# Initialise Logger
logging.basicConfig(level=logging.INFO, format="[{asctime}] - {funcName} - {message}", style='{')
logger = logging.getLogger("LLM_RAG_NEO4J")

dotenv.load_dotenv()

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

