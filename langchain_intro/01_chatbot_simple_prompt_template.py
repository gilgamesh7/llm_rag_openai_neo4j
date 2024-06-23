import dotenv
import logging 

import os

from langchain_openai import ChatOpenAI
from langchain.schema.messages import (
    HumanMessage, 
    SystemMessage
)



# Initialise Logger
logging.basicConfig(level=logging.INFO, format="[{asctime}] - {funcName} - {message}", style='{')
logger = logging.getLogger("LLM_RAG_NEO4J")

dotenv.load_dotenv()

# For debugging only
# logger.info(f'{os.getenv("OPENAI_API_KEY")}=')

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

messages = [
    SystemMessage(
        content="You are an assistance knowledgeable only about the books of H.P.Lovecraft. Answer only questions about H.P.Lovecraft and his books"
    ),
    HumanMessage(content="How many of Lovecrafts books mention Cthulhu")
]
logger.info(f"{chat_model.invoke(messages)}")


messages = [
    SystemMessage(
        content="You are an assistance knowledgeable only about the books of H.P.Lovecraft. Answer only questions about H.P.Lovecraft and his books"
    ),
    HumanMessage(content="How many volts in a bolt of lightning")
]
logger.info(f"{chat_model.invoke(messages)}")