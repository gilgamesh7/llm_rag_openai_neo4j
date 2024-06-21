import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

dotenv.load_dotenv()

# 
# Chains and LangChain Expression Language (LCEL)
#
