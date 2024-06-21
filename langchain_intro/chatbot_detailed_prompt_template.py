import dotenv
import logging 

from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)



# Initialise Logger
logging.basicConfig(level=logging.INFO, format="[{asctime}] - {funcName} - {message}", style='{')
logger = logging.getLogger("LLM_RAG_NEO4J")

dotenv.load_dotenv()

#
# Simple prompt
# string template is a human message by default
#
review_template_str = """Your job is to use patient
    reviews to answer questions about their experience at a hospital.
    Use the following context to answer questions. Be as detailed
    as possible, but don't make up any information that's not
    from the context. If you don't know an answer, say you don't know.

    {context}

    {question}
    """

review_template = ChatPromptTemplate.from_template(review_template_str)

context = "I had a great stay!"
question = "Did anyone have a positive experience?"

logger.info(f"Simple Prompt : \n {review_template.format(context=context, question=question)}")

#
# detailed prompt template
# 
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

context = "I had a great stay!"
question = "Did anyone have a positive experience?"

logger.info(f"Detailed prompt Template : \n {review_prompt_template.format_messages(context=context, question=question)}")
