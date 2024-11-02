import dotenv
import os
import sys

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_directory)

from chatbot_api.src.chains.hospital_review_chain import(
    reviews_vector_chain
)


dotenv.load_dotenv()


query = """What have patients said about hospital efficiency?
Mention details from specific reviews."""

response = reviews_vector_chain.invoke(query)

print(response.get('result'))
