from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

load_dotenv()

# creating sample documents
documents = [
    "Langchain helps build LLM prowered applications",
    "Embeddings converts text into numerical vectors",
    "RAG improves LLM responses using external data",
    "Semantic search finds meaning not keywords",
    "Vector databases store embeddings efficiently"
]

# calling embedding model
model = OpenAIEmbeddings()

embeddings = model.embed_documents(documents)

query = input("Enter your search query : ")
query_embedding = model.embed_query(query)

# finding the similarity of user query and the documents
scores = cosine_similarity(
    [query_embedding],
    embeddings
)[0]

# following line returns max score
best_match_index = np.argmax(scores)

# output
print("\n Best matching document \n")
print(documents[best_match_index])
print("Similarity score : ", scores[best_match_index])
