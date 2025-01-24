import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the FAISS index and dataset
print("Loading FAISS index and dataset...")
index = faiss.read_index("faq_index.faiss")
faq_data = pd.read_csv("E:/intelligent-chatbot-rag/intelligent-chatbot-rag/faq_data_with_embeddings.csv")

# Initialize the SentenceTransformer model
model = SentenceTransformer('msmarco-distilbert-base-tas-b')

def retrieve_question(user_query, top_k=3):
    """
    Retrieves the top-k most relevant questions from the dataset based on the user query.
    
    Args:
        user_query (str): The user's query/question.
        top_k (int): Number of top results to retrieve.

    Returns:
        list of tuples: A list of (question, answer, similarity_score).
    """
    # Generate embeddings for the user query
    query_embedding = model.encode([user_query])

    # Search the FAISS index for the top-k results
    distances, indices = index.search(np.array(query_embedding), top_k)

    # Retrieve the corresponding questions and answers
    results = []
    for i, idx in enumerate(indices[0]):
        question = faq_data.iloc[idx]['Questions']
        answer = faq_data.iloc[idx]['Answers']
        similarity_score = 1 / (1 + distances[0][i])  # Higher is better
        results.append((question, answer, similarity_score))

    return results

# Test the retrieval system
if __name__ == "__main__":
    print("Enter your query (type 'exit' to quit):")
    while True:
        user_query = input(">> ")
        if user_query.lower() == "exit":
            print("Exiting...")
            break

        results = retrieve_question(user_query)
        print("\nTop Results:")
        for i, (question, answer, score) in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Similarity Score: {score:.4f}")
