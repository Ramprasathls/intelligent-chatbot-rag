import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from retrieval_system import retrieve_question

# Load the FAISS index and dataset
print("Loading FAISS index and dataset...")
index = faiss.read_index("faq_index.faiss")
faq_data = pd.read_csv("E:/intelligent-chatbot-rag/intelligent-chatbot-rag/faq_data_with_embeddings.csv")

# Initialize the SentenceTransformer model
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the Hugging Face question-answering pipeline
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def retrieve_context(user_query, top_k=1):
    """
    Retrieves the most relevant context (question and answer) based on the user's query.
    
    Args:
        user_query (str): The user's query/question.
        top_k (int): Number of top results to retrieve.

    Returns:
        tuple: The most relevant question and answer.
    """
    # Generate embeddings for the user query
    query_embedding = retrieval_model.encode([user_query])

    # Search the FAISS index for the top-k results
    distances, indices = index.search(np.array(query_embedding), top_k)

    # Retrieve the top question and answer
    question = faq_data.iloc[indices[0][0]]['Questions']
    answer = faq_data.iloc[indices[0][0]]['Answers']

    return question, answer

def generate_answer(user_query):
    """
    Generates a refined answer and provides a fallback if the model's answer is insufficient.
    """
    # Retrieve multiple contexts
    top_contexts = retrieve_question(user_query, top_k=3)
    combined_context = " ".join([f"{ctx[0]} {ctx[1]}" for ctx in top_contexts])

    # Generate answer using QA model
    response = qa_model(question=user_query, context=combined_context)

    # Fallback to retrieved answer if the generated answer is too short or irrelevant
    if len(response['answer'].split()) < 3:
        return top_contexts[0][1]  # Use the top retrieved answer as fallback

    return response['answer']


# Test the answer generation system
if __name__ == "__main__":
    print("Enter your query (type 'exit' to quit):")
    while True:
        user_query = input(">> ")
        if user_query.lower() == "exit":
            print("Exiting...")
            break

        answer = generate_answer(user_query)
        print("\nGenerated Answer:")
        print(answer)
