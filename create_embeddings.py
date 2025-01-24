from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np

# Load the dataset
data_path = "E:/intelligent-chatbot-rag/intelligent-chatbot-rag/data/preprocessed_faq_data.csv"
data = pd.read_csv(data_path)

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # A lighter retrieval model

# Generate embeddings for the Questions
print("Generating embeddings...")
embeddings = model.encode(data['Questions'].tolist())

# Create a FAISS index for similarity search
dimension = embeddings.shape[1]  # Dimensionality of the embeddings
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
index.add(np.array(embeddings))  # Add embeddings to the index

# Save the FAISS index and dataset
faiss.write_index(index, "faq_index.faiss")  # Save the FAISS index
data.to_csv("faq_data_with_embeddings.csv", index=False)  # Save the dataset

print("Embeddings and FAISS index saved successfully!")
