from flask import Flask, request, jsonify
from retrieval_system import retrieve_question
from answer_generation import generate_answer

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Intelligent Chatbot API!"})

@app.route("/ask", methods=["POST"])
def ask():
    """
    Handles user queries and returns the chatbot's response.
    """
    try:
        # Get the user query from the POST request
        user_query = request.json.get("query", "")
        if not user_query:
            return jsonify({"error": "Query is required!"}), 400

        # Step 1: Retrieve the most relevant question and answer (context)
        context = retrieve_question(user_query, top_k=1)[0]  # Top-1 result
        context_question = context[0]
        context_answer = context[1]

        # Step 2: Generate an answer using the context
        refined_answer = generate_answer(user_query)

        # Return the response
        return jsonify({
            "query": user_query,
            "retrieved_question": context_question,
            "retrieved_answer": context_answer,
            "generated_answer": refined_answer
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5000)
