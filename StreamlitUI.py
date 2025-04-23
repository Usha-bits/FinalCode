import streamlit as st
import json
import requests

# FastAPI server URL (modify if running on a different host/port)
FASTAPI_URL = "https://5b57-223-185-134-35.ngrok-free.app/ask" 

class AiModel:
    """Class to handle communication with the FastAPI backend."""

    @staticmethod
    def get_chatbot_response(user_question):
        """Send a question to FastAPI and return the chatbot's response."""
        try:
            headers = {"Content-Type": "application/json"}
            payload = {"question": user_question}

            response = requests.post(FASTAPI_URL, json=payload, headers=headers)
            response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
            return response.json().get("answer", "No response received.")
        except requests.exceptions.RequestException as e:
            return f"Error: Unable to connect to the AI service. ({e})"
def main():
    """Streamlit UI for the AI chatbot."""
    st.title("Welcome to GitLab. How can I help you?")

    # User input
    question = st.text_input("Enter your question:")

    # Process query
    if st.button("Ask"):
        question = question.strip()  # Remove leading/trailing whitespace
        if not question:
            st.warning("Please enter a valid question.")
        else:
            with st.spinner("Searching for the best answer..."):
                answer = AiModel.get_chatbot_response(question)
            st.subheader("AI Answer:")
            st.write(answer)

# Run Streamlit app
if __name__ == "__main__":
    main()
