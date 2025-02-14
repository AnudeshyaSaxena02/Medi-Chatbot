import os
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient  # ✅ NEW: Using InferenceClient

# Set environment variables
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

# Define FAISS database path
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """Loads FAISS vector store with embeddings."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading FAISS vector store: {str(e)}")
        return None

def set_custom_prompt(custom_prompt_template):
    """Creates a prompt template."""
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    """Initializes the Hugging Face InferenceClient."""
    return InferenceClient(model=huggingface_repo_id, token=HF_TOKEN)

def main():
    st.title("Ask Chatbot!")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # Get user input
    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        # Display and store user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don’t know the answer, just say that you don’t know. Don’t make up an answer.
        Don’t provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk, please.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        # Ensure HF Token is set
        if not HF_TOKEN:
            st.error("Hugging Face Token (HF_TOKEN) not found. Please set it in environment variables.")
            return

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store.")
                return

            llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN)

            # Retrieve relevant documents
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            retrieved_docs = retriever.get_relevant_documents(prompt)

            if not retrieved_docs:
                st.error("No relevant documents found for the query.")
                return

            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Format the prompt with retrieved context
            formatted_prompt = CUSTOM_PROMPT_TEMPLATE.format(context=context, question=prompt)

            # Send request to Hugging Face model
            response = llm.text_generation(formatted_prompt, max_new_tokens=512)

            if response:
                result_to_show = response + "\n\n**Sources:** " + str(retrieved_docs)
                st.chat_message("assistant").markdown(result_to_show)
                st.session_state.messages.append({"role": "assistant", "content": result_to_show})
            else:
                st.error("No response from the model.")

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
