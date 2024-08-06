import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# Sidebar contents
with st.sidebar:
    st.title('💻 Tech9Labs Chat App')
    st.markdown("""
    ## About
    This app is an LLM-powered chatbot built for:
    - **Intelligent Document Analysis:** Utilizes NLP and ML to analyze documents, extracting key information and providing accurate responses to queries.
    - **Seamless User Interaction:** Offers a smooth experience by understanding natural language prompts and delivering clear, concise answers.
    - **Enhanced Business Insights:** Synthesizes data from various sources to highlight trends, identify anomalies, and offer actionable recommendations.
    """)
    add_vertical_space(5)
    st.write('Made by Parth Maheshwari and Arnav Jain.')

# Main function
def main():
    st.header("Tech9ChatBot💬")
    load_dotenv()

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # Load OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set the API key in the environment.")
        return

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Embeddings
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.faiss"):
            VectorStore = FAISS.load_local(store_name)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            VectorStore.save_local(store_name)

        # Display the conversation history
        for i, (user_query, bot_response) in enumerate(st.session_state.conversation):
            st.write(f"**You:** {user_query}")
            st.write(f"**Bot:** {bot_response}")
            st.markdown("---")  # Line to separate past responses

        # Text input for the next question
        query = st.text_input("Ask another question:", value="", key="unique_input_query")

        if st.button("Submit", key="submit"):
            if query:
                # Process the query and generate a response
                docs = VectorStore.similarity_search(query=query, k=3)
                llm = OpenAI()
                chain = load_qa_chain(llm=llm, chain_type="stuff")

                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                
                # Append to conversation
                st.session_state.conversation.append((query, response))

                # Clear input field by setting session state to empty
                st.session_state.user_input = ""

                # Update the input box by resetting the value
                st.text_input("Ask another question:", value="", key="reset_input_query")

if __name__ == '__main__':
    main()
