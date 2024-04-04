import streamlit as st

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from rag import chunk_processing, embeddings
        
pdf_path = '../argentina/constitution/constitution.pdf'

pdf = open(pdf_path, 'rb')

processed_chunks = chunk_processing(pdf)

embedded_chunks = embeddings(processed_chunks)

def get_response(user_query):

    retriever = embedded_chunks.as_retriever()

    template = """
    You are a helpful assistant. Respond to the prompt based on the following context: {context}

    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Initialize ChatOpenAI model
    model = ChatOpenAI(model_name='gpt-4-0125-preview')

    # Define processing chain
    chain = (
        {"context": retriever, "user_question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return chain.stream(user_query)
    

st.title("NationGPT 🌞")

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am NationGPT. How can I help you today?"),
    ]

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("Nation Agent"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query))

    st.session_state.chat_history.append(AIMessage(content=response))

