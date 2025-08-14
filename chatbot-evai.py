# -*- coding: utf-8 -*-
# (c) José Cobas Rodríguez
import streamlit as st
from PIL import Image
from os.path import dirname, join
from langchain.chat_models import init_chat_model
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

if "im" not in st.session_state:
    impath = join(dirname(__file__),"favicon.png")
    st.session_state.im = Image.open(impath)

st.set_page_config(
    page_title="Chatbot EVA",
    page_icon=st.session_state.im,
)

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Modelo de chat
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", google_api_key=GEMINI_API_KEY)

# Modelo de embedings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Vector store con Chroma
from langchain_chroma import Chroma
if "vector_store" not in st.session_state:
    st.session_state.vector_store = Chroma(
        collection_name="evadocs_collection",
        persist_directory="./chroma2_db",
        embedding_function=embeddings,
    )

if "make_rag_prompt" not in st.session_state:
    def make_rag_prompt(query, relevant_context):
        t_text = "\n\n - -\n\n".join([doc.page_content for doc in relevant_context])
        escaped_context = t_text.replace("'", "").replace('"', "").replace("\n", " ")
        prompt = f"""Eres un asistente para tareas de preguntas y respuestas.
    Usa la siguiente INFORMACION para responder la pregunta.
    Si no sabes la respuesta, simplemente di que no la sabes.
    Usa un máximo de cinco oraciones y mantén la respuesta concisa.
        PREGUNTA: '{query}'
        INFORMACION: '{escaped_context}'
        RESPUESTA:
        """
        return prompt
    st.session_state.make_rag_prompt = make_rag_prompt

# Configuramos colores con CSS personalizado
st.markdown(
    f"""
    <style>
    #chatbot-de-asistencia-eva-desoft {{
    font-size: xx-large;
    color: #ff6900;
    text-align: center;
    }}    
    .stAppHeader {{display:none;}}
    .st-emotion-cache-1ghhuty {{
        background-color: rgb(129,184,193)!important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Chatbot de Asistencia EVA Desoft")

# Inicializamos memoria para contexto
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()


# Cadena de conversación con memoria
if "chain" not in st.session_state:
    st.session_state.chain = ConversationChain(llm=llm, memory=st.session_state.memory)

# Mostrar mensajes anteriores
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Capturamos input del usuario
user_input = st.chat_input(placeholder="Pregunte sobre el EVA, los cursos, etc")

if user_input:
    # print('vector store lenght ->', len(vector_store.get()['documents']))
    # print('user_input ->', user_input)
    contexto = st.session_state.vector_store.similarity_search(user_input, k=30)
    # print('contexto ->', contexto)
    prompt = st.session_state.make_rag_prompt(user_input, contexto)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Obtener respuesta del LLM con contexto
    # Spinner y texto mientras esperamos la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Esperando respuesta..."):
            response = st.session_state.chain.run(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
