import streamlit as st
from langchain.chat_models import init_chat_model
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Modelo de chat
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", google_api_key=GEMINI_API_KEY)

# Configuramos colores con CSS personalizado
st.markdown(
    f"""
    <style>
    #chatbot-de-asistencia-con-gemini {{
    font-size: xx-large;
    color: #ff6900;
    text-align: center;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Chatbot de Asistencia con Gemini")

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
user_input = st.chat_input(placeholder="Escriba su pregunta")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Obtener respuesta del LLM con contexto
        # Spinner y texto mientras esperamos la respuesta
        with st.chat_message("assistant"):
            with st.spinner("Esperando respuesta..."):
                response = st.session_state.chain.run(user_input)
            st.markdown(response)
    # print(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
