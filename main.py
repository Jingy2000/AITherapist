import requests
import streamlit as st

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama


st.set_page_config(
    page_title="AI Therapist",
    page_icon=":coffee:",
    initial_sidebar_state="auto"
    )
st.title("Your AI Therapist")

ss = st.session_state

def send_post_request(local_model_name: str):
    url = "http://ollama:11434/api/pull"
    data = {"name": local_model_name,
            "stream": False,
            }
    response = requests.post(url, json=data)
    return response

with st.sidebar:
    st.markdown("<h1 style='text-align:left;font-family:Georgia'>AI Therapist </h1>",
                unsafe_allow_html=True)

    with st.form("config"):
        st.header("Configuration")
        st.divider()
        model = st.selectbox(
            "Model", 
            options=("gpt-3.5-turbo", "llama2", "mistral"),
            index=0,
            )
        openai_api_key = st.text_input(
            "Your OpenAI API key",
            placeholder="only for gpt-3.5-turbo",
            type="password"
            )
        temperature = st.slider("Temperature", 0.0, 2.0, 0.0, 0.1, format="%.1f")
        if st.form_submit_button("Submit"):
            ss.model_config = {
                "openai_api_key": openai_api_key,
                "model": model,
                "temperature": temperature,
            }

            if ss.model_config['model'] in ["llama2", "mistral"]:
                with st.spinner(text="Loading model ..."):
                    response = send_post_request(ss.model_config['model'])
                    if response.ok and response.json()['status'] == 'success':
                        st.success('Model Loaded!')
                    else:
                        st.error('This is an error', icon="🚨")

            if ss.model_config['model'] == "gpt-3.5-turbo":
                # Get an OpenAI API Key before continuing
                # Attempt to retrieve API key from secrets
                try:
                    openai_api_key = st.secrets["openai_api_key"]
                except (FileNotFoundError, KeyError):
                    openai_api_key = ss.model_config['openai_api_key']

                if not openai_api_key:
                    st.info("Enter an OpenAI API Key to continue")
                    st.stop()
    
                try:
                    response = ChatOpenAI(temperature=0,
                                          max_tokens=2,
                                          api_key=openai_api_key).invoke("Hello")
                except Exception as e:
                    st.warning('Please enter your valid OpenAI API key!', icon='⚠')
                    st.stop()

    st.divider()
    st.markdown("<h2 style='text-align:left;font-family:Georgia'>Introduction</h2>",
                unsafe_allow_html=True)
    st.markdown("An AI-powered therapy application designed to be your virtual companion "
                "on your journey toward better mental well-being. "
                "Built with advanced natural language processing and machine learning algorithms, "
                "AI Therapist provides a safe, non-judgmental space for you "
                "to express your thoughts, concerns, and emotions.")
    st.markdown("Our AI therapist is trained to actively listen, empathize, "
                "and provide personalized insights and coping strategies tailored to your unique situation. "
                "Whether you're dealing with anxiety, depression, relationship issues, "
                "or simply seeking self-improvement, "
                "AI Therapist is here to support you every step of the way.")
    st.markdown("<h2 style='text-align:left;font-family:Georgia'>Features</h2>",
                unsafe_allow_html=True)
    st.markdown(" - one")
    st.markdown(" - two")

st.divider()

if not "model_config" in ss:
    st.stop()

msgs = StreamlitChatMessageHistory(key="chat_messages")

if len(msgs.messages) == 0:
    msgs.add_ai_message("Hi, how are you doing today!")

# view_messages = st.expander("View the message contents in session state")

# Set up the LangChain, passing in Message History
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a therapist having a counseling with a visitor. "
                   "The counselor's replies should incorporate elements of empathy " 
                   "based on the user's descriptions, such as listening, leading, "
                   "comforting, understanding, trust, acknowledgment, "
                   "sincerity, and emotional support."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

if ss.model_config['model'] == "gpt-3.5-turbo":
    chain = prompt | ChatOpenAI(temperature=ss.model_config['temperature'],
                                api_key=openai_api_key)
elif ss.model_config['model'] in ["llama2", "mistral"]:
    if ss.model_config['model'] == "llama2":
        stop_tokens = ["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"]
    else:
        stop_tokens = ["[INST]", "[/INST]"]
    chain = prompt | ChatOllama(model=ss.model_config['model'],
                                base_url='http://ollama:11434',
                                stop=stop_tokens,
                                temperature=ss.model_config['temperature'])

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.stream({"question": prompt}, config)
    st.chat_message("ai").write_stream(response)

# Draw the messages at the end, so newly generated ones show up immediately
# with view_messages:
#     view_messages.json(st.session_state.chat_messages)
