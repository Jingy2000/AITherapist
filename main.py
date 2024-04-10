import streamlit as st
from openai import OpenAI

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

from langchain_core.language_models import SimpleChatModel

st.set_page_config(
    page_title="AI Therapist", page_icon=":coffee:", initial_sidebar_state="auto"
)

st.title("Your AI Therapist")

model_selection = st.sidebar.selectbox(
    "What's your preferred model of communication?",
    ("gpt-3.5-turbo", "llama2-7b-chat", "mistral-7b")
)

msgs = StreamlitChatMessageHistory(key="chat_messages")

if len(msgs.messages) == 0:
    msgs.add_ai_message("Hi, how are you doing today!")

view_messages = st.expander("View the message contents in session state")  

if model_selection == "gpt-3.5-turbo":
    # Get an OpenAI API Key before continuing
    
    # Attempt to retrieve API key from secrets
    try:
        openai_api_key = st.secrets["openai_api_key"]
    except (FileNotFoundError, KeyError):
        openai_api_key = None
        openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

    if not openai_api_key:
        st.info("Enter an OpenAI API Key to continue")
        st.stop()
    elif not openai_api_key.startswith('sk-'):
        st.warning('Please enter your valid OpenAI API key!', icon='⚠')
        st.stop()
    
    try:
        response = ChatOpenAI(temperature=0, api_key=openai_api_key).invoke("Hello")
    except Exception as e:
        st.warning('Please enter your valid OpenAI API key!', icon='⚠')
        st.stop()

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

if model_selection == "gpt-3.5-turbo":
    chain = prompt | ChatOpenAI(temperature=0, api_key=openai_api_key)
elif model_selection in ["llama2-7b-chat", "mistral-7b"]:
    if model_selection == "llama2-7b-chat":
        model = "llama2"
        stop_tokens = ["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"]
    else:
        model = "mistral"
        stop_tokens = ["[INST]", "[/INST]"]
    chain = prompt | ChatOllama(model=model, base_url='http://ollama:11434', stop=stop_tokens)

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
with view_messages:
    view_messages.json(st.session_state.chat_messages)
