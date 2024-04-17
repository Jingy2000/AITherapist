import os, time
import requests
import streamlit as st
from datetime import datetime
from sqlalchemy.exc import OperationalError
from sqlalchemy import (create_engine, Column, Integer,
                        String, DateTime, Enum, ForeignKey)
from sqlalchemy.orm import (sessionmaker,
                            relationship,
                            declarative_base)

from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory


# SQL database
db_user = os.getenv('MYSQL_USER')
db_password = os.getenv('MYSQL_PASSWORD')
db_host = os.getenv('MYSQL_HOST')
db_name = os.getenv('MYSQL_DB')

def create_engine_with_checks(dsn, retries=7, delay=5):
    for _ in range(retries):
        try:
            engine = create_engine(dsn)
            with engine.connect() as connection:
                return engine
        except OperationalError as e:
            time.sleep(delay)
    
    return None

engine = create_engine_with_checks(dsn=f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}')

if not engine: 
    raise Exception("Failed to connect to the database after several attempts.")

Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime, default=datetime.now())

    # Relationship to link messages to a conversation
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'))
    message = Column(String(2048))
    timestamp = Column(DateTime, default=datetime.now())
    role = Column(Enum('human', 'ai', name='role_types'))

    # Relationship to link a message back to its conversation
    conversation = relationship("Conversation", back_populates="messages")

def start_conversation():
    new_conversation = Conversation()
    session.add(new_conversation)
    session.commit()
    return new_conversation.id

def store_message(conversation_id, message, role):
    new_message = Message(
        conversation_id=conversation_id,
        message=message,
        role=role,
        timestamp=datetime.now()
    )
    session.add(new_message)
    session.commit()

def get_conversation_messages(conversation_id):
    messages = session.query(
        Message
        ).filter_by(
            conversation_id=conversation_id
            ).order_by(
                Message.timestamp
                ).all()
    return messages

def get_all_conversations():
    return session.query(Conversation).all()

Base.metadata.create_all(engine)

# Webpage rendering
st.set_page_config(
    page_title="AI Therapist",
    page_icon=":coffee:",
    initial_sidebar_state="auto"
    )
st.markdown("<h2 style='text-align:left;font-family:Georgia'>Your AI Therapist</h2>",
            unsafe_allow_html=True)

ss = st.session_state

def send_post_request(local_model_name: str):
    url = "http://ollama:11434/api/pull"
    data = {"name": local_model_name,
            "stream": False,
            }
    response = requests.post(url, json=data)
    return response

with st.sidebar:
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
            type="password",
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
                        ss.model_is_ready = True
                    else:
                        st.error('This is an error', icon="🚨")
                        ss.model_is_raedy = False

            if ss.model_config['model'] == "gpt-3.5-turbo":
                # Get an OpenAI API Key before continuing
                # Attempt to retrieve API key from secrets
                try:
                    openai_api_key = st.secrets["openai_api_key"]
                    st.warning('Found a key in local secrets.toml')
                except (FileNotFoundError, KeyError):
                    openai_api_key = ss.model_config['openai_api_key']

                if not openai_api_key:
                    st.info("Enter an OpenAI API Key to continue")
                    ss.model_is_ready = False
                else:
                    try:
                        response = ChatOpenAI(temperature=0,
                                            max_tokens=2,
                                            api_key=openai_api_key).invoke("Hello")
                        st.success('Model Loaded!')
                        ss.model_is_ready = True
                    except Exception as e:
                        st.warning('Please enter your valid OpenAI API key!', icon='⚠')
                        ss.model_is_ready = False

    with st.form("new conversation"):
        st.header("New Conversation")
        if st.form_submit_button("Start"):
            if not 'model_is_ready' in ss or not ss.model_is_ready:
                st.warning("Please set up the configuration")
            else:
                ss.initiate_conversation = True
                if "selected_chat_history" in ss:
                    del ss.selected_chat_history
                if "chat_messages" in ss:
                    del ss.chat_messages

    with st.form("history"):
        st.header("Chat history")
        chat_history_in_db = get_all_conversations()
        chat_history_start_time_list = [conversation.id
                                        for conversation in chat_history_in_db]
        selected_item = st.selectbox("Chat history:", chat_history_start_time_list)

        if st.form_submit_button("Confirm",
                                 disabled=(selected_item==None)):
            if not 'model_is_ready' in ss or not ss.model_is_ready:
                st.warning("Please set up the configuration")
            else:
                ss.current_conversation_id = selected_item
                ss.selected_chat_history = get_conversation_messages(selected_item)
                if "chat_messages" in ss:
                    del ss.chat_messages
                if "initiate_conversation" in ss:
                    del ss.initiate_conversation
                ss.initiate_conversation = False
                st.write(f"You selected {selected_item}")
            

st.divider()

if "initiate_conversation" not in ss and "selected_chat_history" not in ss:
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
    st.stop()


msgs = StreamlitChatMessageHistory(key="chat_messages")

if 'selected_chat_history' in ss:
    if len(msgs.messages) == 0:
        for msg_history in ss.selected_chat_history:
            if msg_history.role == "ai":
                msgs.add_ai_message(msg_history.message)
            elif msg_history.role == "human":
                msgs.add_user_message(msg_history.message)
elif "initiate_conversation" in ss:
    if len(msgs.messages) == 0:
        ss.current_conversation_id = start_conversation()
        msgs.add_ai_message("Hi, how are you doing today!")
        store_message(conversation_id=ss.current_conversation_id,
                      message=msgs.messages[-1].content,
                      role="ai",)
else:
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

if ss.model_config['model'] == "gpt-3.5-turbo":
    chain = prompt | ChatOpenAI(temperature=ss.model_config['temperature'],
                                api_key=ss.model_config['openai_api_key'])
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
    store_message(conversation_id=ss.current_conversation_id,
                  message=prompt,
                  role="human",)
    
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.stream({"question": prompt}, config)
    st.chat_message("ai").write_stream(response)
    store_message(conversation_id=ss.current_conversation_id,
                  message=msgs.messages[-1].content,
                  role="ai")
