import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

from database import (create_engine_with_checks,
                      get_all_conversations,
                      create_session,
                      get_conversation_messages,
                      start_conversation,
                      store_message,
                      get_conversation_summary,
                      store_summary,
                      )
from restful_ollama import pull, generate
import prompts


# SQL database
db_user = os.getenv('MYSQL_USER')
db_password = os.getenv('MYSQL_PASSWORD')
db_host = os.getenv('MYSQL_HOST')
db_name = os.getenv('MYSQL_DB')

db_engine = create_engine_with_checks(dsn=f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}')

if not db_engine: 
        raise Exception("Failed to connect to the database after several attempts.")

db_session = create_session(engine=db_engine)

# Webpage rendering
st.set_page_config(
    page_title="AI Therapist",
    page_icon=":coffee:",
    initial_sidebar_state="auto"
    )
st.markdown("<h2 style='text-align:left;font-family:Georgia'>Your AI Therapist</h2>",
            unsafe_allow_html=True)

ss = st.session_state

with st.sidebar:
    with st.form("config"):
        st.header("Configuration")
        st.divider()
        model = st.selectbox(
            "Model", 
            options=("gpt-3.5-turbo", "gpt-4", "llama2-ft", "llama2", "mistral", "llama3"),
            index=0,
            )
        openai_api_key = st.text_input(
            "Your OpenAI API key",
            placeholder="only for gpt-3.5-turbo and gpt-4",
            type="password",
            )
        temperature = st.slider("Temperature", 0.0, 2.0, 0.0, 0.1, format="%.1f")

        if st.form_submit_button("Submit"):
            ss.model_config = {
                "openai_api_key": openai_api_key,
                "model": model if model!="llama2-ft" else "junyao/llama2-ft-4bit",
                "temperature": temperature,
            }

            if ss.model_config['model'] in ["llama2", "mistral", "junyao/llama2-ft-4bit", "llama3"]:
                with st.spinner(text="Loading model ..."):
                    response = pull(ss.model_config['model'])
                    if response.ok and response.json()['status'] == 'success':
                        st.success('Model Loaded!')
                        ss.model_is_ready = True
                    else:
                        st.error('This is an error', icon="🚨")
                        ss.model_is_raedy = False

            if ss.model_config['model'] in ["gpt-3.5-turbo", "gpt-4"]:
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
        chat_history_in_db = get_all_conversations(session=db_session)
        chat_history_start_time_list = [conversation.id
                                        for conversation in chat_history_in_db]
        selected_item = st.selectbox("Chat history:", chat_history_start_time_list)

        if st.form_submit_button("Confirm",
                                 disabled=(selected_item==None)):
            if not 'model_is_ready' in ss or not ss.model_is_ready:
                st.warning("Please set up the configuration")
            else:
                ss.current_conversation_id = selected_item
                ss.selected_chat_history = get_conversation_messages(session=db_session,
                                                                     conversation_id=selected_item)
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
        ss.current_conversation_id = start_conversation(session=db_session)
        msgs.add_ai_message("Hi, how are you doing today!")
        store_message(session=db_session,
                      conversation_id=ss.current_conversation_id,
                      message=msgs.messages[-1].content,
                      role="ai",)
else:
    st.stop()

# Set up the LangChain, passing in Message History
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompts.SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

if ss.model_config['model'] in ["gpt-3.5-turbo", "gpt-4"]:
    chain = prompt | ChatOpenAI(temperature=ss.model_config['temperature'],
                                api_key=ss.model_config['openai_api_key'])
elif ss.model_config['model'] in ["llama2", "mistral", "junyao/llama2-ft-4bit", "llama3"]:
    if ss.model_config['model'] == "mistral":
        stop_tokens = ["[INST]", "[/INST]"]
    elif ss.model_config['model'] == "llama3":
        stop_tokens = ["<|start_header_id|>",
                       "<|end_header_id|>",
                       "<|eot_id|>"]
    else:
        stop_tokens = ["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"]
    
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
    store_message(session=db_session,
                  conversation_id=ss.current_conversation_id,
                  message=prompt,
                  role="human",)
    
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.stream({"question": prompt}, config)
    st.chat_message("ai").write_stream(response)
    store_message(session=db_session,
                  conversation_id=ss.current_conversation_id,
                  message=msgs.messages[-1].content,
                  role="ai")
