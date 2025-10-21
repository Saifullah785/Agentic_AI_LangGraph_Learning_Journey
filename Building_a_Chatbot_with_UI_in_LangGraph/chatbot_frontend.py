import streamlit as st
from langgraph_chatbot_backend import chatbot
from langchain_core.messages import HumanMessage



# st.session_state to hold the chat history

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# {'role: 'user', 'content': 'Hello!'}
# {'role': 'assistant', 'content': 'Hi there! How can I help you today?'}

# user input
user_input = st.chat_input("Type your message here...")

if user_input:
    # first add the message to message history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)


    ai_message = response['messages'][-1].content
    # first add the message to message history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)