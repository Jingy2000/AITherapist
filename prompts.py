INTRO_1 = """
An AI-powered therapy application designed to be your virtual companion on your journey toward better mental well-being. 
Built with advanced natural language processing and machine learning algorithms, AI Therapist provides a safe, non-judgmental space for you to express your thoughts, concerns, and emotions.
"""

INTRO_2 = """
Our AI therapist is trained to actively listen, empathize, and provide personalized insights and coping strategies tailored to your unique situation. 
Whether you're dealing with anxiety, depression, relationship issues, or simply seeking self-improvement, AI Therapist is here to support you every step of the way.
"""

SYSTEM_PROMPT = """
You are a therapist having a counseling with a visitor. 
The counselor's replies should incorporate elements of empathy based on the user's descriptions, such as listening, leading, comforting, understanding, trust, acknowledgment, sincerity, and emotional support.
"""

SUMMARIZATION_PROMPT = """
Summarize the following therapeutic conversation, highlighting the key issues discussed, the emotions expressed by the client, and any advice given by the therapist.
Ensure the summary is concise and captures the essence of the session without using any direct quotations from the conversation.
"""


def get_summary_prompt(conversation_text):
    return f"""{SUMMARIZATION_PROMPT}
    Conversation: {conversation_text}"""
