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
