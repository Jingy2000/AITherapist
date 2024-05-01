import unittest
from sqlalchemy import create_engine
from database import Base, Conversation, Message
from database import (start_conversation,
                      store_message,
                      get_conversation_messages,
                      create_session,
                      get_all_conversations,
                      store_summary,
                      get_conversation_summary,
                      concate_messages,
                      )


class TestDatabaseOperations(unittest.TestCase):
    def setUp(self):
        """Set up a database engine and session before each test method."""
        self.engine = create_engine('sqlite:///:memory:')
        self.session = create_session(engine=self.engine)

    def tearDown(self):
        """Tear down and clean up after each test method."""
        Base.metadata.drop_all(self.engine)
        self.session.close()

    def test_start_conversation(self):
        """Test the start of a conversation is recorded correctly."""
        conv_id = start_conversation(self.session)
        self.assertEqual(self.session.query(Conversation).count(), 1, "Should be one conversation in the database.")
        self.assertIsInstance(conv_id, int, "Conversation ID should be an integer.")

    def test_store_and_retrieve_message(self):
        """Test storing and retrieving messages works correctly."""
        conversation_id = start_conversation(self.session)
        store_message(self.session, conversation_id, "Test message", "human")
        messages = get_conversation_messages(self.session, conversation_id)
        self.assertEqual(len(messages), 1, "Should retrieve one message.")
        self.assertEqual(messages[0].message, "Test message", "Message content should match the stored value.")
        self.assertEqual(messages[0].role, 'human', "Message role should be 'human'.")
    
    def test_get_all_conversations(self):
        """Test that all conversations are retrieved."""
        for _ in range(5):
            start_conversation(self.session)
        all_conversations = get_all_conversations(self.session)
        self.assertEqual(len(all_conversations), 5, "Should retrieve five conversations.")
        self.assertTrue(all(isinstance(conv, Conversation) for conv in all_conversations),
                        "All items should be instances of Conversation.")
    
    def test_store_and_get_summary(self):
        """Test storing and retrieving a summary."""
        conv_id = start_conversation(self.session)
        summary_text = "This is a test summary."
        store_summary(self.session, conv_id, summary_text)
        retrieved_summary = get_conversation_summary(self.session, conv_id)
        self.assertEqual(retrieved_summary, summary_text, "The retrieved summary should match the stored summary.")
    
    def test_concat_messages(self):
        """Test the concatenation of messages."""
        conversation_id = start_conversation(self.session)
        store_message(self.session, conversation_id, "How have you been feeling this week?", "ai")
        store_message(self.session, conversation_id, "I've been feeling anxious about work.", "human")
        store_message(self.session, conversation_id, "Have you tried any relaxation techniques?", "ai")
        messages = get_conversation_messages(self.session, conversation_id)
        
        result = concate_messages(messages)
        expected_result = "[ai]: How have you been feeling this week? [human]: I've been feeling anxious about work. [ai]: Have you tried any relaxation techniques?"
        self.assertEqual(result, expected_result, "The concatenated messages should match the expected format.")


if __name__ == '__main__':
    unittest.main()
