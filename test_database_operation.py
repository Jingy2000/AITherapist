import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from main import Base, Conversation, Message
from main import start_conversation, store_message, get_conversation_messages

class TestDatabaseOperations(unittest.TestCase):
    def setUp(self):
        """Set up a database engine and session before each test method."""
        self.engine = create_engine('sqlite:///:memory:')  # Use an in-memory SQLite database
        Base.metadata.create_all(self.engine)  # Create all tables
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    def tearDown(self):
        """Tear down and clean up after each test method."""
        Base.metadata.drop_all(self.engine)  # Clean up the database
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

# Run the tests
if __name__ == '__main__':
    unittest.main()
