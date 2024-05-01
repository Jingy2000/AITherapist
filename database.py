import time
from datetime import datetime
from sqlalchemy.exc import OperationalError
from sqlalchemy import (create_engine, Column, Integer,
                        String, DateTime, Enum, ForeignKey)
from sqlalchemy.orm import (sessionmaker,
                            relationship,
                            declarative_base)


Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime, default=datetime.now())
    summary = Column(String(2048))

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


def create_engine_with_checks(dsn, retries=7, delay=5):
    for _ in range(retries):
        try:
            engine = create_engine(dsn)
            with engine.connect() as connection:
                return engine
        except OperationalError as e:
            time.sleep(delay)
    
    return None

def create_session(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    Base.metadata.create_all(engine)
    return session

def start_conversation(session):
    new_conversation = Conversation()
    session.add(new_conversation)
    session.commit()
    return new_conversation.id

def store_message(session, conversation_id, message, role):
    new_message = Message(
        conversation_id=conversation_id,
        message=message,
        role=role,
        timestamp=datetime.now()
    )
    session.add(new_message)
    session.commit()

def store_summary(session, conversation_id, summary):
    conversation = session.query(Conversation).filter_by(id=conversation_id).one()
    conversation.summary = summary
    session.commit()

def get_conversation_messages(session, conversation_id):
    messages = session.query(
        Message
        ).filter_by(
            conversation_id=conversation_id
            ).order_by(
                Message.timestamp
                ).all()
    return messages

def get_conversation_summary(session, conversation_id):
    return session.query(Conversation).filter_by(id=conversation_id).one().summary

def get_all_conversations(session):
    return session.query(Conversation).all()


if __name__ == "__main__":
    pass
