from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime

Base = declarative_base()

class ChatSession(Base):
    __tablename__ = 'chat_sessions'
    id = Column(Integer, primary_key=True)
    chat_name = Column(String, unique=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('chat_sessions.id'))
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    session = relationship("ChatSession", back_populates="messages")

ChatSession.messages = relationship("ChatMessage", order_by=ChatMessage.id, back_populates="session")

class ChatHistoryManager:
    def __init__(self, db_file, chat_name='default'):
        self.chat_name = chat_name
        self.engine = create_engine(f'sqlite:///{db_file}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    def _get_session_id(self, chat_name):
        chat_session = self.session.query(ChatSession).filter_by(chat_name=chat_name).first()
        if chat_session:
            return chat_session.id
        else:
            new_session = ChatSession(chat_name=chat_name)
            self.session.add(new_session)
            self.session.commit()
            return new_session.id

    def set_chat_name(self, chat_name):
        self.chat_name = chat_name

    def save_history(self, history):
        session_id = self._get_session_id(self.chat_name)
        self.session.query(ChatMessage).filter_by(session_id=session_id).delete()
        for message in history:
            new_message = ChatMessage(
                session_id=session_id,
                role=message['role'],
                content=message['content']
            )
            self.session.add(new_message)
        self.session.commit()

    def load_history(self):
        session_id = self._get_session_id(self.chat_name)
        messages = self.session.query(ChatMessage).filter_by(session_id=session_id).order_by(ChatMessage.timestamp).all()
        return [{'role': message.role, 'content': message.content} for message in messages]

    def clear_history(self):
        session_id = self._get_session_id(self.chat_name)
        self.session.query(ChatMessage).filter_by(session_id=session_id).delete()
        self.session.commit()

    def rename_history(self, new_name):
        session_id = self._get_session_id(self.chat_name)
        chat_session = self.session.query(ChatSession).filter_by(id=session_id).first()
        if chat_session:
            chat_session.chat_name = new_name
            self.session.commit()
            self.set_chat_name(new_name)

    def delete_history(self):
        session_id = self._get_session_id(self.chat_name)
        self.session.query(ChatMessage).filter_by(session_id=session_id).delete()
        self.session.query(ChatSession).filter_by(id=session_id).delete()
        self.session.commit()
        self.set_chat_name('default')

    def list_chats(self):
        chat_sessions = self.session.query(ChatSession).all()
        return [session.chat_name for session in chat_sessions]

