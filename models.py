from sqlalchemy import Column, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class QueryHistory(Base):
    __tablename__ = "query_history"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text)
    result = Column(Text)
    sources = Column(Text)

engine = create_engine("sqlite:///./history.db")
SessionLocal = sessionmaker(bind=engine)
