# database.py

from sqlalchemy import create_engine, Column, String, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
import os
import uuid

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")  # Change to your database URL

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ThreadDB(Base):
    __tablename__ = "threads"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doctor_name = Column(String, index=True)
    user_id = Column(String, index=True)
    content = Column(Text)
    messages = Column(JSON)  # To store messages as JSON
    uploaded_files = Column(JSON)  # To store file paths as JSON

# Create the database tables
Base.metadata.create_all(bind=engine)
