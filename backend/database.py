import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Local docker fallback
    DATABASE_URL = "mysql+aiomysql://root:password@db:3306/vibe_db"
    connect_args = {}
else:
    # Aiven Fix: Strip out trailing query strings like '?ssl-mode=REQUIRED'
    if "?" in DATABASE_URL:
        DATABASE_URL = DATABASE_URL.split("?")[0]  # <-- Added [0] to extract the URL string cleanly

    # Enforce SSL connection for Aiven via aiomysql
    connect_args = {"ssl": True}

# Pass connect_args into the engine configuration block
engine = create_async_engine(DATABASE_URL, connect_args=connect_args, echo=False)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()


class SkippedMovie(Base):
    __tablename__ = "skipped_movies"
    id = Column(Integer, primary_key=True, index=True)
    ip_address = Column(String(50), index=True)
    username = Column(String(100), index=True)
    movie_link = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
