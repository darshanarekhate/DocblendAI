"""SQLAlchemy engine and session setup for the SQLite relational store.

Supports Modules 1, 5, 6, 7: SQLite holds Document, Query, Answer, and
RetrievalResult rows (ChromaDB holds chunk vectors, see vector_store.py).
"""

from collections.abc import Iterator

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

# check_same_thread=False: FastAPI may use a session from a different thread than the one that created it.
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    """FastAPI dependency: yields a session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class SchemaOutdatedError(RuntimeError):
    """The SQLite file predates a model change (create_all never alters existing tables)."""


def check_schema() -> None:
    """Fail fast if an existing table is missing columns the models now define."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    missing = [
        f"{table.name}.{column.name}"
        for table in Base.metadata.sorted_tables
        if table.name in existing_tables
        for column in table.columns
        if column.name not in {c["name"] for c in inspector.get_columns(table.name)}
    ]
    if missing:
        raise SchemaOutdatedError(
            f"The database {engine.url.database} is out of date (missing {', '.join(missing)}). "
            "It holds only local dev data: stop the server, delete that file and the contents of "
            "data/chroma_db (except .gitkeep), restart, and re-upload your documents."
        )


def init_db() -> None:
    """Create all tables, then verify existing ones match the models. Called once on app startup."""
    from app.db import models  # noqa: F401  (registers models on Base.metadata)

    Base.metadata.create_all(bind=engine)
    check_schema()
