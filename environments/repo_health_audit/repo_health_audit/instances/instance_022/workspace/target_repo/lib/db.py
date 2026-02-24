from sqlalchemy import create_engine

# Hardcoded connection string — no multi-env support
DATABASE_URL = "postgresql://dataforge:secret@prod-db.internal:5432/warehouse"


def get_engine():
    return create_engine(DATABASE_URL)
