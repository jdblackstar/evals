"""User data store backed by SQLAlchemy."""

import uuid
from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import declarative_base, Session

from libs.common.config import get_database_url

Base = declarative_base()


class UserModel(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True)


class UserStore:
    def __init__(self):
        self.engine = create_engine(get_database_url())
        Base.metadata.create_all(self.engine)

    def create(self, name: str, email: str) -> dict:
        user = UserModel(id=str(uuid.uuid4()), name=name, email=email)
        with Session(self.engine) as session:
            session.add(user)
            session.commit()
            return {"id": user.id, "name": user.name, "email": user.email}

    def get(self, user_id: str) -> dict | None:
        with Session(self.engine) as session:
            user = session.get(UserModel, user_id)
            if user is None:
                return None
            return {"id": user.id, "name": user.name, "email": user.email}

    def list_all(self) -> list[dict]:
        with Session(self.engine) as session:
            users = session.query(UserModel).all()
            return [{"id": u.id, "name": u.name, "email": u.email} for u in users]
