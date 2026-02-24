from dataclasses import dataclass


@dataclass
class UserContext:
    user_id: str
    roles: list[str] | None = None


@dataclass
class AuditEvent:
    actor: str
    action: str
    resource: str
