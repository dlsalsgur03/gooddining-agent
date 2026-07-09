"""사용자 프로필 장기 저장소.

`InMemoryProfileStore`는 SQLite 기반 구현체로 교체되기 전까지 쓰는 임시 구현체다.
`ProfileStore` 프로토콜만 지키면 `app/agent.py`의 그래프 로직은 건드리지 않고
저장소 구현체만 교체할 수 있다.
"""

from typing import Protocol

from app.schemas import UserProfile


class ProfileStore(Protocol):
    def get(self, user_id: str) -> UserProfile | None: ...

    def save(self, user_id: str, profile: UserProfile) -> None: ...


class InMemoryProfileStore:
    def __init__(self) -> None:
        self._data: dict[str, UserProfile] = {}

    def get(self, user_id: str) -> UserProfile | None:
        return self._data.get(user_id)

    def save(self, user_id: str, profile: UserProfile) -> None:
        self._data[user_id] = profile
