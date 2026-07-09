"""레시피/배달 메뉴 Document를 임베딩해 검색 가능한 vectorstore로 만든다.

`build_vectorstore`는 임베딩 객체를 주입받으므로, 테스트에서는 API 키가 필요한
`OpenAIEmbeddings` 대신 `FakeEmbeddings`를 넣어 파이프라인만 검증할 수 있다.
"""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from app.rag.delivery_menu_loader import load_delivery_menu_documents
from app.rag.loader import load_recipe_documents

_recipe_vectorstore: InMemoryVectorStore | None = None
_delivery_menu_vectorstore: InMemoryVectorStore | None = None


def build_vectorstore(documents: list[Document], embeddings: Embeddings) -> InMemoryVectorStore:
    store = InMemoryVectorStore(embeddings)
    store.add_documents(documents)
    return store


def get_recipe_vectorstore() -> InMemoryVectorStore:
    """운영용 진입점. 최초 호출 시 OpenAIEmbeddings로 색인하고 이후 호출은 캐시된 인스턴스를 재사용한다."""
    global _recipe_vectorstore
    if _recipe_vectorstore is None:
        from langchain_openai import OpenAIEmbeddings

        documents = load_recipe_documents()
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        _recipe_vectorstore = build_vectorstore(documents, embeddings)
    return _recipe_vectorstore


def get_delivery_menu_vectorstore() -> InMemoryVectorStore:
    """운영용 진입점. get_recipe_vectorstore와 동일 패턴, 별도 캐시 변수를 사용한다."""
    global _delivery_menu_vectorstore
    if _delivery_menu_vectorstore is None:
        from langchain_openai import OpenAIEmbeddings

        documents = load_delivery_menu_documents()
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        _delivery_menu_vectorstore = build_vectorstore(documents, embeddings)
    return _delivery_menu_vectorstore
