from qdrant_client.models import (
    FieldCondition,
    Filter,
    IsEmptyCondition,
    IsNullCondition,
    MatchValue,
    MinShould,
    PayloadField,
)

ALL_FILTER_VALUE = "All"
PUBLIC_VISIBILITY = "public"
PRIVATE_VISIBILITY = "private"

def _is_active_filter(value: str | None) -> bool:
    """
    Return True when a frontend filter value should be applied.

    Streamlit dropdowns often send values such as "All" or an empty string to
    represent no filtering. This helper normalizes that behavior so the Qdrant
    filter builder only includes real user-selected filters.
    """
    if value is not None and value.strip() != "" and value != ALL_FILTER_VALUE:
        return True

    return False


def build_chunk_filter(
    tag_name: str | None = None,
    doc_name: str | None = None,
    source_type: str | None = None,
    session_id: str | None = None,
) -> Filter | None:
    """
    Build an optional Qdrant payload filter for Lucid chunk retrieval.

    The returned filter restricts semantic search to chunks whose payload
    metadata matches the selected tag, document, and/or source type. It also
    limits private uploads to the active anonymous session while keeping public
    and legacy chunks visible to everyone.

    Args:
        tag_name: Optional user-created tag, such as "CSC420" or "Midterm".
        doc_name: Optional document name, such as "week3_cnn.pdf".
        source_type: Optional source type, such as "lecture", "exam", or
            "textbook".
        session_id: Optional anonymous browser session ID for private uploads.

    Returns:
        A Qdrant Filter object enforcing metadata and visibility constraints.
    """
    conditions = []

    if _is_active_filter(tag_name):
        conditions.append(
            FieldCondition(
                key="tag_name",
                match=MatchValue(value=tag_name),
            )
        )

    if _is_active_filter(doc_name):
        conditions.append(
            FieldCondition(
                key="doc_name",
                match=MatchValue(value=doc_name),
            )
        )

    if _is_active_filter(source_type):
        conditions.append(
            FieldCondition(
                key="source_type",
                match=MatchValue(value=source_type),
            )
        )

    visibility_conditions = [
        FieldCondition(
            key="visibility",
            match=MatchValue(value=PUBLIC_VISIBILITY),
        ),
        IsEmptyCondition(is_empty=PayloadField(key="visibility")),
        IsNullCondition(is_null=PayloadField(key="visibility")),
    ]

    if session_id:
        visibility_conditions.append(
            Filter(
                must=[
                    FieldCondition(
                        key="visibility",
                        match=MatchValue(value=PRIVATE_VISIBILITY),
                    ),
                    FieldCondition(
                        key="owner_id",
                        match=MatchValue(value=session_id),
                    ),
                ]
            )
        )

    filter_kwargs = {
        "min_should": MinShould(
            conditions=visibility_conditions,
            min_count=1,
        )
    }
    if conditions:
        filter_kwargs["must"] = conditions

    return Filter(
        **filter_kwargs,
    )
