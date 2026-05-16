"""Shared metadata helpers for adapter packages."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .normalization import as_list, compact_dict, get_value, object_to_dict


FieldResolver = str | Sequence[str] | Callable[[Any], Any]


def resolve_value(source: Any, resolver: FieldResolver, *, default: Any = None) -> Any:
    """Resolve one metadata value from a field sequence or callback.

    Parameters
    ----------
    source : Any
        Framework object or mapping.
    resolver : FieldResolver
        Field name, sequence of fallback field names, or callback.
    default : Any
        Default value when nothing is resolved.

    Returns
    -------
    Any
        Resolved value or ``default``.
    """

    if callable(resolver):
        value = resolver(source)
        return default if value is None else value
    if isinstance(resolver, str):
        return get_value(source, resolver, default=default)
    return get_value(source, *resolver, default=default)


def resolve_identifier(source: Any, *names: str) -> str | None:
    """Resolve an identifier from fallback field names.

    Parameters
    ----------
    source : Any
        Framework object or mapping.
    *names : str
        Candidate field names in priority order.

    Returns
    -------
    str | None
        Stringified identifier when present.
    """

    value = get_value(source, *names, default=None)
    return str(value) if value is not None else None


def serialize_object_list(values: Any) -> list[Any]:
    """Serialize a list-like field into compact JSON-ready items.

    Parameters
    ----------
    values : Any
        Sequence-like or scalar value containing framework objects.

    Returns
    -------
    list[Any]
        Serialized items using ``object_to_dict`` with string fallback.
    """

    return [object_to_dict(value) or str(value) for value in as_list(values)]


def build_metadata(source: Any, *, fields: Mapping[str, FieldResolver]) -> dict[str, Any]:
    """Build compact metadata from reusable field resolvers.

    Parameters
    ----------
    source : Any
        Framework object or mapping.
    fields : Mapping[str, FieldResolver]
        Metadata field names mapped to fallback field sequences or callbacks.

    Returns
    -------
    dict[str, Any]
        Compact metadata dictionary with empty values removed.
    """

    return compact_dict({key: resolve_value(source, resolver) for key, resolver in fields.items()})


__all__ = ["FieldResolver", "build_metadata", "resolve_identifier", "resolve_value", "serialize_object_list"]