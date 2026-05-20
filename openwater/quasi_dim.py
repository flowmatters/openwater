"""Quasi-dimensions: registered lookups that look like model dimensions at the
parameterisation and reporting APIs but are not part of the model graph.

See `py/openwater/doc/design-quasi-dimensions.md` for the design.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd


# On-disk layout version for /META/quasi_dimensions/<name>/. Bump if the
# schema gains required fields (e.g. multi-key support).
_HDF5_LAYOUT_VERSION = 1


_UNSET = object()


class QuasiDimensionError(Exception):
    """Base class for quasi-dimension errors."""


class CycleError(QuasiDimensionError):
    """Raised when registering a quasi-dimension would create a cycle."""


class NameCollisionError(QuasiDimensionError):
    """Raised when a quasi-dim name collides with an existing real or quasi dim."""


class UnknownDimensionError(QuasiDimensionError):
    """Raised when a chain terminates without reaching a real dimension."""


class PartialCoverageError(QuasiDimensionError):
    """Raised when resolving encounters key values not present in the mapping."""


class QuasiDimension:
    """A named 1:1 mapping from one dimension's values to derived values.

    Parameters
    ----------
    name : str
        Identifier used at call sites.
    keyed_by : str
        Dimension this quasi-dim is a function of. May name a real dimension
        or another quasi-dimension (chaining).
    mapping : pd.Series
        Index = `keyed_by` values, data = derived values. Index must be unique.
    default : optional
        If set, missing keys resolve to this value instead of raising.
    """

    __slots__ = ("name", "keyed_by", "mapping", "default")

    def __init__(
        self,
        name: str,
        keyed_by: str,
        mapping: pd.Series,
        default: Any = _UNSET,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(keyed_by, str) or not keyed_by:
            raise ValueError("keyed_by must be a non-empty string")
        if name == keyed_by:
            raise CycleError(
                f"quasi-dim {name!r} cannot be keyed by itself"
            )
        if not isinstance(mapping, pd.Series):
            raise TypeError("mapping must be a pandas.Series")
        if not mapping.index.is_unique:
            dupes = mapping.index[mapping.index.duplicated()].unique().tolist()
            raise ValueError(
                f"mapping index must be unique (quasi-dim must be 1:1); "
                f"duplicates: {dupes}"
            )
        _check_hashable(mapping.index, "index")
        _check_hashable(mapping.values, "value")

        self.name = name
        self.keyed_by = keyed_by
        # Copy + relabel so external mutation of the input doesn't leak in.
        self.mapping = mapping.copy()
        self.mapping.index = self.mapping.index.rename(keyed_by)
        self.mapping.name = name
        self.default = default

    @property
    def has_default(self) -> bool:
        return self.default is not _UNSET

    def project(self, values: pd.Series) -> pd.Series:
        """Apply the mapping to a Series of `keyed_by` values.

        Raises `PartialCoverageError` if any value is unmapped and no default is set.
        """
        out = values.map(self.mapping)
        missing = out.isna() & ~values.isna()
        if missing.any():
            if self.has_default:
                out = out.where(~missing, self.default)
            else:
                unmapped = sorted(set(values[missing].tolist()))
                raise PartialCoverageError(
                    f"quasi-dim {self.name!r} has no mapping for "
                    f"{self.keyed_by} value(s): {unmapped}"
                )
        out.name = self.name
        return out

    def __repr__(self) -> str:
        d = "" if not self.has_default else f", default={self.default!r}"
        return (
            f"QuasiDimension(name={self.name!r}, keyed_by={self.keyed_by!r}, "
            f"size={len(self.mapping)}{d})"
        )

    @classmethod
    def from_source(cls, source, *, name=None, keyed_by=None,
                    key=None, value=None, default=_UNSET) -> 'QuasiDimension':
        '''Construct a QuasiDimension from any supported source type.

        ``source`` may be:
          * a ``QuasiDimension`` (returned as-is; ``name``/``keyed_by``/
            ``default`` overrides are not accepted in this case)
          * a path (``str``) to a CSV — requires ``key`` and ``value``;
            ``name``/``keyed_by`` default to the value/key columns
          * a ``pandas.Series`` — ``name``/``keyed_by`` default to the
            Series name and index name
          * a ``dict`` — ``name`` and ``keyed_by`` are required

        Centralises the polymorphic dispatch so ``ModelGraph`` and ``ModelFile``
        (or any other owner of a registry) can share one entry point.
        '''
        default_kwargs = {} if default is _UNSET else {'default': default}

        if isinstance(source, QuasiDimension):
            if (name is not None or keyed_by is not None or
                key is not None or value is not None or default is not _UNSET):
                raise ValueError(
                    'overrides not accepted when source is a QuasiDimension')
            return source
        if isinstance(source, str):
            if key is None or value is None:
                raise ValueError(
                    'from_source(path, ...) requires key= and value=')
            return from_csv(source, key=key, value=value,
                            name=name, keyed_by=keyed_by, **default_kwargs)
        if isinstance(source, pd.Series):
            return from_series(source, name=name, keyed_by=keyed_by,
                               **default_kwargs)
        if isinstance(source, dict):
            if name is None or keyed_by is None:
                raise ValueError(
                    'from_source(dict, ...) requires name= and keyed_by=')
            return from_dict(source, name=name, keyed_by=keyed_by,
                             **default_kwargs)
        raise TypeError(
            f'unsupported source type for QuasiDimension.from_source: '
            f'{type(source).__name__}')


def _check_hashable(values: Iterable, kind: str) -> None:
    for v in values:
        try:
            hash(v)
        except TypeError as exc:
            raise TypeError(f"{kind} value {v!r} is not hashable") from exc


# ---------------------------------------------------------------------------
# Standalone loaders
# ---------------------------------------------------------------------------

def from_series(
    series: pd.Series,
    name: Optional[str] = None,
    keyed_by: Optional[str] = None,
    default: Any = _UNSET,
) -> QuasiDimension:
    """Build a QuasiDimension from a pandas Series.

    `name` defaults to `series.name`; `keyed_by` defaults to `series.index.name`.
    Either or both may be overridden.
    """
    name = name if name is not None else series.name
    keyed_by = keyed_by if keyed_by is not None else series.index.name
    if name is None:
        raise ValueError(
            "name not supplied and series has no name; pass name=..."
        )
    if keyed_by is None:
        raise ValueError(
            "keyed_by not supplied and series.index has no name; pass keyed_by=..."
        )
    return QuasiDimension(name=name, keyed_by=keyed_by, mapping=series, default=default)


def from_dict(
    d: dict,
    name: str,
    keyed_by: str,
    default: Any = _UNSET,
) -> QuasiDimension:
    """Build a QuasiDimension from a {key: value} dict."""
    series = pd.Series(d)
    return QuasiDimension(name=name, keyed_by=keyed_by, mapping=series, default=default)


def _is_set_value(v) -> bool:
    '''True if v is a non-string iterable (set-membership constraint).'''
    if isinstance(v, str):
        return False
    return isinstance(v, (list, tuple, set, frozenset, np.ndarray))


def _as_set(v) -> set:
    return set(v) if _is_set_value(v) else {v}


class QuasiDimResolver:
    '''Resolve quasi-dim names against a registry, for use by callers that
    consume real-dim names (parameterisers, reporting).

    Parameters
    ----------
    real_dim_names : callable -> Iterable[str]
        Returns the set of real dimension names in the current context.
    registry : QuasiDimRegistry
        The quasi-dim registry to resolve against.
    '''

    def __init__(self, real_dim_names: Callable[[], Iterable[str]],
                 registry: 'QuasiDimRegistry'):
        self._real_dim_names = real_dim_names
        self.registry = registry

    def real_dims(self) -> set:
        return set(self._real_dim_names())

    def is_real(self, name: str) -> bool:
        return name in self.real_dims()

    def is_quasi(self, name: str) -> bool:
        return self.registry.is_quasi(name)

    def project_index(self, name: str) -> 'QuasiDimension':
        '''Return a QuasiDimension keyed by the real dim that ``name`` resolves to.

        Walks the chain and composes the mappings. Raises if ``name`` is not
        a registered quasi-dim.
        '''
        _, composed = self.registry.resolve_chain(name)
        return composed

    def extend_nodes_df(self, nodes_df: pd.DataFrame,
                        quasi_names: Iterable[str]) -> pd.DataFrame:
        '''Return a copy of ``nodes_df`` with extra columns for each named quasi-dim.

        Each new column maps the corresponding real-dim values (already
        present as a column in ``nodes_df``) through the quasi-dim's composed
        projection. The original DataFrame is not mutated.
        '''
        names = list(quasi_names)
        if not names:
            return nodes_df
        out = nodes_df.copy()
        for name in names:
            _, composed = self.registry.resolve_chain(name)
            real_dim = composed.keyed_by
            if real_dim not in out.columns:
                raise QuasiDimensionError(
                    f"cannot project quasi-dim {name!r}: real dim "
                    f"{real_dim!r} not present in nodes_df"
                )
            projected = composed.project(out[real_dim])
            out[name] = projected.values
        return out

    def resolve_constraints(self, constraints: dict) -> dict:
        '''Expand quasi-dim constraints to real-dim constraints.

        For each constraint key that is a quasi-dim, look up the real dim
        it resolves to and convert the value(s) to the corresponding set of
        real-dim values. If the same real dim appears both directly and via
        a quasi-dim expansion, the resulting constraint is the intersection.

        Unknown names (neither real nor quasi) pass through unchanged so
        downstream code can raise its existing error.
        '''
        out: dict = {}
        # Track which real-dim keys were introduced by quasi-dim expansion;
        # multiple quasi-dims hitting the same real dim must intersect.
        for k, v in constraints.items():
            if not self.is_quasi(k):
                out = self._merge_constraint(out, k, _as_set(v) if _is_set_value(v) else v)
                continue
            _, composed = self.registry.resolve_chain(k)
            mapping = composed.mapping
            allowed = _as_set(v)
            matching = mapping.index[mapping.isin(allowed)].tolist()
            if not matching:
                raise QuasiDimensionError(
                    f"no values of {composed.keyed_by!r} map to "
                    f"{k}={sorted(allowed) if len(allowed) > 1 else next(iter(allowed))!r}"
                )
            out = self._merge_constraint(out, composed.keyed_by, set(matching))
        return out

    @staticmethod
    def _merge_constraint(constraints: dict, key: str, value) -> dict:
        '''Add or intersect a constraint, preserving scalar form where possible.'''
        if key not in constraints:
            constraints[key] = value
            return constraints
        existing = constraints[key]
        existing_set = _as_set(existing)
        new_set = _as_set(value)
        merged = existing_set & new_set
        if not merged:
            raise QuasiDimensionError(
                f"constraints on {key!r} have empty intersection: "
                f"{sorted(existing_set)} ∩ {sorted(new_set)}"
            )
        # Preserve scalar form when the intersection is a single value AND
        # the original wasn't explicitly a set.
        if len(merged) == 1 and not _is_set_value(existing) and not _is_set_value(value):
            constraints[key] = next(iter(merged))
        else:
            constraints[key] = sorted(merged) if all(
                isinstance(x, str) for x in merged
            ) else list(merged)
        return constraints


def add_to_registry(registry: 'QuasiDimRegistry', source, *,
                    name=None, keyed_by=None, key=None, value=None,
                    default=None, persist=False) -> 'QuasiDimension':
    '''Build a QuasiDimension from a polymorphic ``source`` and register it.

    Shared by ``ModelGraph.add_quasi_dim``, ``ModelFile.add_quasi_dim`` and
    ``OpenwaterResults.add_quasi_dim`` so the dispatch and registration live
    in one place. Owner-specific concerns (e.g. flushing to disk after a
    persisted add) stay on the owner.

    Returns the registered ``QuasiDimension`` so callers can introspect it.
    '''
    qd = QuasiDimension.from_source(
        source, name=name, keyed_by=keyed_by,
        key=key, value=value,
        **({} if default is None else {'default': default}),
    )
    registry.add(qd, persist=persist)
    return qd


def _encode_for_h5(arr: np.ndarray) -> np.ndarray:
    '''Convert a numpy array to a form h5py can write directly.

    Objects/strings become bytes; numeric arrays pass through unchanged.
    '''
    if arr.dtype.kind in ('U', 'O'):
        return np.array([np.bytes_(str(v)) for v in arr])
    return arr


def _decode_from_h5(arr: np.ndarray):
    '''Inverse of _encode_for_h5: decode bytes back to str; leave numerics alone.'''
    if arr.dtype.kind == 'S':
        return np.array([v.decode('utf-8') for v in arr])
    return arr


def _write_one_quasi_dim(parent_grp, qd: 'QuasiDimension') -> None:
    grp = parent_grp.create_group(qd.name)
    grp.attrs['keyed_by'] = qd.keyed_by
    grp.attrs['version'] = _HDF5_LAYOUT_VERSION
    if qd.has_default:
        # Store default with its native dtype where possible. Strings get
        # serialised to bytes so the read path can detect & decode.
        if isinstance(qd.default, str):
            grp.attrs['default'] = np.bytes_(qd.default)
            grp.attrs['default_is_str'] = True
        else:
            grp.attrs['default'] = qd.default
            grp.attrs['default_is_str'] = False
    keys = _encode_for_h5(np.asarray(qd.mapping.index.tolist()))
    values = _encode_for_h5(np.asarray(qd.mapping.tolist()))
    grp.create_dataset('keys', data=keys)
    grp.create_dataset('values', data=values)


def _read_one_quasi_dim(grp, name: str) -> 'QuasiDimension':
    version = int(grp.attrs.get('version', _HDF5_LAYOUT_VERSION))
    if version > _HDF5_LAYOUT_VERSION:
        raise QuasiDimensionError(
            f'quasi-dim {name!r} has on-disk version {version}, '
            f'newer than this code supports ({_HDF5_LAYOUT_VERSION})'
        )
    keyed_by = grp.attrs['keyed_by']
    if isinstance(keyed_by, bytes):
        keyed_by = keyed_by.decode('utf-8')
    keys = _decode_from_h5(grp['keys'][...])
    values = _decode_from_h5(grp['values'][...])
    series = pd.Series(values, index=keys)
    kwargs = {}
    if 'default' in grp.attrs:
        default = grp.attrs['default']
        if grp.attrs.get('default_is_str') and isinstance(default, bytes):
            default = default.decode('utf-8')
        kwargs['default'] = default
    return QuasiDimension(name=name, keyed_by=keyed_by, mapping=series, **kwargs)


def from_csv(
    path: str,
    key: str,
    value: str,
    name: Optional[str] = None,
    keyed_by: Optional[str] = None,
    default: Any = _UNSET,
    **read_csv_kwargs,
) -> QuasiDimension:
    """Build a QuasiDimension from a CSV file with `key` and `value` columns.

    `name` defaults to the `value` column header; `keyed_by` defaults to the
    `key` column header.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, **read_csv_kwargs)
    for col in (key, value):
        if col not in df.columns:
            raise ValueError(
                f"column {col!r} not found in {path}; columns={list(df.columns)}"
            )
    series = df.set_index(key)[value]
    return from_series(
        series,
        name=name if name is not None else value,
        keyed_by=keyed_by if keyed_by is not None else key,
        default=default,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class QuasiDimRegistry:
    """Holds registered quasi-dimensions for a model/results context.

    Parameters
    ----------
    real_dim_names : callable -> set[str]
        Supplied by the owner (template or results). Called on each add/lookup
        so the registry sees up-to-date real-dim names without holding a
        stale snapshot.
    """

    def __init__(self, real_dim_names: Callable[[], Iterable[str]]):
        self._real_dim_names = real_dim_names
        self._by_name: dict[str, QuasiDimension] = {}
        # Track persistence intent per quasi-dim. Owned by registry to keep
        # the template's add/remove API thin.
        self._persist: dict[str, bool] = {}

    # -- introspection ------------------------------------------------------

    def names(self) -> list[str]:
        return list(self._by_name.keys())

    def items(self):
        return self._by_name.items()

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> QuasiDimension:
        return self._by_name[name]

    def __len__(self) -> int:
        return len(self._by_name)

    def is_persisted(self, name: str) -> bool:
        return self._persist.get(name, False)

    # -- mutation -----------------------------------------------------------

    def add(self, qd: QuasiDimension, persist: bool = False) -> None:
        real = set(self._real_dim_names())
        if qd.name in real:
            raise NameCollisionError(
                f"quasi-dim name {qd.name!r} collides with real dimension"
            )
        if qd.name in self._by_name:
            raise NameCollisionError(
                f"quasi-dim {qd.name!r} already registered; "
                f"remove_quasi_dim() first to replace"
            )
        # Cycle check: walk keyed_by, considering this new qd as part of the
        # graph. A cycle exists if we revisit a name we've already seen.
        # Forward references (keyed_by names a quasi-dim not yet registered)
        # are allowed at registration time; resolution will raise later if
        # the chain doesn't terminate in a real dim.
        self._check_no_cycle(qd, real)
        self._by_name[qd.name] = qd
        self._persist[qd.name] = persist

    def remove(self, name: str) -> None:
        if name not in self._by_name:
            raise KeyError(name)
        # Block removal if another quasi-dim depends on this one — otherwise
        # the dependent's chain breaks silently.
        dependents = [n for n, q in self._by_name.items() if q.keyed_by == name]
        if dependents:
            raise QuasiDimensionError(
                f"cannot remove quasi-dim {name!r}: depended on by "
                f"{dependents}"
            )
        del self._by_name[name]
        self._persist.pop(name, None)

    def _check_no_cycle(self, qd: QuasiDimension, real: set[str]) -> None:
        seen = {qd.name}
        cur = qd.keyed_by
        while cur not in real:
            if cur in seen:
                raise CycleError(
                    f"registering {qd.name!r} would create a cycle "
                    f"via {' -> '.join(list(seen) + [cur])}"
                )
            if cur not in self._by_name:
                # forward reference; chain may complete once `cur` is added
                return
            seen.add(cur)
            cur = self._by_name[cur].keyed_by

    # -- resolution ---------------------------------------------------------

    def is_quasi(self, name: str) -> bool:
        return name in self._by_name

    def write_to_h5(self, meta_group) -> None:
        '''Serialise persisted quasi-dims under <meta_group>/quasi_dimensions/<name>/.

        Only quasi-dims registered with ``persist=True`` are written. Any
        existing ``quasi_dimensions`` subtree is replaced — including being
        removed entirely when there are no persisted quasi-dims left.
        '''
        persisted = [n for n in self._by_name if self._persist.get(n)]
        if 'quasi_dimensions' in meta_group:
            del meta_group['quasi_dimensions']
        if not persisted:
            return
        qd_grp = meta_group.create_group('quasi_dimensions')
        for name in persisted:
            qd = self._by_name[name]
            _write_one_quasi_dim(qd_grp, qd)

    def load_from_h5(self, meta_group) -> None:
        '''Rehydrate the registry from <meta_group>/quasi_dimensions/.

        Each loaded quasi-dim is registered with ``persist=True``. No-op if
        the subtree is absent.
        '''
        if meta_group is None or 'quasi_dimensions' not in meta_group:
            return
        qd_grp = meta_group['quasi_dimensions']
        # Forward references between quasi-dims are allowed at registration,
        # so insertion order doesn't matter.
        for name in qd_grp:
            qd = _read_one_quasi_dim(qd_grp[name], name)
            self.add(qd, persist=True)

    def resolve_chain(self, name: str) -> tuple[str, QuasiDimension]:
        """Walk a quasi-dim's chain to a real dimension, composing the mapping.

        Returns `(real_dim_name, composed_quasi_dim)` where the composed
        quasi-dim has `keyed_by == real_dim_name` and a mapping that takes
        a real-dim value directly to the original quasi-dim's value.

        Raises CycleError or UnknownDimensionError on a broken chain.
        """
        real = set(self._real_dim_names())
        if name in real:
            raise ValueError(
                f"{name!r} is a real dimension; resolve_chain is for quasi-dims"
            )
        if name not in self._by_name:
            raise KeyError(name)

        chain: list[QuasiDimension] = []
        seen: set[str] = set()
        cur_name = name
        while cur_name not in real:
            if cur_name in seen:
                raise CycleError(
                    f"cycle in quasi-dim chain at {cur_name!r}; "
                    f"path: {[q.name for q in chain]}"
                )
            if cur_name not in self._by_name:
                raise UnknownDimensionError(
                    f"chain from {name!r} reaches {cur_name!r} which is "
                    f"neither a real dimension nor a registered quasi-dim"
                )
            seen.add(cur_name)
            qd = self._by_name[cur_name]
            chain.append(qd)
            cur_name = qd.keyed_by

        real_dim = cur_name
        # Compose: start from the innermost (closest to the real dim) mapping
        # and project successively outward.
        innermost = chain[-1]
        composed = innermost.mapping.copy()
        composed.index = composed.index.rename(real_dim)
        for qd in reversed(chain[:-1]):
            composed = qd.project(composed)
        composed.name = name
        # Wrap into a QuasiDimension keyed by the real dim. Default propagates
        # from the outermost link.
        outermost = chain[0]
        kwargs = {}
        if outermost.has_default:
            kwargs["default"] = outermost.default
        return real_dim, QuasiDimension(
            name=name, keyed_by=real_dim, mapping=composed, **kwargs
        )
