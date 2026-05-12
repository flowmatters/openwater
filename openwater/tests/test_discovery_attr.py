"""Tests for the module-level __getattr__ fallback that reports when no
OpenWater release has been activated."""
import pytest

from openwater import discovery, single, ensemble, lib, nodes


@pytest.fixture
def no_release():
  prev = discovery._release_active
  discovery._release_active = False
  try:
    yield
  finally:
    discovery._release_active = prev


@pytest.fixture
def active_release():
  prev = discovery._release_active
  discovery._release_active = True
  try:
    yield
  finally:
    discovery._release_active = prev


@pytest.mark.parametrize('module', [single, ensemble, lib, nodes])
def test_missing_attr_without_release_raises_no_active_release(module, no_release):
  with pytest.raises(discovery.NoActiveReleaseError) as exc_info:
    getattr(module, 'SomeNonexistentModel')
  msg = str(exc_info.value)
  assert 'No OpenWater release active' in msg
  assert 'SomeNonexistentModel' in msg
  assert module.__name__ in msg


@pytest.mark.parametrize('module', [single, ensemble, lib, nodes])
def test_no_active_release_is_attribute_error(module, no_release):
  # Subclass of AttributeError so hasattr/getattr fallback semantics still work.
  assert issubclass(discovery.NoActiveReleaseError, AttributeError)
  assert not hasattr(module, 'SomeNonexistentModel')


@pytest.mark.parametrize('module', [single, ensemble, lib, nodes])
def test_missing_attr_with_release_raises_plain_attribute_error(module, active_release):
  with pytest.raises(AttributeError) as exc_info:
    getattr(module, 'SomeNonexistentModel')
  assert not isinstance(exc_info.value, discovery.NoActiveReleaseError)
  assert 'No OpenWater release active' not in str(exc_info.value)


@pytest.mark.parametrize('module', [single, ensemble, lib, nodes])
def test_dunder_lookup_does_not_trigger_no_active_release(module, no_release):
  # Pickling/introspection probes for dunder attributes shouldn't be
  # misreported as "no active release".
  with pytest.raises(AttributeError) as exc_info:
    getattr(module, '__nonexistent_dunder__')
  assert not isinstance(exc_info.value, discovery.NoActiveReleaseError)
