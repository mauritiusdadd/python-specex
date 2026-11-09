#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for specex.exceptions."""
import pytest
from specex.exceptions import (
    get_ipython_embedder,
    exception_handler,
    HAS_IPYTHON
)


def test_get_ipython_embedder():
    embedder = get_ipython_embedder()
    if HAS_IPYTHON:
        assert embedder is not None
    else:
        assert embedder is None


def test_exception_handler(monkeypatch):
    import specex.exceptions as sexc

    # Test non-ipython path
    monkeypatch.setattr(sexc, "HAS_IPYTHON", False)
    with pytest.raises(SystemExit):
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_type, exc_val, exc_tb = sys.exc_info()
            sexc.exception_handler(exc_type, exc_val, exc_tb)

    # Test ipython path without entering interactive pdb loop
    monkeypatch.setattr(sexc, "HAS_IPYTHON", True)
    dummy_tb = lambda t, v, tb: "formatted_tb"
    monkeypatch.setattr(sexc.ultratb, "FormattedTB", lambda **kwargs: dummy_tb)
    try:
        raise RuntimeError("Test error 2")
    except RuntimeError:
        import sys
        exc_type, exc_val, exc_tb = sys.exc_info()
        res = sexc.exception_handler(exc_type, exc_val, exc_tb)
        assert res == "formatted_tb"
