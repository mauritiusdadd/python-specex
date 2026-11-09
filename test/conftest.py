#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pytest session-wide configuration and fixtures."""
import os
import tempfile
from pathlib import Path
import pytest

# Configure a temporary directory inside the project root for tests
# to avoid filling up the small /tmp tmpfs during large operations.
_LOCAL_TMP_DIR = Path(__file__).parent.parent.resolve() / ".tmp"
_LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)

# Set environment variables for any subprocesses or modules
os.environ["TMPDIR"] = str(_LOCAL_TMP_DIR)
os.environ["TEMP"] = str(_LOCAL_TMP_DIR)
os.environ["TMP"] = str(_LOCAL_TMP_DIR)

# Override Python's tempfile module default directory
tempfile.tempdir = str(_LOCAL_TMP_DIR)


@pytest.fixture(scope="session", autouse=True)
def manage_local_tempdir():
    """Ensure local .tmp directory exists."""
    _LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)
    yield _LOCAL_TMP_DIR
