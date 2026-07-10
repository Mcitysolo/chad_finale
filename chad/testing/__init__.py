"""chad.testing — test-support helpers importable from every test tree.

Currently hosts the G3C-HF repo-write leak guard (``repo_write_guard``). Inert in
production: nothing here is imported by the runtime hot path — it is wired only by the
repo-root ``conftest.py`` during a pytest session.
"""
