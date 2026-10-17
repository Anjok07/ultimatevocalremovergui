"""Headless web-service layer for Ultimate Vocal Remover.

This package wraps the existing separation engines in ``separate.py`` and the
``ModelData`` config object in ``UVR.py`` so they can run without the Tkinter
GUI, exposing them over a FastAPI HTTP API.
"""
