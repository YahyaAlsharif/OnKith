"""Repo scripts, as a real package.

Without this file ``scripts`` is a namespace package, and a regular package of
the same name always wins over one. ``nemo_toolkit`` installs exactly that -
``site-packages/scripts/__init__.py`` - so ``from scripts.build_release_zip
import ...`` resolved into NeMo's directory and two test modules failed to
import on any machine with NeMo present, which includes the Raspberry Pi.
"""
