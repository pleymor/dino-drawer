"""Dino Drawer — scientific infographics for prehistoric species.

Importing this package auto-loads a `.env` file from the current working
directory (or any parent) into the process environment, so secrets like
`HF_TOKEN` don't have to be sourced manually.
"""
from dotenv import load_dotenv

load_dotenv()

__version__ = "0.1.0"
