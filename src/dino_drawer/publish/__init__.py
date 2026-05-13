"""Publish pipeline: optimise generated PNGs to WebP and upload to Cloudflare R2.

Submodules
----------
optimize  — Convert master PNGs to multi-resolution WebP files.
r2        — S3-compatible client for Cloudflare R2.
manifest  — Build per-species meta.json and maintain the global catalog.json.
"""
