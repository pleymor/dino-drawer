"""S3-compatible client for Cloudflare R2."""
from __future__ import annotations

import os
from pathlib import Path

import boto3
from botocore.config import Config


class R2Error(RuntimeError):
    """Raised on R2 upload/download failures or misconfiguration."""


class R2Client:
    """Thin boto3 wrapper for Cloudflare R2.

    Reads credentials and endpoint from environment variables:

    - ``R2_BUCKET`` — name of the R2 bucket.
    - ``R2_ACCESS_KEY_ID`` — R2 API token key ID.
    - ``R2_SECRET_ACCESS_KEY`` — R2 API token secret.
    - ``R2_ENDPOINT`` — ``https://<accountid>.r2.cloudflarestorage.com``.
    - ``R2_PUBLIC_BASE_URL`` — public CDN base URL (no trailing slash).

    Raises
    ------
    R2Error
        If any of the required environment variables are absent.
    """

    def __init__(self) -> None:
        for var in (
            "R2_BUCKET",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_ENDPOINT",
            "R2_PUBLIC_BASE_URL",
        ):
            if not os.environ.get(var):
                raise R2Error(f"Missing env var: {var}")
        self.bucket = os.environ["R2_BUCKET"]
        self.public_base = os.environ["R2_PUBLIC_BASE_URL"].rstrip("/")
        self._s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            config=Config(signature_version="s3v4", region_name="auto"),
        )

    def upload_file(
        self,
        src: Path,
        key: str,
        *,
        content_type: str,
        cache_control: str = "public, max-age=31536000, immutable",
    ) -> str:
        """Upload *src* to the bucket under *key* and return the public URL.

        Parameters
        ----------
        src:
            Local file path to upload.
        key:
            Destination object key inside the bucket.
        content_type:
            MIME type sent as ``ContentType`` metadata.
        cache_control:
            ``Cache-Control`` header value (default: one-year immutable).
        """
        self._s3.upload_file(
            str(src),
            self.bucket,
            key,
            ExtraArgs={"ContentType": content_type, "CacheControl": cache_control},
        )
        return f"{self.public_base}/{key}"

    def upload_bytes(
        self,
        data: bytes,
        key: str,
        *,
        content_type: str,
        cache_control: str = "public, max-age=60",
    ) -> str:
        """Upload raw *data* to the bucket under *key* and return the public URL.

        Parameters
        ----------
        data:
            Bytes to upload.
        key:
            Destination object key inside the bucket.
        content_type:
            MIME type sent as ``ContentType`` metadata.
        cache_control:
            ``Cache-Control`` header value (default: 60-second cache for JSON).
        """
        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            CacheControl=cache_control,
        )
        return f"{self.public_base}/{key}"

    def delete_prefix(self, prefix: str) -> int:
        """Delete every object whose key starts with *prefix*.

        Objects are collected via ``list_objects_v2`` pagination and deleted in
        batches of 1 000 (the S3 ``delete_objects`` maximum).

        Parameters
        ----------
        prefix:
            Key prefix to match, e.g. ``"tyrannosaurus-rex/"``.

        Returns
        -------
        int
            Number of objects deleted.
        """
        paginator = self._s3.get_paginator("list_objects_v2")
        keys: list[dict] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                keys.append({"Key": obj["Key"]})
        if not keys:
            return 0
        deleted = 0
        for i in range(0, len(keys), 1000):
            chunk = keys[i : i + 1000]
            resp = self._s3.delete_objects(Bucket=self.bucket, Delete={"Objects": chunk})
            deleted += len(resp.get("Deleted", []))
        return deleted

    def get_bytes(self, key: str) -> bytes | None:
        """Return the body of object *key*, or ``None`` if the key does not exist.

        Parameters
        ----------
        key:
            Object key to fetch from the bucket.
        """
        from botocore.exceptions import ClientError

        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise
