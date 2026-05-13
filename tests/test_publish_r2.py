"""Tests for dino_drawer.publish.r2 — Cloudflare R2 S3-compatible client."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


_ENV = {
    "R2_BUCKET": "test-bucket",
    "R2_ACCESS_KEY_ID": "fake-key",
    "R2_SECRET_ACCESS_KEY": "fake-secret",
    "R2_ENDPOINT": "https://test.r2.cloudflarestorage.com",
    "R2_PUBLIC_BASE_URL": "https://pub.r2.dev",
}


def _make_client(mock_boto3_client: MagicMock) -> "R2Client":  # noqa: F821
    """Instantiate R2Client with the patched boto3."""
    from dino_drawer.publish.r2 import R2Client

    return R2Client()


class TestR2ClientInit:
    """Tests for R2Client construction and env-var validation."""

    def test_raises_r2error_when_env_var_missing(self) -> None:
        """R2Client should raise R2Error if any required env var is absent."""
        from dino_drawer.publish.r2 import R2Error

        env = {k: v for k, v in _ENV.items() if k != "R2_BUCKET"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(R2Error, match="R2_BUCKET"):
                from dino_drawer.publish.r2 import R2Client
                R2Client()

    @pytest.mark.parametrize("missing_var", list(_ENV.keys()))
    def test_raises_for_each_missing_var(self, missing_var: str) -> None:
        """R2Error should name the missing variable."""
        from dino_drawer.publish.r2 import R2Client, R2Error

        env = {k: v for k, v in _ENV.items() if k != missing_var}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(R2Error, match=missing_var):
                R2Client()

    def test_creates_boto3_client_with_correct_args(self) -> None:
        """R2Client should pass endpoint, credentials and region to boto3.client."""
        with patch.dict(os.environ, _ENV, clear=True):
            with patch("boto3.client") as mock_client:
                from importlib import reload
                import dino_drawer.publish.r2 as r2_mod
                reload(r2_mod)
                r2_mod.R2Client()
                mock_client.assert_called_once()
                _, kwargs = mock_client.call_args
                assert kwargs["endpoint_url"] == _ENV["R2_ENDPOINT"]
                assert kwargs["aws_access_key_id"] == _ENV["R2_ACCESS_KEY_ID"]
                assert kwargs["aws_secret_access_key"] == _ENV["R2_SECRET_ACCESS_KEY"]


class TestUploadFile:
    """Tests for R2Client.upload_file."""

    def test_calls_upload_file_with_correct_args(self, tmp_path: Path) -> None:
        """upload_file should delegate to s3.upload_file with bucket, key and ExtraArgs."""
        src = tmp_path / "hero@1600.webp"
        src.write_bytes(b"fake-webp")

        with patch.dict(os.environ, _ENV, clear=True):
            with patch("boto3.client") as mock_boto3:
                mock_s3 = MagicMock()
                mock_boto3.return_value = mock_s3

                from importlib import reload
                import dino_drawer.publish.r2 as r2_mod
                reload(r2_mod)
                client = r2_mod.R2Client()
                url = client.upload_file(src, "tyrannosaurus-rex/hero@1600.webp",
                                         content_type="image/webp")

                mock_s3.upload_file.assert_called_once_with(
                    str(src),
                    "test-bucket",
                    "tyrannosaurus-rex/hero@1600.webp",
                    ExtraArgs={
                        "ContentType": "image/webp",
                        "CacheControl": "public, max-age=31536000, immutable",
                    },
                )
                assert url == "https://pub.r2.dev/tyrannosaurus-rex/hero@1600.webp"

    def test_returns_public_url(self, tmp_path: Path) -> None:
        """upload_file should return the public URL composed from base URL and key."""
        src = tmp_path / "file.webp"
        src.write_bytes(b"x")

        with patch.dict(os.environ, _ENV, clear=True):
            with patch("boto3.client") as mock_boto3:
                mock_s3 = MagicMock()
                mock_boto3.return_value = mock_s3

                from importlib import reload
                import dino_drawer.publish.r2 as r2_mod
                reload(r2_mod)
                client = r2_mod.R2Client()
                url = client.upload_file(src, "slug/image.webp", content_type="image/webp")

                assert url == "https://pub.r2.dev/slug/image.webp"


class TestUploadBytes:
    """Tests for R2Client.upload_bytes."""

    def test_calls_put_object_with_correct_args(self) -> None:
        """upload_bytes should call s3.put_object with all required parameters."""
        with patch.dict(os.environ, _ENV, clear=True):
            with patch("boto3.client") as mock_boto3:
                mock_s3 = MagicMock()
                mock_boto3.return_value = mock_s3

                from importlib import reload
                import dino_drawer.publish.r2 as r2_mod
                reload(r2_mod)
                client = r2_mod.R2Client()
                data = b'{"hello": "world"}'
                url = client.upload_bytes(data, "slug/meta.json", content_type="application/json")

                mock_s3.put_object.assert_called_once_with(
                    Bucket="test-bucket",
                    Key="slug/meta.json",
                    Body=data,
                    ContentType="application/json",
                    CacheControl="public, max-age=60",
                )
                assert url == "https://pub.r2.dev/slug/meta.json"


class TestDeletePrefix:
    """Tests for R2Client.delete_prefix."""

    def test_deletes_listed_objects(self) -> None:
        """delete_prefix should call delete_objects for every listed key."""
        with patch.dict(os.environ, _ENV, clear=True):
            with patch("boto3.client") as mock_boto3:
                mock_s3 = MagicMock()
                mock_boto3.return_value = mock_s3

                # Simulate paginator returning one page with two objects
                mock_paginator = MagicMock()
                mock_s3.get_paginator.return_value = mock_paginator
                mock_paginator.paginate.return_value = [
                    {"Contents": [{"Key": "slug/a.webp"}, {"Key": "slug/b.webp"}]}
                ]
                mock_s3.delete_objects.return_value = {
                    "Deleted": [{"Key": "slug/a.webp"}, {"Key": "slug/b.webp"}]
                }

                from importlib import reload
                import dino_drawer.publish.r2 as r2_mod
                reload(r2_mod)
                client = r2_mod.R2Client()
                count = client.delete_prefix("slug/")

                mock_s3.delete_objects.assert_called_once_with(
                    Bucket="test-bucket",
                    Delete={"Objects": [{"Key": "slug/a.webp"}, {"Key": "slug/b.webp"}]},
                )
                assert count == 2

    def test_returns_zero_when_prefix_empty(self) -> None:
        """delete_prefix should return 0 when there are no objects under prefix."""
        with patch.dict(os.environ, _ENV, clear=True):
            with patch("boto3.client") as mock_boto3:
                mock_s3 = MagicMock()
                mock_boto3.return_value = mock_s3

                mock_paginator = MagicMock()
                mock_s3.get_paginator.return_value = mock_paginator
                mock_paginator.paginate.return_value = [{"Contents": []}]

                from importlib import reload
                import dino_drawer.publish.r2 as r2_mod
                reload(r2_mod)
                client = r2_mod.R2Client()
                count = client.delete_prefix("nonexistent/")

                mock_s3.delete_objects.assert_not_called()
                assert count == 0

    def test_paginates_over_multiple_pages(self) -> None:
        """delete_prefix should process all pages from the paginator."""
        with patch.dict(os.environ, _ENV, clear=True):
            with patch("boto3.client") as mock_boto3:
                mock_s3 = MagicMock()
                mock_boto3.return_value = mock_s3

                # Two pages of results
                mock_paginator = MagicMock()
                mock_s3.get_paginator.return_value = mock_paginator
                mock_paginator.paginate.return_value = [
                    {"Contents": [{"Key": "slug/a.webp"}]},
                    {"Contents": [{"Key": "slug/b.webp"}]},
                ]
                mock_s3.delete_objects.return_value = {
                    "Deleted": [{"Key": "slug/a.webp"}, {"Key": "slug/b.webp"}]
                }

                from importlib import reload
                import dino_drawer.publish.r2 as r2_mod
                reload(r2_mod)
                client = r2_mod.R2Client()
                count = client.delete_prefix("slug/")

                # Both pages collected then deleted in a single batch
                assert count == 2


class TestGetBytes:
    """Tests for R2Client.get_bytes."""

    def test_returns_body_bytes(self) -> None:
        """get_bytes should return the object body on success."""
        with patch.dict(os.environ, _ENV, clear=True):
            with patch("boto3.client") as mock_boto3:
                mock_s3 = MagicMock()
                mock_boto3.return_value = mock_s3

                mock_body = MagicMock()
                mock_body.read.return_value = b"catalog-data"
                mock_s3.get_object.return_value = {"Body": mock_body}

                from importlib import reload
                import dino_drawer.publish.r2 as r2_mod
                reload(r2_mod)
                client = r2_mod.R2Client()
                result = client.get_bytes("catalog.json")

                assert result == b"catalog-data"

    def test_returns_none_on_no_such_key(self) -> None:
        """get_bytes should return None when the key does not exist."""
        from botocore.exceptions import ClientError

        with patch.dict(os.environ, _ENV, clear=True):
            with patch("boto3.client") as mock_boto3:
                mock_s3 = MagicMock()
                mock_boto3.return_value = mock_s3

                # Simulate a NoSuchKey ClientError from botocore
                mock_s3.get_object.side_effect = ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
                    "GetObject",
                )

                from importlib import reload
                import dino_drawer.publish.r2 as r2_mod
                reload(r2_mod)
                client = r2_mod.R2Client()
                result = client.get_bytes("missing.json")

                assert result is None
