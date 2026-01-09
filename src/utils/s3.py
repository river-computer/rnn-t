"""S3 utilities for checkpoint save/load.

Provides:
- Upload checkpoints to S3
- Download checkpoints from S3
- S3Checkpoint class for transparent save/load
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import json

import torch
import boto3
from botocore.exceptions import ClientError


def get_s3_client():
    """Get configured S3 client."""
    return boto3.client('s3')


def upload_checkpoint(
    checkpoint: Dict[str, Any],
    s3_path: str,
    local_path: Optional[str] = None,
) -> str:
    """Upload checkpoint to S3.

    Args:
        checkpoint: Checkpoint dictionary to save
        s3_path: S3 path (s3://bucket/key or bucket/key)
        local_path: Optional local path to also save to

    Returns:
        S3 URI of uploaded checkpoint
    """
    # Parse S3 path
    if s3_path.startswith('s3://'):
        s3_path = s3_path[5:]
    bucket, key = s3_path.split('/', 1)

    # Save locally first
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
        torch.save(checkpoint, f)

    try:
        # Upload to S3
        s3 = get_s3_client()
        s3.upload_file(temp_path, bucket, key)
        s3_uri = f's3://{bucket}/{key}'
        print(f"Uploaded checkpoint to {s3_uri}")

        # Optionally save local copy
        if local_path:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            torch.save(checkpoint, local_path)
            print(f"Saved local copy to {local_path}")

        return s3_uri

    finally:
        # Clean up temp file
        os.unlink(temp_path)


def download_checkpoint(
    s3_path: str,
    local_path: Optional[str] = None,
    map_location: str = 'cpu',
) -> Dict[str, Any]:
    """Download checkpoint from S3.

    Args:
        s3_path: S3 path (s3://bucket/key or bucket/key)
        local_path: Optional local path to save to
        map_location: Device to load tensors to

    Returns:
        Loaded checkpoint dictionary
    """
    # Parse S3 path
    if s3_path.startswith('s3://'):
        s3_path = s3_path[5:]
    bucket, key = s3_path.split('/', 1)

    # Check if we have a cached local copy
    if local_path and os.path.exists(local_path):
        print(f"Loading cached checkpoint from {local_path}")
        return torch.load(local_path, map_location=map_location)

    # Download from S3
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name

    try:
        s3 = get_s3_client()
        print(f"Downloading checkpoint from s3://{bucket}/{key}")
        s3.download_file(bucket, key, temp_path)

        # Load checkpoint
        checkpoint = torch.load(temp_path, map_location=map_location)

        # Save local copy if requested
        if local_path:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            torch.save(checkpoint, local_path)
            print(f"Cached checkpoint to {local_path}")

        return checkpoint

    finally:
        os.unlink(temp_path)


def checkpoint_exists(s3_path: str) -> bool:
    """Check if checkpoint exists in S3.

    Args:
        s3_path: S3 path to check

    Returns:
        True if checkpoint exists
    """
    if s3_path.startswith('s3://'):
        s3_path = s3_path[5:]
    bucket, key = s3_path.split('/', 1)

    s3 = get_s3_client()
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def list_checkpoints(s3_prefix: str) -> list:
    """List all checkpoints under S3 prefix.

    Args:
        s3_prefix: S3 prefix (s3://bucket/prefix or bucket/prefix)

    Returns:
        List of checkpoint S3 URIs
    """
    if s3_prefix.startswith('s3://'):
        s3_prefix = s3_prefix[5:]
    bucket, prefix = s3_prefix.split('/', 1)

    s3 = get_s3_client()
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    checkpoints = []
    for obj in response.get('Contents', []):
        if obj['Key'].endswith('.pt'):
            checkpoints.append(f"s3://{bucket}/{obj['Key']}")

    return sorted(checkpoints)


class S3Checkpoint:
    """Checkpoint manager with S3 backend.

    Handles saving/loading checkpoints with automatic S3 sync.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        local_dir: Optional[str] = None,
    ):
        """Initialize checkpoint manager.

        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix for checkpoints
            local_dir: Local directory for caching
        """
        self.bucket = bucket
        self.prefix = prefix.rstrip('/')
        self.local_dir = local_dir

        if local_dir:
            os.makedirs(local_dir, exist_ok=True)

    def save(
        self,
        checkpoint: Dict[str, Any],
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save checkpoint to S3.

        Args:
            checkpoint: Checkpoint dict (model state, optimizer state, etc.)
            name: Checkpoint name (e.g., 'epoch_10.pt', 'best.pt')
            metadata: Optional metadata to save alongside

        Returns:
            S3 URI of saved checkpoint
        """
        if not name.endswith('.pt'):
            name = f"{name}.pt"

        s3_key = f"{self.prefix}/{name}"
        local_path = None
        if self.local_dir:
            local_path = os.path.join(self.local_dir, name)

        # Add metadata to checkpoint
        if metadata:
            checkpoint['_metadata'] = metadata

        return upload_checkpoint(
            checkpoint,
            f"{self.bucket}/{s3_key}",
            local_path,
        )

    def load(
        self,
        name: str,
        map_location: str = 'cpu',
    ) -> Dict[str, Any]:
        """Load checkpoint from S3.

        Args:
            name: Checkpoint name
            map_location: Device to load tensors to

        Returns:
            Checkpoint dictionary
        """
        if not name.endswith('.pt'):
            name = f"{name}.pt"

        s3_path = f"{self.bucket}/{self.prefix}/{name}"
        local_path = None
        if self.local_dir:
            local_path = os.path.join(self.local_dir, name)

        return download_checkpoint(s3_path, local_path, map_location)

    def exists(self, name: str) -> bool:
        """Check if checkpoint exists.

        Args:
            name: Checkpoint name

        Returns:
            True if checkpoint exists in S3
        """
        if not name.endswith('.pt'):
            name = f"{name}.pt"
        s3_path = f"s3://{self.bucket}/{self.prefix}/{name}"
        return checkpoint_exists(s3_path)

    def list_all(self) -> list:
        """List all checkpoints.

        Returns:
            List of checkpoint names
        """
        s3_prefix = f"s3://{self.bucket}/{self.prefix}"
        uris = list_checkpoints(s3_prefix)
        return [uri.split('/')[-1] for uri in uris]

    def get_latest(self) -> Optional[str]:
        """Get the most recent checkpoint name.

        Returns:
            Name of most recent checkpoint, or None
        """
        checkpoints = self.list_all()
        if not checkpoints:
            return None

        # Sort by epoch number if available
        def extract_epoch(name):
            try:
                # Handle names like 'epoch_10.pt'
                return int(name.split('_')[1].split('.')[0])
            except (IndexError, ValueError):
                return -1

        checkpoints.sort(key=extract_epoch, reverse=True)
        return checkpoints[0]

    def get_best(self) -> Optional[str]:
        """Get the best checkpoint (if saved as 'best.pt').

        Returns:
            'best.pt' if exists, else None
        """
        if self.exists('best'):
            return 'best.pt'
        return None
