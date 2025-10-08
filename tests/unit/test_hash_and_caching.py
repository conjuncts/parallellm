"""
Unit tests for hash computation and caching logic

Tests the core hashing and caching functionality including:
- compute_hash function for different document types
- Hash status tracking and dashboard logging
- Cache retrieval logic in backends
- Call identifier matching
"""

import pytest
from unittest.mock import Mock, patch
import hashlib
from io import BytesIO
from PIL import Image
from parallellm.core.hash import compute_hash
from parallellm.logging.dash_logger import DashboardLogger, HashStatus
from parallellm.core.backend import _call_matches
from parallellm.types import CallIdentifier


class TestComputeHash:
    """Test the compute_hash function"""

    def test_hash_text_only(self):
        """Test hashing with text documents only"""
        instructions = "Test instructions"
        documents = ["Document 1", "Document 2"]

        result = compute_hash(instructions, documents)

        # Should return a hex string
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex length

        # Should be deterministic
        result2 = compute_hash(instructions, documents)
        assert result == result2

    def test_hash_no_instructions(self):
        """Test hashing with None instructions"""
        documents = ["Only document"]

        result = compute_hash(None, documents)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_empty_documents(self):
        """Test hashing with empty document list"""
        instructions = "Just instructions"
        documents = []

        result = compute_hash(instructions, documents)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_with_images(self):
        """Test hashing with PIL Images"""
        instructions = "Process this image"

        # Create a simple test image
        img = Image.new("RGB", (10, 10), color="red")
        documents = ["Text doc", img]

        result = compute_hash(instructions, documents)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_deterministic_with_images(self):
        """Test that hash is deterministic with identical images"""
        instructions = "Process images"

        img1 = Image.new("RGB", (5, 5), color="blue")
        img2 = Image.new("RGB", (5, 5), color="blue")  # Identical image

        result1 = compute_hash(instructions, ["text", img1])
        result2 = compute_hash(instructions, ["text", img2])

        assert result1 == result2

    def test_hash_different_with_different_images(self):
        """Test that hash differs with different images"""
        instructions = "Process images"

        img1 = Image.new("RGB", (5, 5), color="red")
        img2 = Image.new("RGB", (5, 5), color="green")  # Different image

        result1 = compute_hash(instructions, ["text", img1])
        result2 = compute_hash(instructions, ["text", img2])

        assert result1 != result2

    def test_hash_different_orders(self):
        """Test that document order affects hash"""
        instructions = "Order test"

        result1 = compute_hash(instructions, ["doc1", "doc2"])
        result2 = compute_hash(instructions, ["doc2", "doc1"])

        assert result1 != result2

    def test_hash_unsupported_type(self):
        """Test error handling for unsupported document types"""
        instructions = "Test"
        documents = ["valid", 123]  # Invalid type

        with pytest.raises(ValueError, match="Unsupported document type"):
            compute_hash(instructions, documents)

    def test_hash_manual_verification(self):
        """Test hash computation with manual verification"""
        instructions = "manual test"
        documents = ["test doc"]

        # Manually compute expected hash
        hasher = hashlib.sha256()
        hasher.update(instructions.encode("utf-8"))
        hasher.update("test doc".encode("utf-8"))
        expected = hasher.hexdigest()

        result = compute_hash(instructions, documents)
        assert result == expected


class TestDashboardLogger:
    """Test DashboardLogger functionality"""

    def test_update_hash_new_entry(self):
        """Test adding a new hash entry"""
        logger = DashboardLogger(k=3, display=False)
        full_hash = "abcdef123456" + "0" * 52

        logger.update_hash(full_hash, HashStatus.SENT)

        assert len(logger._hashes) == 1
        assert "abcdef12" in logger._hashes
        assert logger._hashes["abcdef12"].status == HashStatus.SENT

    def test_update_hash_existing_entry(self):
        """Test updating an existing hash entry"""
        logger = DashboardLogger(k=3, display=False)
        full_hash = "abcdef123456" + "0" * 52

        # Add initial entry
        logger.update_hash(full_hash, HashStatus.SENT)

        # Update existing entry
        logger.update_hash(full_hash, HashStatus.RECEIVED)

        assert len(logger._hashes) == 1
        assert logger._hashes["abcdef12"].status == HashStatus.RECEIVED

    def test_hash_limit_enforcement(self):
        """Test that logger respects k limit"""
        logger = DashboardLogger(k=2, display=False)

        # Add 3 hashes (should only keep 2)
        hashes = [
            "aaaaaaaa" + "0" * 56,
            "bbbbbbbb" + "1" * 56,
            "cccccccc" + "2" * 56,
        ]

        for i, h in enumerate(hashes):
            logger.update_hash(h, HashStatus.SENT)

        assert len(logger._hashes) == 2
        # Should keep the most recent 2
        assert "bbbbbbbb" in logger._hashes
        assert "cccccccc" in logger._hashes
        assert "aaaaaaaa" not in logger._hashes

    def test_get_status(self):
        """Test getting status of a hash"""
        logger = DashboardLogger(k=3, display=False)
        full_hash = "12345678" + "9" * 56

        # Hash not present
        assert logger.get_status(full_hash) is None

        # Add hash and check status
        logger.update_hash(full_hash, HashStatus.CACHED)
        assert logger.get_status(full_hash) == HashStatus.CACHED

    def test_get_all_hashes(self):
        """Test getting all hashes and statuses"""
        logger = DashboardLogger(k=3, display=False)

        hashes = [
            ("aaaaaaaa" + "0" * 56, HashStatus.SENT),
            ("bbbbbbbb" + "1" * 56, HashStatus.RECEIVED),
        ]

        for hash_val, status in hashes:
            logger.update_hash(hash_val, status)

        all_hashes = logger.get_all_hashes()
        assert len(all_hashes) == 2
        assert all_hashes[hashes[0][0]] == HashStatus.SENT
        assert all_hashes[hashes[1][0]] == HashStatus.RECEIVED

    def test_clear_hashes(self):
        """Test clearing all hashes"""
        logger = DashboardLogger(k=3, display=False)

        # Add some hashes
        logger.update_hash("test_hash" + "0" * 56, HashStatus.SENT)
        assert len(logger._hashes) == 1

        # Clear
        logger.clear()
        assert len(logger._hashes) == 0

    @patch("sys.stdout.write")
    @patch("shutil.get_terminal_size")
    def test_console_update_disabled(self, mock_terminal_size, mock_stdout):
        """Test that console update is disabled when display=False"""
        mock_terminal_size.return_value.columns = 80

        logger = DashboardLogger(k=3, display=False)
        logger.update_hash("test_hash" + "0" * 56, HashStatus.SENT)

        # stdout.write should not be called when display is disabled
        mock_stdout.assert_not_called()

    @patch("sys.stdout.write")
    @patch("shutil.get_terminal_size")
    def test_console_update_enabled(self, mock_terminal_size, mock_stdout):
        """Test that console update works when display=True"""
        mock_terminal_size.return_value.columns = 80

        logger = DashboardLogger(k=3, display=True)
        logger.update_hash("test_hash" + "0" * 56, HashStatus.SENT)

        # stdout.write should be called when display is enabled
        mock_stdout.assert_called()


class TestCallMatching:
    """Test call identifier matching logic"""

    def test_call_matches_identical(self):
        """Test that identical calls match"""
        call1 = self._create_call_id("agent1", "chk1", "hash123", 1, 100)
        call2 = self._create_call_id("agent1", "chk1", "hash123", 1, 100)

        assert _call_matches(call1, call2) is True

    def test_call_matches_different_session(self):
        """Test that calls with different session_id still match"""
        call1 = self._create_call_id("agent1", "chk1", "hash123", 1, 100)
        call2 = self._create_call_id(
            "agent1", "chk1", "hash123", 1, 200
        )  # Different session

        # Should match because session_id is ignored for matching
        assert _call_matches(call1, call2) is True

    def test_call_matches_different_checkpoint(self):
        """Test that calls with different checkpoints don't match"""
        call1 = self._create_call_id("agent1", "chk1", "hash123", 1, 100)
        call2 = self._create_call_id(
            "agent1", "chk2", "hash123", 1, 100
        )  # Different checkpoint

        assert _call_matches(call1, call2) is False

    def test_call_matches_different_hash(self):
        """Test that calls with different hashes don't match"""
        call1 = self._create_call_id("agent1", "chk1", "hash123", 1, 100)
        call2 = self._create_call_id(
            "agent1", "chk1", "hash456", 1, 100
        )  # Different hash

        assert _call_matches(call1, call2) is False

    def test_call_matches_different_seq_id(self):
        """Test that calls with different seq_id don't match"""
        call1 = self._create_call_id("agent1", "chk1", "hash123", 1, 100)
        call2 = self._create_call_id(
            "agent1", "chk1", "hash123", 2, 100
        )  # Different seq_id

        assert _call_matches(call1, call2) is False

    def test_call_matches_none_checkpoints(self):
        """Test matching with None checkpoints"""
        call1 = self._create_call_id("agent1", None, "hash123", 1, 100)
        call2 = self._create_call_id("agent1", None, "hash123", 1, 200)

        assert _call_matches(call1, call2) is True

    def test_call_matches_one_none_checkpoint(self):
        """Test that None and non-None checkpoints don't match"""
        call1 = self._create_call_id("agent1", None, "hash123", 1, 100)
        call2 = self._create_call_id("agent1", "chk1", "hash123", 1, 100)

        assert _call_matches(call1, call2) is False

    def _create_call_id(
        self, agent_name, checkpoint, doc_hash, seq_id, session_id
    ) -> CallIdentifier:
        """Helper to create call identifiers"""
        return {
            "agent_name": agent_name,
            "checkpoint": checkpoint,
            "doc_hash": doc_hash,
            "seq_id": seq_id,
            "session_id": session_id,
            "provider_type": "openai",
        }


class TestCacheIntegration:
    """Test integration between hashing and caching systems"""

    def test_hash_to_cache_workflow(self):
        """Test the typical workflow from hash computation to cache lookup"""
        # 1. Compute hash
        instructions = "Test instructions"
        documents = ["Test document"]
        computed_hash = compute_hash(instructions, documents)

        # 2. Create call ID with hash
        call_id: CallIdentifier = {
            "agent_name": "test_agent",
            "checkpoint": "test_checkpoint",
            "doc_hash": computed_hash,
            "seq_id": 1,
            "session_id": 1,
            "provider_type": "openai",
        }

        # 3. Test call matching
        identical_call_id = call_id.copy()
        assert _call_matches(call_id, identical_call_id)

        # 4. Test hash status tracking
        logger = DashboardLogger(k=5, display=False)
        logger.update_hash(computed_hash, HashStatus.SENT)

        assert logger.get_status(computed_hash) == HashStatus.SENT

    def test_cache_miss_to_hit_workflow(self):
        """Test workflow from cache miss to cache hit"""
        logger = DashboardLogger(k=5, display=False)
        test_hash = "abcdef123456" + "0" * 52

        # 1. Initial request (cache miss)
        logger.update_hash(test_hash, HashStatus.SENT)
        assert logger.get_status(test_hash) == HashStatus.SENT

        # 2. Response received
        logger.update_hash(test_hash, HashStatus.RECEIVED)
        assert logger.get_status(test_hash) == HashStatus.RECEIVED

        # 3. Stored in cache
        logger.update_hash(test_hash, HashStatus.STORED)
        assert logger.get_status(test_hash) == HashStatus.STORED

        # 4. Next request (cache hit)
        logger.update_hash(test_hash, HashStatus.CACHED)
        assert logger.get_status(test_hash) == HashStatus.CACHED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
