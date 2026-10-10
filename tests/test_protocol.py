from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "apps"))

from PurpleTeam_EMG_RAW_Recorder import RawPacketParser, encode_packet  # noqa: E402


class RawPacketParserTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        parser = RawPacketParser()
        frame = encode_packet(42, 100, 200, 300)
        self.assertEqual(parser.feed(frame), [(42, 100, 200, 300, frame)])

    def test_recovers_after_noise_and_split_packet(self) -> None:
        parser = RawPacketParser()
        frame = encode_packet(7, 10, 20, 30)
        self.assertEqual(parser.feed(b"noise" + frame[:5]), [])
        parsed = parser.feed(frame[5:])
        self.assertEqual(parsed[0][:4], (7, 10, 20, 30))
        self.assertGreaterEqual(parser.discarded_bytes, 5)

    def test_rejects_corrupted_crc(self) -> None:
        parser = RawPacketParser()
        frame = bytearray(encode_packet(1, 2, 3, 4))
        frame[-1] ^= 0xFF
        self.assertEqual(parser.feed(bytes(frame)), [])
        self.assertEqual(parser.crc_errors, 1)


if __name__ == "__main__":
    unittest.main()

