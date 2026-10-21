import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import find_warnings

def test_flags_upfront_fee():
    assert "Upfront payment or fee" in find_warnings("Pay a registration fee before joining.")

def test_no_known_warning_phrase():
    assert find_warnings("Apply through the official careers portal and attend an interview.") == []
