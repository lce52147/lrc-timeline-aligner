#!/usr/bin/env python
"""Manual test for accept decision."""

import json
import sys
import time
import threading
from pathlib import Path
from urllib.request import Request, urlopen

# Add lyricsee to path
sys.path.insert(0, str(Path(__file__).parent / "lyricsee"))

from review_server import ReviewServerConfig, start_review_server

def test_accept_via_api():
    """Test accepting a candidate via the API."""
    test_output = Path("F:/_RuntimePrompt/LYRICSEE/AA_review_20261026/out_accepttest")

    if not test_output.exists():
        print("ERROR: Test output directory not found")
        return False

    # Find a song with candidates
    candidates_dir = test_output / "candidates"
    if not candidates_dir.exists():
        print("ERROR: No candidates directory")
        return False

    songs = list(candidates_dir.iterdir())
    if not songs:
        print("ERROR: No songs with candidates")
        return False

    test_stem = songs[0].name
    print(f"Testing with stem: {test_stem}")

    # Start the review server
    config = ReviewServerConfig(output_root=test_output)
    url, server = start_review_server(config)
    print(f"Server started at {url}")

    try:
        # Wait for server to be ready
        time.sleep(0.5)

        # Make accept request
        decision_data = {
            "stem": test_stem,
            "decision": "accept",
            "candidate_idx": 0
        }

        request = Request(
            f"{url}/api/decide",
            data=json.dumps(decision_data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        response = urlopen(request)
        result = json.loads(response.read().decode("utf-8"))
        print(f"Accept response: {result}")

        # Check that lyrics file was created
        lyrics_file = test_output / "txt" / f"{test_stem}.lyrics.txt"
        if not lyrics_file.exists():
            print(f"ERROR: Lyrics file not created at {lyrics_file}")
            return False

        print(f"✓ Lyrics file created: {lyrics_file}")

        # Check that provenance was created
        provenance_file = test_output / "report" / f"{test_stem}.sources.json"
        if not provenance_file.exists():
            print(f"ERROR: Provenance file not created at {provenance_file}")
            return False

        provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
        if not provenance.get("human_approved"):
            print("ERROR: Provenance does not have human_approved=True")
            return False

        print(f"✓ Provenance file created with human_approved=True")

        # Check that decision was recorded
        decisions_file = test_output / "report" / "review_decisions.jsonl"
        if not decisions_file.exists():
            print("ERROR: Decision not recorded")
            return False

        lines = decisions_file.read_text(encoding="utf-8").strip().split("\n")
        last_decision = json.loads(lines[-1])
        if last_decision.get("stem") != test_stem or last_decision.get("decision") != "accept":
            print("ERROR: Decision not properly recorded")
            return False

        print(f"✓ Decision recorded in review_decisions.jsonl")

        print("\n✓ All checks passed!")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Server shuts down when we lose the reference
        server.shutdown()

if __name__ == "__main__":
    success = test_accept_via_api()
    sys.exit(0 if success else 1)
