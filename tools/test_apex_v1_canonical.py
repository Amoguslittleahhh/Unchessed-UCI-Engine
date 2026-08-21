#!/usr/bin/env python3

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent


class ApexV1ExperimentalAliasTests(unittest.TestCase):
    def test_apex_name_is_now_an_experimental_alias(self):
        registry = json.loads((ROOT / "config/architecture_registry.json").read_text())
        self.assertEqual(registry["canonical"]["id"], "unarchitectured-v1")
        apex = json.loads((ROOT / "config/unchessed_apex_v1.json").read_text())
        self.assertIn("experimental naming candidate", apex["status"])
        self.assertEqual(
            apex["canonical_successor"], "config/unarchitectured_v1.json"
        )

    def test_hydra_lineage_points_to_unarchitectured(self):
        for version in range(1, 6):
            config = json.loads(
                (ROOT / f"config/unchessed_hydra_v{version}.json").read_text()
            )
            self.assertIn("experimental predecessor", config["status"])
            self.assertEqual(
                config["canonical_successor"], "config/unarchitectured_v1.json"
            )


if __name__ == "__main__":
    unittest.main()
