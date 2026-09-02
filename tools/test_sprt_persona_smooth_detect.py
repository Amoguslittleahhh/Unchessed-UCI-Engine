#!/usr/bin/env python3
"""Sanity-check the persona SPRT scripts without running cutechess."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPRT = ROOT / "scripts/sprt-history/sprt_persona_smooth_detect.sh"
SMOKE = ROOT / "scripts/sprt-history/smoke_persona_smooth_detect.sh"


def test_sprt_script_matches_punish_latch_shape():
    text = SPRT.read_text()
    assert "option.Adaptive=true" in text
    assert "option.PersonaSmooth=true" in text
    assert "option.EngineDetectV2=true" in text
    assert "option.PersonaSmooth=false" in text
    assert "option.EngineDetectV2=false" in text
    assert "tc=5+0.05" in text
    assert "elo0=0 elo1=5" in text
    assert "cutechess-cli" in text
    # same binary both sides (UCI toggle, not two trees)
    assert text.count('cmd="$ENGINE"') == 2


def test_smoke_exists_and_is_short():
    text = SMOKE.read_text()
    assert "rounds 1" in text
    assert "PersonaSmooth=true" in text
    assert "Adaptive=true" in text


if __name__ == "__main__":
    test_sprt_script_matches_punish_latch_shape()
    test_smoke_exists_and_is_short()
    print("ok")
