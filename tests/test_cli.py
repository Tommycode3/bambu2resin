"""CLI argument parsing: English flags plus the legacy Italian aliases."""
from travaso.cli import build_parser


def test_priority_accepts_english_and_legacy_names():
    parser = build_parser()
    assert parser.parse_args(["orient", "x.stl", "--priority", "finish"]).priority == "finish"
    assert parser.parse_args(["orient", "x.stl", "--priorita", "velocita"]).priority == "speed"
    assert parser.parse_args(["build", "x.stl", "--out", "o.pm3"]).priority == "reliability"


def test_calibration_range_flags():
    parser = build_parser()
    args = parser.parse_args(["calibrate", "exposure", "--out", "t.pm3",
                              "--from", "2", "--to", "5", "--steps", "4"])
    assert (args.kind, args.t_from, args.t_to, args.steps) == ("exposure", 2.0, 5.0, 4)
    legacy = parser.parse_args(["calibrate", "exposure", "--out", "t.pm3",
                                "--da", "2", "--a", "5", "--passi", "4"])
    assert (legacy.t_from, legacy.t_to, legacy.steps) == (2.0, 5.0, 4)
