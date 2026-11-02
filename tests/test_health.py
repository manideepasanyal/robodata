import pytest

from robodata.config import HealthThresholds
from robodata.health.checks import check_episode
from robodata.health.faults import default_suite, inject
from robodata.io.raw_episode import RawEpisode


def test_raw_roundtrip(robot, clean_episode):
    ep = RawEpisode(clean_episode)
    assert set(ep.streams) == set(robot.streams)
    assert ep.meta.sensor_config_hash == robot.sensor_config_hash()
    assert len(ep.times("ft")) == 800 and ep.data("ft").shape == (800, 6)
    assert ep.root["depth"]["data"].shape == (120, 48, 64)
    assert ep.video_path("cam").exists()
    assert not clean_episode.with_name(clean_episode.name + ".partial").exists()


def test_clean_episode_passes(robot, clean_episode):
    rep = check_episode(RawEpisode(clean_episode), robot, HealthThresholds(), resync=True)
    assert rep["status"] != "fail", rep["failed"]


@pytest.mark.parametrize("tag,faults,expected", default_suite("wrist_cam", "depth", "joints", "ft"),
                         ids=[s[0] for s in default_suite("wrist_cam", "depth", "joints", "ft")])
def test_fault_is_detected(robot, clean_episode, tag, faults, expected):
    bad = inject(clean_episode, faults, tag, seed=3)
    rep = check_episode(RawEpisode(bad), robot, HealthThresholds(), resync=True)
    assert rep["status"] == "fail"
    assert any(f.split(":")[0] == expected for f in rep["failed"]), rep["failed"]
    assert RawEpisode(bad).meta.injected_faults == faults
