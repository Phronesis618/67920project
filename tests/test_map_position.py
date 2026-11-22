import pytest
from map_position import MapPosition, map_headers
from navigation import navigation_state, shaping_reward
from red_gym_env_v2_nuzlocke import RedGymEnv


def header(map_id, pointer=None):
    return (*map_headers()[map_id], map_id if pointer is None else pointer, 0, 0, 0, 1, 0, 0)


def test_warp_holds_source_until_header_and_lcd_are_ready():
    p = MapPosition()
    p.observe((4, 7, 50), header(50), True)
    p.observe((4, 7, 13), header(50), True)
    assert p.coords == (4, 7, 50) and not p.valid
    p.observe((4, 7, 13), header(13), False)
    assert p.coords == (4, 7, 50) and not p.valid
    p.observe((3, 43, 13), header(13), True)
    assert p.coords == (3, 43, 13) and p.valid
    assert navigation_state(13, 3, 43, got_parcel=True, got_pokedex=True)[1] == 226


def test_same_sized_maps_require_new_identity_or_completed_lcd_cycle():
    p = MapPosition()
    p.observe((10, 5, 12), header(12), True)
    assert map_headers()[12] == map_headers()[16]
    p.observe((10, 5, 16), header(12), True)
    assert p.coords[2] == 12 and not p.valid
    p.observe((10, 5, 16), header(16), True)
    assert p.coords[2] == 16 and p.valid
    p.observe((10, 5, 12), header(16), False)
    p.observe((10, 5, 12), header(16), True)
    assert p.coords[2] == 12 and p.valid


def test_outdoor_connections_and_unsigned_boundary_coordinates():
    p = MapPosition()
    p.observe((10, 0, 0), header(0), True)
    p.observe((10, 255, 0), header(0), True)
    assert p.coords == (10, 0, 0) and not p.valid
    p.observe((10, 35, 12), header(0), True)
    assert p.coords[2] == 0 and not p.valid
    p.observe((10, 35, 12), header(12), True)
    assert p.coords == (10, 35, 12) and p.valid


def test_environment_uses_one_position_for_reward_exploration_and_info(fake_pyboy, config):
    env = RedGymEnv(dict(config, navigation_reward=True, navigation_gamma=0.9995, max_steps=5))
    try:
        env.reset()
        env.pyboy.set_map(50, 4, 7)
        env.pyboy.memory[0xD74B] = 1 << 5
        env._observe_rules()
        *_, before = env.step(5)
        old_potential = env.previous_potential
        env.pyboy.memory[0xD35E] = 13
        *_, during = env.step(5)
        assert during["raw_position"] == (4, 7, 13)
        assert during["position"] == (4, 7, 50)
        assert not during["position_coherent"]
        assert during["map_coverage"] == before["map_coverage"]
        assert during["path_reward"] == pytest.approx(
            shaping_reward(old_potential, old_potential, 0.9995, False)
        )
        assert "x:4 y:7 m:13" not in env.seen_coords
        env.pyboy.set_map(13, 3, 43)
        *_, after = env.step(5)
        assert after["navigation_distance"] == 226
        assert abs(after["path_reward"]) < 0.02
    finally:
        env.close()


def test_loading_does_not_skip_faint_detection(fake_pyboy, config):
    env = RedGymEnv(config)
    try:
        env.reset()
        env.pyboy.memory[0xFF40] = 0
        env.pyboy.transitions = {2: lambda p: p.set_mon(hp=0)}
        _, _, terminated, _, info = env.step(0)
        assert terminated and info["end_reason"] == "wipe" and info["deaths"] == 1
    finally:
        env.close()


def test_ledge_intermediate_tile_is_not_a_new_position():
    p = MapPosition()
    p.observe((6, 60, 13), header(13), True)
    p.observe((6, 61, 13), header(13), True, jumping=True)
    assert p.coords == (6, 60, 13) and not p.valid
    p.observe((6, 62, 13), header(13), True, jumping=False)
    assert p.coords == (6, 62, 13) and p.valid
