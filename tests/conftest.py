from types import SimpleNamespace

import numpy as np
import pytest
import torch


class FakePyBoy:
    """Deterministic RAM test double, deliberately not a game simulation."""

    def __init__(self, *args, **kwargs):
        self.memory = bytearray(65536)
        self.screen = SimpleNamespace(ndarray=np.zeros((144, 160, 4), dtype=np.uint8))
        self.frames = 0
        self.transitions = {}
        self.stopped = False

    def load_state(self, f):
        self.memory[:] = bytes(65536)
        self.memory[0xD163] = 1
        self.set_mon()
        self.memory[0xD35E] = 0
        self.frames = 0

    def set_mon(self, slot=0, hp=20, level=14, identity=(123, 456)):
        b = 0xD16B + slot * 0x2C
        self.memory[b] = 1
        for offset, value in ((1, hp), (0x22, 20), (0x0C, identity[0]), (0x1B, identity[1])):
            self.memory[b + offset : b + offset + 2] = value.to_bytes(2, "big")
        self.memory[b + 0x21] = level

    def tick(self, count=1, render=True):
        for _ in range(count):
            self.frames += 1
            if self.frames in self.transitions:
                self.transitions[self.frames](self)

    def send_input(self, action):
        pass

    def set_emulation_speed(self, speed):
        pass

    def stop(self, save=False):
        self.stopped = True


@pytest.fixture
def fake_pyboy(monkeypatch):
    import red_gym_env_v2

    torch.set_num_threads(1)
    monkeypatch.setattr(red_gym_env_v2, "PyBoy", FakePyBoy)
    return FakePyBoy


@pytest.fixture
def config(tmp_path):
    state = tmp_path / "test.state"
    state.write_bytes(b"test double only")
    rom = tmp_path / "test.gb"
    rom.write_bytes(b"test double only")
    return {
        "session_path": tmp_path / "env",
        "init_state": str(state),
        "gb_path": str(rom),
        "headless": True,
        "save_final_state": False,
        "print_rewards": False,
        "save_video": False,
        "fast_video": True,
        "action_freq": 24,
        "max_steps": 4,
        "observation_mode": "partial",
        "reward_scale": 0.5,
        "explore_weight": 0.25,
    }
