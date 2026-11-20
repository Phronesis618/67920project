from dataclasses import replace

import pytest
from nuzlocke_rules import IdentityCollisionError, NuzlockeTracker, PartyMon

MON = PartyMon((123, 456), 1, 20, 20, 14)


def observe(t, party=(MON,), region=12, battle=0, owned=1):
    t.observe(party, map_id=region, battle_type=battle, owned_count=owned)


def test_no_starter_is_not_a_wipe():
    t = NuzlockeTracker()
    observe(t, (), owned=0)
    assert not t.terminated


@pytest.mark.parametrize("catch_on_exit", [False, True])
def test_first_catch_and_later_illegal_catch(catch_on_exit):
    t = NuzlockeTracker()
    observe(t)
    observe(t, battle=1)
    observe(t, battle=0 if catch_on_exit else 1, owned=2)
    assert not t.terminated
    observe(t, owned=2)
    observe(t, battle=1, owned=2)
    observe(t, battle=1, owned=3)
    assert t.violations == {"illegal_catch"}


def test_flee_uses_first_encounter():
    t = NuzlockeTracker()
    observe(t)
    observe(t, battle=1)
    observe(t)
    observe(t, battle=1)
    observe(t, battle=1, owned=2)
    assert "illegal_catch" in t.violations


def test_box_switch_and_gift_do_not_count_as_wild_catches():
    t = NuzlockeTracker(initial_used_regions=[12])
    observe(t, owned=1)
    observe(t, owned=15)
    observe(t, owned=1)
    observe(t, owned=2)
    assert not t.terminated


def test_faint_persists_through_reordering_evolution_and_healing():
    other = replace(MON, identity=(123, 789))
    t = NuzlockeTracker()
    observe(t, (MON, other), owned=2)
    observe(t, (replace(MON, hp=0), other), owned=2)
    assert t.dead == {MON.identity}
    assert not t.wiped
    observe(t, (other, replace(MON, species=2, hp=0)), owned=2)
    assert not t.terminated
    observe(t, (other, replace(MON, species=2)), owned=2)
    assert t.violations == {"revived_dead_mon"}


def test_collision_is_an_instrumentation_error_not_a_fake_wipe():
    t = NuzlockeTracker()
    with pytest.raises(IdentityCollisionError):
        observe(t, (MON, replace(MON, hp=0)))
    assert not t.terminated
    assert not t.dead


def test_wipe_counts_once_and_latches():
    t = NuzlockeTracker()
    observe(t)
    observe(t, (replace(MON, hp=0),))
    assert t.wiped and t.terminated
    observe(t)
    assert t.wiped and len(t.dead) == 1


def test_cap_at_entry_but_not_during_battle():
    t = NuzlockeTracker()
    observe(t, battle=2)
    observe(t, (replace(MON, level=15),), battle=2)
    assert not t.terminated
    observe(t, (replace(MON, level=15),))
    observe(t, (replace(MON, level=15),), battle=1)
    assert t.violations == {"level_cap"}


def test_region_aliases_share_one_encounter():
    t = NuzlockeTracker(region_map={12: 12, 13: 12})
    observe(t)
    observe(t, battle=1)
    observe(t)
    observe(t, region=13, battle=1)
    observe(t, region=13, battle=1, owned=2)
    assert t.used_regions == {12}
    assert "illegal_catch" in t.violations
