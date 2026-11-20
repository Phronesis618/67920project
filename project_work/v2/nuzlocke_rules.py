"""ROM-independent rule monitor. See README.md for the deliberately limited ruleset.

The emulator reports snapshots; this module never edits RAM or supplies policy input.
OT ID + DVs survive party reordering and evolution. They are not guaranteed unique:
ambiguous rosters raise an instrumentation error instead of fabricating a death.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PartyMon:
    identity: tuple[int, int]
    species: int
    hp: int
    max_hp: int
    level: int
    status: int = 0


class IdentityCollisionError(RuntimeError):
    """This roster cannot safely be tracked using its RAM identity fields."""


class NuzlockeTracker:
    def __init__(self, level_cap=14, initial_used_regions=(), region_map=None):
        if level_cap is not None and not 1 <= level_cap <= 100:
            raise ValueError("level_cap must be 1..100 or None")
        self.level_cap = level_cap
        self.used_regions = set(initial_used_regions)
        if any(not isinstance(r, int) or not 0 <= r < 256 for r in self.used_regions):
            raise ValueError("Initial region IDs must be integers in 0..255")
        self.region_map = region_map or {}
        self.dead = set()
        self.violations = set()
        self.wiped = False
        self.previous_battle = 0
        self.previous_owned = None
        self.encounter_region = None
        self.encounter_catchable = False

    @property
    def terminated(self):
        return self.wiped or bool(self.violations)

    def observe(self, party, *, map_id, battle_type, owned_count):
        identities = [m.identity for m in party]
        if len(identities) != len(set(identities)):
            raise IdentityCollisionError(
                "Two party members share OT ID and DVs. Identity is ambiguous; "
                "stop this run and stage a distinguishable team. This is not a rule violation."
            )
        if self.terminated:
            return
        region = self.region_map.get(map_id, map_id)
        if not 0 <= region < 256:
            raise ValueError("Region IDs must be in 0..255")

        # Count individual faintings exactly once, including outside-battle poison damage.
        self.dead.update(m.identity for m in party if m.max_hp > 0 and m.hp == 0)
        if any(m.identity in self.dead and m.hp > 0 for m in party):
            self.violations.add("revived_dead_mon")
        self.wiped = bool(party) and all(m.identity in self.dead for m in party)

        # Explicit convention: the cap applies to the party at battle entry.
        # Level-ups within the fight do not retroactively invalidate that fight.
        if (
            battle_type in (1, 2)
            and self.previous_battle not in (1, 2)
            and self.level_cap is not None
            and any(m.level > self.level_cap for m in party)
        ):
            self.violations.add("level_cap")

        if battle_type == 1 and self.previous_battle != 1:
            self.encounter_region = region
            self.encounter_catchable = region not in self.used_regions
            self.used_regions.add(region)

        # Only a wild battle can yield a catch. PC box changes, withdrawals and gifts
        # outside battle must not be mistaken for catches. Include the exit snapshot:
        # some action intervals observe the party increment after the battle flag clears.
        if (
            self.previous_owned is not None
            and owned_count > self.previous_owned
            and (battle_type == 1 or self.previous_battle == 1)
        ):
            if not self.encounter_catchable:
                self.violations.add("illegal_catch")
            self.encounter_catchable = False

        self.previous_owned = owned_count
        self.previous_battle = battle_type
