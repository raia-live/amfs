from __future__ import annotations

from .base import ArmAccounting, EpisodeSession, MemoryArm, MemoryHit, Outcome, render_hits


def make_arm(name: str) -> MemoryArm:
    if name == "none":
        from .none_arm import NoMemoryArm
        return NoMemoryArm()
    if name == "pgvector":
        from .pgvector_arm import PgVectorArm
        return PgVectorArm()
    if name == "pgvector-diy":
        from .pgvector_arm import PgVectorDiyArm
        return PgVectorDiyArm()
    if name == "mem0":
        from .mem0_arm import Mem0Arm
        return Mem0Arm()
    if name == "zep":
        from .zep_arm import ZepArm
        return ZepArm()
    if name == "senselab":
        from .senselab_arm import SenseLabArm
        return SenseLabArm()
    if name == "senselab-nofeedback":
        from .senselab_arm import SenseLabNoFeedbackArm
        return SenseLabNoFeedbackArm()
    if name == "senselab-episode":
        from .senselab_arm import SenseLabEpisodeArm
        return SenseLabEpisodeArm()
    if name == "senselab-attempts":
        from .senselab_arm import SenseLabAttemptsArm
        return SenseLabAttemptsArm()
    if name == "senselab-nopriors":
        from .senselab_arm import SenseLabNoPriorsArm
        return SenseLabNoPriorsArm()
    if name == "senselab-lean":
        from .senselab_arm import SenseLabLeanArm
        return SenseLabLeanArm()
    if name == "pgvector-diy+outcomes":
        from .pgvector_arm import PgVectorDiyOutcomesArm
        return PgVectorDiyOutcomesArm()
    raise ValueError(f"unknown arm {name!r}")


__all__ = ["ArmAccounting", "EpisodeSession", "MemoryArm", "MemoryHit", "Outcome", "make_arm", "render_hits"]
