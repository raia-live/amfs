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
    if name == "senselab-induce":
        from .senselab_arm import SenseLabInduceArm
        return SenseLabInduceArm()
    if name == "senselab-lean":
        from .senselab_arm import SenseLabLeanArm
        return SenseLabLeanArm()
    if name == "senselab-repair":
        from .senselab_arm import SenseLabRepairArm
        return SenseLabRepairArm()
    if name == "senselab-compose":
        from .senselab_arm import SenseLabComposeArm
        return SenseLabComposeArm()
    if name == "raw-traces":
        from .raw_traces_arm import RawTracesArm
        return RawTracesArm()
    if name == "skillfile":
        from .skillfile_arm import SkillFileArm
        return SkillFileArm()
    if name.startswith("senselab-rsi"):
        from . import senselab_rsi_arm as rsi
        cls = {
            "senselab-rsi": rsi.SenseLabRsiArm, "senselab-rsi-nocanary": rsi.SenseLabRsiNoCanaryArm,
            "senselab-rsi-noreplay": rsi.SenseLabRsiNoReplayArm, "senselab-rsi-review": rsi.SenseLabRsiReviewArm,
            "senselab-rsi-minimal": rsi.SenseLabRsiMinimalArm, "senselab-rsi-selfreport": rsi.SenseLabRsiSelfReportArm,
            "senselab-rsi-judges": rsi.SenseLabRsiJudgesArm,
        }.get(name)
        if cls is not None:
            return cls()
    if name.startswith("senselab-oracle-fix"):
        from .oracle_fix_arm import SenseLabOracleFixArm
        lag = name.rsplit("-", 1)[-1]
        return SenseLabOracleFixArm(lag=int(lag) if lag.isdigit() else 10)
    if name == "pgvector-diy+outcomes":
        from .pgvector_arm import PgVectorDiyOutcomesArm
        return PgVectorDiyOutcomesArm()
    raise ValueError(f"unknown arm {name!r}")


__all__ = ["ArmAccounting", "EpisodeSession", "MemoryArm", "MemoryHit", "Outcome", "make_arm", "render_hits"]
