"""Students: SenseLab Managed Models distilled from the harness's own decision traces.

``train.py`` turns a training matrix into models (scope from the local episode logs, create,
readiness check, train, poll, registry). ``grid.py`` evaluates them on held-out seeds under
the harness with the same scenarios, arms and metrics as the frontier grid.
"""
