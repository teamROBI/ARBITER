# Copyright 2026 ARBITER Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Sim scene construction and the motion backend.

Deliberately empty of gym registrations. ARBITER's collection, gate and evaluation paths all
drive `scene.build_scene` plus `isaac_backend.IsaacMotionBackend` directly; there is no
ManagerBasedRLEnv in the loop, so importing this package must not pull gymnasium or any env cfg.
"""
