"""Helpers for tests that need the F1TENTH simulator (skipped when it is not installed)."""

import importlib.util
import unittest

try:
    import gym  # noqa: F401
    import f110_gym  # noqa: F401
    HAVE_SIM = True
except Exception:  # pragma: no cover - depends on environment
    HAVE_SIM = False

HAVE_SB3 = importlib.util.find_spec("stable_baselines3") is not None

requires_sim = unittest.skipUnless(HAVE_SIM, "f110_gym not installed")
requires_sb3 = unittest.skipUnless(HAVE_SIM and HAVE_SB3, "f110_gym / stable_baselines3 not installed")
