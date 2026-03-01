# profiles.py
# Loads profiles from profiles.json.
# Use backtest.py to automatically update score_threshold and trend_penalty.

import json
import os

PROFILES_PATH = os.path.join(os.path.dirname(__file__), "profiles.json")


def load_profiles():
    """Load profiles from profiles.json."""
    with open(PROFILES_PATH, "r") as f:
        return json.load(f)


def save_profiles(profiles):
    """Save profiles back to profiles.json."""
    with open(PROFILES_PATH, "w") as f:
        json.dump(profiles, f, indent=4)


def update_profile(profile_key, params):
    """
    Update specific parameters for a profile in profiles.json.
    Called automatically by backtest.py after optimization.
    """
    profiles = load_profiles()

    if profile_key not in profiles:
        print(f"  ❌ Profile '{profile_key}' not found in profiles.json")
        return

    for key, value in params.items():
        profiles[profile_key][key] = value

    save_profiles(profiles)
    print(f"\n  ✅ profiles.json updated for '{profile_key}':")
    for key, value in params.items():
        print(f"     {key} → {value}")
    print()


# ─── Load on import ────────────────────────────────────────
PROFILES     = load_profiles()
PROFILE_KEYS = list(PROFILES.keys())


def ask_profile():
    """Prompt the user to choose a trading profile and return it."""
    # Reload each time so backtest updates are reflected immediately
    profiles     = load_profiles()
    profile_keys = list(profiles.keys())

    print("\n" + "═" * 50)
    print("  📋 SELECT YOUR TRADING STYLE")
    print("═" * 50)
    for i, (key, profile) in enumerate(profiles.items(), 1):
        print(f"  [{i}] {profile['label']}")
    print("═" * 50)

    while True:
        choice = input("  Enter number (default: 3 — swing): ").strip()
        if not choice:
            key = "swing"
            break
        if choice.isdigit() and 1 <= int(choice) <= len(profiles):
            key = profile_keys[int(choice) - 1]
            break
        print(f"  ❌ Invalid choice. Enter a number between 1 and {len(profiles)}.")

    profile = profiles[key]
    print(f"\n  ✅ Profile selected: {profile['label']}")
    print(f"     Default period:    {profile['default_period']}")
    print(f"     Minimum period:    {profile['min_period']}")
    print(f"     Forward window:    {profile['forward_window']} days")
    print(f"     Score threshold:   ±{profile['score_threshold']}")
    print(f"     Confirmation:      {profile['confirmation_window']} day(s)\n")
    return key, profile