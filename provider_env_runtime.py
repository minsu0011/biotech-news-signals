"""Process-local provider environment import without secret serialization.

Only the explicitly authorized read-only market-data variables are read
from the Windows *User* environment, and only when the current process is
missing them.  Public helpers return PRESENT/MISSING markers, never values.
"""
from __future__ import annotations

import os
from collections.abc import Iterable


PROVIDER_ENV_NAMES = (
    "MASSIVE_API_KEY",
    "POLYGON_API_KEY",
    "KIS_APP_KEY",
    "KIS_APP_SECRET",
    "ALPACA_API_KEY",
    "ALPACA_API_SECRET",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "TIINGO_API_TOKEN",
)

ALPACA_ENV_NAMES = (
    "ALPACA_API_KEY",
    "ALPACA_API_SECRET",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
)

TIINGO_ENV_NAMES = ("TIINGO_API_TOKEN",)


def _windows_user_value(name: str) -> str | None:
    if os.name != "nt":
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, value_type = winreg.QueryValueEx(key, name)
    except FileNotFoundError:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    if value_type == winreg.REG_EXPAND_SZ:
        value = os.path.expandvars(value)
    return value


def import_user_environment_if_missing(
    names: Iterable[str] = PROVIDER_ENV_NAMES,
) -> dict[str, str]:
    for name in names:
        if name not in PROVIDER_ENV_NAMES:
            raise ValueError(f"provider environment name is not allowlisted: {name}")
        if os.environ.get(name):
            continue
        value = _windows_user_value(name)
        if value:
            os.environ[name] = value
    return presence(names)


def refresh_user_environment(
    names: Iterable[str] = PROVIDER_ENV_NAMES,
) -> dict[str, str]:
    """Replace process values with current Windows User-scope authority.

    A missing User-scope value removes a possibly stale inherited process
    value.  This is intended for fresh credential retry children only.
    """
    for name in names:
        if name not in PROVIDER_ENV_NAMES:
            raise ValueError(f"provider environment name is not allowlisted: {name}")
        value = _windows_user_value(name)
        if value:
            os.environ[name] = value
        else:
            os.environ.pop(name, None)
    return presence(names)


def presence(names: Iterable[str] = PROVIDER_ENV_NAMES) -> dict[str, str]:
    return {
        name: "PRESENT" if bool(os.environ.get(name)) else "MISSING"
        for name in names
    }


def all_present(names: Iterable[str] = PROVIDER_ENV_NAMES) -> bool:
    return all(value == "PRESENT" for value in presence(names).values())
