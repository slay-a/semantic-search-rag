"""Supabase Auth (email/password) — thin wrapper used by the Streamlit app.

Enabled only when ``SUPABASE_URL`` and ``SUPABASE_ANON_KEY`` are set (see
:pyattr:`Settings.auth_enabled`). When enabled, the app requires sign-in and uses
the authenticated user's id as the ``tenant_id``, so each account gets a private,
isolated knowledge base.

We use email/password rather than OAuth: it needs no redirect URLs, which keeps
it simple to run inside Streamlit. All calls return plain dataclasses so the app
never touches the SDK's types directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, settings


@dataclass
class AuthUser:
    id: str
    email: str


class AuthError(Exception):
    """Raised for a failed sign-in / sign-up, with a user-friendly message."""


def _client(cfg: Settings):
    try:
        from supabase import create_client
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "supabase is required for auth. Install it with `pip install supabase`."
        ) from exc
    if not cfg.auth_enabled:
        raise RuntimeError("SUPABASE_URL / SUPABASE_ANON_KEY are not set.")
    return create_client(cfg.supabase_url, cfg.supabase_anon_key)


def _to_user(session_user) -> AuthUser:
    return AuthUser(id=session_user.id, email=session_user.email or "")


def sign_in(email: str, password: str, *, cfg: Settings = settings) -> AuthUser:
    """Sign in an existing user; raises :class:`AuthError` on bad credentials."""
    try:
        res = _client(cfg).auth.sign_in_with_password(
            {"email": email, "password": password}
        )
    except Exception as exc:
        raise AuthError("Invalid email or password.") from exc
    if not res.user:
        raise AuthError("Invalid email or password.")
    return _to_user(res.user)


def sign_up(email: str, password: str, *, cfg: Settings = settings) -> AuthUser:
    """Register a new user. Depending on project settings, email confirmation
    may be required before the account can sign in."""
    try:
        res = _client(cfg).auth.sign_up({"email": email, "password": password})
    except Exception as exc:
        raise AuthError(str(exc) or "Sign-up failed.") from exc
    if not res.user:
        raise AuthError("Sign-up failed.")
    return _to_user(res.user)
