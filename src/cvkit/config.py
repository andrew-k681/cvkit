"""Credentials and the Roboflow target, resolved from flags, env, then .env.

The .env file is read a variable at a time and never sourced -- sourcing
executes it, which would export every other secret it holds and run anything
someone managed to append to it.
"""
import os
import re
from pathlib import Path

API_KEY_VAR = "ROBOFLOW_API_KEY"
WORKSPACE_VAR = "ROBOFLOW_WORKSPACE"
PROJECT_VAR = "ROBOFLOW_PROJECT"


def read_env_var(env_path, name):
    """Pull one variable out of a .env-style file. Returns None if absent."""
    path = Path(env_path)
    if not path.exists():
        return None
    pattern = re.compile(rf"\s*(?:export\s+)?{re.escape(name)}\s*=\s*(.*?)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    return None


def _lookup(name, env_path, override=None):
    if override:
        return override
    if os.environ.get(name):
        return os.environ[name]
    if env_path:
        return read_env_var(env_path, name)
    return None


def api_key(args=None, env_path=None, override=None):
    """The Roboflow API key. Flag, then environment, then .env file."""
    env_path = env_path or getattr(args, "env", None)
    key = _lookup(API_KEY_VAR, env_path, override)
    if not key:
        raise SystemExit(
            f"{API_KEY_VAR} not set. Export it, or put it in {env_path or '.env'} "
            f"(see .env.example). Never pass a key on the command line -- it "
            f"lands in your shell history."
        )
    return key


def target(args):
    """(workspace, project) from --workspace/--project, env, or .env."""
    env_path = getattr(args, "env", None)
    ws = _lookup(WORKSPACE_VAR, env_path, getattr(args, "workspace", None))
    pr = _lookup(PROJECT_VAR, env_path, getattr(args, "project", None))
    missing = [n for n, v in ((WORKSPACE_VAR, ws), (PROJECT_VAR, pr)) if not v]
    if missing:
        raise SystemExit(
            f"no Roboflow target: {' and '.join(missing)} unset. Pass "
            f"--workspace/--project, or set them in your .env. Both are the URL "
            f"slugs from app.roboflow.com/<workspace>/<project>."
        )
    return ws, pr


def scrub(text, key):
    """Blank a secret out of text on its way to a print or an exception.

    The key rides in the query string -- ours and the SDK's alike -- so it
    turns up in the message of any error requests raises.
    """
    return text.replace(key, "<redacted>") if key else text


def require(module, extra, package=None):
    """Import an optional dependency, or explain how to install it.

    The deferred imports live inside functions, so the CLI's top-level
    ImportError handler never sees them -- without this the user gets a bare
    ModuleNotFoundError traceback instead of the install command.
    """
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError:
        raise SystemExit(
            f"this command needs {package or module!r}, which is not "
            f"installed.\n    pip install 'cvkit[{extra}]'")


def add_roboflow_args(ap, default_env=".env"):
    """Flags shared by every command that talks to Roboflow."""
    ap.add_argument("--workspace", default=None,
                    help=f"Roboflow workspace slug (default: ${WORKSPACE_VAR})")
    ap.add_argument("--project", default=None,
                    help=f"Roboflow project slug (default: ${PROJECT_VAR})")
    ap.add_argument("--env", default=default_env,
                    help="dotenv file to read credentials from (default: .env)")
    return ap


def rf_project(args):
    """An authenticated roboflow Project handle for the resolved target."""
    key = api_key(args)
    ws, pr = target(args)
    os.environ[API_KEY_VAR] = key           # the SDK reads this in places
    Roboflow = require("roboflow", "roboflow").Roboflow
    # The SDK puts the key in its URLs, so an uncaught failure here is a
    # traceback carrying the secret.
    try:
        return Roboflow(api_key=key).workspace(ws).project(pr)
    except Exception as e:
        msg = scrub(f"{type(e).__name__}: {e}", key)
    raise SystemExit(f"could not open {ws}/{pr}: {msg}")
