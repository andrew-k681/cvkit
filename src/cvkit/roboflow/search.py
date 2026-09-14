"""Paginated image search, worked around.

Offset pagination over the search endpoint is not stable: the ordering shifts
between calls, so a single sweep both returns duplicates and misses rows. One
observed pass over 800 images came back with 735 rows holding 657 unique ids.

So we sweep repeatedly and keep a set, stopping once two consecutive sweeps
add nothing. It is not a guarantee -- it is the best this endpoint supports.

Every request goes through _call, which is what keeps the API key out of error
messages -- see its docstring before adding a call that bypasses it.
"""
from .. import config

API = "https://api.roboflow.com"


class ApiError(Exception):
    """A failed Roboflow call, with the API key scrubbed out of the message."""


def _call(requests, key, method, url, **kw):
    """Make one request, and let nothing out of `requests` escape raw.

    The key travels as a query parameter, because that is what this endpoint
    takes -- so requests puts the whole URL, key included, in the message of
    every error it raises. Uncaught, that message reaches a traceback, a CI
    log, and whatever gets pasted into a bug report.

    The raise sits *outside* the except block, and that placement is the point:
    raised inside it, the original exception stays attached as __context__ even
    with `from None` -- which only sets a flag the default traceback printer
    happens to honour. Anything that walks the chain itself (a logger, a crash
    reporter, pytest's assertion output) prints the key after all. Outside the
    handler there is no context to attach.

    ApiError derives from Exception rather than SystemExit deliberately, so the
    per-item `except Exception` in the tagging loops still catches it and the
    batch keeps going instead of aborting on one bad row.
    """
    try:
        r = getattr(requests, method)(url, params={"api_key": key}, **kw)
        r.raise_for_status()
        return r
    except Exception as e:
        # Reason first, URL last: the per-item handlers print str(e)[:90], and
        # a Roboflow image URL alone eats that whole budget -- lead with the
        # address and every failure line reads "...failed: H".
        msg = (f"{type(e).__name__}: {config.scrub(str(e), key)} "
               f"[{method.upper()} {config.scrub(url, key)}]")
    raise ApiError(msg)


def page(requests, key, workspace, project, body, offset, limit=100, timeout=60):
    r = _call(requests, key, "post", f"{API}/{workspace}/{project}/search",
              json={**body, "offset": offset, "limit": limit}, timeout=timeout)
    return r.json().get("results", [])


def all_images(requests, key, workspace, project, body, passes=6, quiet=False):
    """Collect unique images, repeating until the set stops growing."""
    seen, stable = {}, 0
    for p in range(passes):
        before = len(seen)
        offset = 0
        while True:
            batch = page(requests, key, workspace, project, body, offset)
            if not batch:
                break
            for im in batch:
                seen.setdefault(im["id"], im)
            offset += len(batch)
        if not quiet:
            print(f"  sweep {p + 1}: {len(seen)} unique (+{len(seen) - before})")
        stable = stable + 1 if len(seen) == before else 0
        if stable >= 2:
            break
    return list(seen.values())


def set_tags(requests, key, workspace, project, image_id, tags, operation, timeout=60):
    """add or remove tags on one image. The endpoint needs the operation,
    not just a tag list."""
    return _call(requests, key, "post",
                 f"{API}/{workspace}/{project}/images/{image_id}/tags",
                 json={"operation": operation, "tags": list(tags)}, timeout=timeout)


def annotation(requests, key, workspace, project, image_id, timeout=60):
    """The stored annotation for one image, including per-object geometry."""
    r = _call(requests, key, "get",
              f"{API}/{workspace}/{project}/images/{image_id}", timeout=timeout)
    return r.json()["image"]["annotation"]
