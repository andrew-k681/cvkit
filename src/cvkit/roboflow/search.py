"""Paginated image search, worked around.

Offset pagination over the search endpoint is not stable: the ordering shifts
between calls, so a single sweep both returns duplicates and misses rows. One
observed pass over 800 images came back with 735 rows holding 657 unique ids.

So we sweep repeatedly and keep a set, stopping once two consecutive sweeps
add nothing. It is not a guarantee -- it is the best this endpoint supports.

Every request goes through _call, which keeps the API key out of error
messages. Do not add one that bypasses it.
"""
from .. import config

API = "https://api.roboflow.com"


class ApiError(Exception):
    """A failed Roboflow call, with the API key scrubbed out of the message."""


def _call(requests, key, method, url, **kw):
    """Make one request, letting nothing out of `requests` escape raw.

    The key rides in the query string, so requests names it in every error it
    raises. Two things are deliberate: the raise sits *outside* the except
    block, because inside it the original stays on __context__ even with
    `from None` and anything walking the chain prints the key anyway; and
    ApiError is an Exception, not SystemExit, so the per-item handlers in the
    tagging loops still catch it and the batch keeps going.
    """
    try:
        r = getattr(requests, method)(url, params={"api_key": key}, **kw)
        r.raise_for_status()
        return r
    except Exception as e:
        # Reason first, URL last: the handlers print str(e)[:90], and one
        # Roboflow image URL fills that on its own.
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
