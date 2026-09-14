# Third-party licensing

`cvkit` itself is MIT — see [LICENSE](LICENSE). It bundles no model weights and
no third-party code.

What it *optionally installs* is another matter, and the distinction matters if
you ship anything built with this.

## Ultralytics — AGPL-3.0 (the one to actually think about)

The `detect` extra installs [Ultralytics](https://github.com/ultralytics/ultralytics),
which is **AGPL-3.0**, and its pretrained checkpoints carry the same licence.
Any model you train by starting from those weights inherits that obligation.

In practice: fine for research, internal work, and anything you keep to
yourself. If you deploy a derived model in a network service, AGPL-3.0 requires
you to offer the corresponding source to its users. Ultralytics sells a
commercial licence for this reason. Decide deliberately — do not let it arrive
by accident through a `pip install`.

cvkit keeps the downloads out of your repository (see `detector.load`), which
is a hygiene measure, not a licensing one. It does not change what the licence
requires.

The SBOM makes this checkable rather than a matter of trust: `bom.json` carries
the declared licence of every resolved component, and `ultralytics`,
`ultralytics-platform` and `ultralytics-thop` come through as AGPL-3.0. A
Dependency-Track licence policy will flag them automatically, so an AGPL
dependency cannot enter a build unnoticed.

If AGPL does not suit your use, start from permissively licensed weights
instead. `--model` accepts any local checkpoint Ultralytics can load, and
nothing in cvkit depends on a specific model family.

## Other optional dependencies

The base install has no dependencies at all; each package below arrives only
with the extra that needs it.

| extra | package | licence |
|---|---|---|
| `images` | numpy | BSD-3-Clause |
| `images` | opencv-python | Apache-2.0 |
| `detect` | ultralytics | **AGPL-3.0** |
| `detect` | torch | BSD-3-Clause |
| `detect` | pyyaml | MIT |
| `detect` | numpy, opencv-python | as above |
| `download` | yt-dlp | Unlicense |
| `roboflow-api` | requests | Apache-2.0 |
| `roboflow` | roboflow | Apache-2.0 |
| `roboflow` | requests | Apache-2.0 |

Only the direct dependencies are listed. For the full resolved closure —
every transitive component with its version, PackageURL and declared licence —
see [`bom.json`](bom.json), the CycloneDX SBOM.

`ffmpeg`, required by `download` and `frames`, is invoked as an external
program and is not distributed with cvkit. Its licence depends on the build you
installed — LGPL-2.1+ or GPL-2+ depending on which codecs were compiled in.

Licences are stated as of the versions declared in `pyproject.toml`. Verify
against your own lockfile, or against `bom.json`, before relying on this
table; it is a pointer, not legal advice.

## Your data

Video you download and frames you extract carry whatever rights attach to the
source. cvkit does not grant you any.
