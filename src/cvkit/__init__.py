"""cvkit -- building object-detection datasets from video.

Four stages, each a command:

    source   download, frames         video in, frames out
    mine     mine, sheets, collect    a detector proposes, you dispose
    curate   fix-export, verify       repair and re-check an export
    ship     upload-*, tag, train     move it into Roboflow, train on it

Nothing here decides what a label is. Detectors propose; a human approves.
"""

__version__ = "0.1.0"
