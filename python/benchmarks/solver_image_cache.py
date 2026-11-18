"""Store and verify exact 8-bit solver images for experimental script reuse."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_solver_images(directory, archive, reference, target, *, source_files=None):
    """Write the solver's already quantized images and bind them to one archive."""
    directory, archive = Path(directory), Path(archive)
    reference, target = np.asarray(reference), np.asarray(target)
    if (reference.dtype != np.uint8 or target.dtype != np.uint8 or reference.ndim != 2
            or reference.shape != target.shape):
        raise ValueError("solver images must be equal-sized 2D uint8 arrays")
    source_hashes = None
    if source_files is not None:
        if len(source_files) != 2:
            raise ValueError("source files must contain reference and target")
        source_hashes = [sha256(path) for path in source_files]
    directory.mkdir(parents=True, exist_ok=True)
    ref = directory / f"{archive.stem}_reference.png"
    deformed = directory / f"{archive.stem}_target.png"
    metadata = directory / f"{archive.stem}.json"
    if any(path.exists() for path in (ref, deformed, metadata)):
        raise FileExistsError(f"solver image cache already exists for {archive.stem}")
    Image.fromarray(reference).save(ref)
    Image.fromarray(target).save(deformed)
    record = {"schema": 1, "archive_sha256": sha256(archive), "reference_sha256": sha256(ref),
              "target_sha256": sha256(deformed), "shape": list(reference.shape)}
    if source_hashes is not None:
        record["source_sha256"] = source_hashes
    metadata.write_text(json.dumps(record, indent=2) + "\n")


def read_solver_images(directory, archive, *, source_files=None):
    """Verify hashes and dimensions before returning the stored uint8 images."""
    directory, archive = Path(directory), Path(archive)
    ref = directory / f"{archive.stem}_reference.png"
    deformed = directory / f"{archive.stem}_target.png"
    record = json.loads((directory / f"{archive.stem}.json").read_text())
    if (record.get("schema") != 1 or record["archive_sha256"] != sha256(archive)
            or record["reference_sha256"] != sha256(ref) or record["target_sha256"] != sha256(deformed)):
        raise ValueError("solver image cache fingerprint does not match")
    if (source_files is None) != ("source_sha256" not in record):
        raise ValueError("solver image cache source inputs do not match")
    if source_files is not None and (len(source_files) != 2
                                     or [sha256(path) for path in source_files] != record["source_sha256"]):
        raise ValueError("solver image cache source fingerprint does not match")
    with Image.open(ref) as image:
        reference = np.asarray(image).copy()
    with Image.open(deformed) as image:
        target = np.asarray(image).copy()
    if (reference.dtype != np.uint8 or target.dtype != np.uint8 or reference.ndim != 2
            or list(reference.shape) != record["shape"] or reference.shape != target.shape):
        raise ValueError("solver image cache dimensions or type do not match")
    return reference, target
