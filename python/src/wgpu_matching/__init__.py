"""Native wgpu-py interface to the WGSL matchers in this distribution."""

from .api import (
    CorrespondenceSet,
    CPMOptions,
    DeepMatchingOptions,
    FeatureMatchField,
    HORGOptions,
    LocalAffineField,
    Match,
    MatchField,
    ProjectionModel,
    ProjectionOptions,
    WGPUMatching,
)
from .dic_detector import DICDetectorOptions, DICDetectorResult
from .features import (
    DaDDetector,
    DeDoDeBDescriptor,
    DeDoDeBOnnxWGPUDescriptor,
    DeDoDeV2Detector,
    DescribedFeatures,
    FeatureCorrespondences,
    FeatureSet,
    KeypointDetector,
    OnnxWGPUKeypointDetector,
    mutual_nearest_cosine,
    mutual_nearest_cosine_wgpu,
)
from .images import load_gray
from .ndf import LocalNCCResult
from .projection import PatchInitialization

__all__ = [
    "CPMOptions",
    "CorrespondenceSet",
    "DICDetectorOptions",
    "DICDetectorResult",
    "DaDDetector",
    "DeDoDeBDescriptor",
    "DeDoDeBOnnxWGPUDescriptor",
    "DeDoDeV2Detector",
    "DeepMatchingOptions",
    "DescribedFeatures",
    "FeatureCorrespondences",
    "FeatureMatchField",
    "FeatureSet",
    "HORGOptions",
    "KeypointDetector",
    "LocalAffineField",
    "LocalNCCResult",
    "Match",
    "MatchField",
    "OnnxWGPUKeypointDetector",
    "PatchInitialization",
    "ProjectionModel",
    "ProjectionOptions",
    "WGPUMatching",
    "load_gray",
    "mutual_nearest_cosine",
    "mutual_nearest_cosine_wgpu",
]

__version__ = "0.1.0"
