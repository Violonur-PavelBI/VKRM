from .segmentation import (
    SegmentationHead,
    DynamicSegmentationHead,
    ZeroSegmentationHead,
    DynamicZeroSegmentationHead,
)


head_name2class = {
    SegmentationHead.__name__: SegmentationHead,
    DynamicSegmentationHead.__name__: DynamicSegmentationHead,
    ZeroSegmentationHead.__name__: ZeroSegmentationHead,
    DynamicZeroSegmentationHead.__name__: DynamicZeroSegmentationHead,
}
