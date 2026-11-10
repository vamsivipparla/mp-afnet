"""Lung field extraction on the 1 mm isotropic CT grid."""
import numpy as np
import SimpleITK as sitk


def lung_air_threshold(image):
    """Estimate the air–tissue boundary by alternating class means."""
    sample = sitk.GetArrayViewFromImage(image)[::4, ::4, ::4]
    counts, edges = np.histogram(np.clip(sample, -1000, 400), bins=140, range=(-1000, 400))
    centers = (edges[:-1] + edges[1:]) / 2
    threshold = -500.0
    for _ in range(32):
        air = centers < threshold
        if not counts[air].sum() or not counts[~air].sum():
            break
        updated = (np.average(centers[air], weights=counts[air])
                   + np.average(centers[~air], weights=counts[~air])) / 2
        updated = float(np.clip(updated, -700, -350))
        if abs(updated - threshold) < 1:
            return updated
        threshold = updated
    return threshold


def segment_lung_field(image):
    """Smooth, isolate lung air regions, and restore their tissue margins."""
    smoothed = sitk.SmoothingRecursiveGaussian(image, sigma=1.0)
    air = smoothed < lung_air_threshold(smoothed)
    # Break narrow links before region labeling. The input grid is 1 mm isotropic.
    cores = sitk.BinaryErode(air, [2, 2, 2], boundaryToForeground=True)
    components = sitk.RelabelComponentImageFilter()
    labels = components.Execute(sitk.ConnectedComponent(cores))
    label_array = sitk.GetArrayViewFromImage(labels)
    # Exclude exterior air touching the in-plane image rim. Lung fields can
    # legitimately reach the first and last axial slices of a partial scan.
    rim = np.concatenate((label_array[:, 0, :].ravel(), label_array[:, -1, :].ravel(),
                          label_array[:, :, 0].ravel(), label_array[:, :, -1].ravel()))
    exterior = set(np.unique(rim))
    sizes = components.GetSizeOfObjectsInPixels()
    interior = [(label, size) for label, size in enumerate(sizes, start=1) if label not in exterior]
    if not interior:
        raise ValueError("No interior lung air regions found")
    first_label, first_size = interior[0]
    mask = labels == first_label
    if len(interior) > 1:
        second_label, second_size = interior[1]
        if second_size >= 0.1 * first_size:
            mask = mask | (labels == second_label)
    # Restore the eroded rim plus a small pleural margin and fill enclosed
    # dense regions so vessels and solid nodule tissue keep their CT intensity.
    mask = sitk.BinaryFillhole(sitk.BinaryDilate(mask, [4, 4, 4]))
    return sitk.Mask(smoothed, mask, outsideValue=-1000)
