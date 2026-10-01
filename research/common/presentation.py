"""Reader-facing model names, without changing saved experiment identifiers."""

MODEL_LABELS = {
    'hybrid': 'Reference model',
    'hybrid_sst': 'Reference model + SST',
    'unet': 'U-Net',
    'blend': 'Reference model + U-Net',
    'blend_reference': 'Reference model + U-Net',
    'blend_sst': 'Reference model + SST + U-Net',
    'climatology': 'Climatology',
    'direct_unet': 'Direct U-Net',
    'residual_unet': 'Residual U-Net correction',
    'direct_blend': 'Reference model + direct U-Net',
    'unet_corrected': 'Bias-corrected U-Net',
    'blend_corrected': 'Reference model + bias-corrected U-Net',
}


def model_label(value):
    """Leave unknown names intact so unrelated table values are never relabelled."""
    return MODEL_LABELS.get(value, value)


def display_frame(frame):
    """Return a presentation copy; preserve the original data and numeric values."""
    result = frame.copy()
    for column in ('model', 'modelo', 'Model', 'reference', 'comparator'):
        if column in result:
            result[column] = result[column].map(model_label)
    return result
