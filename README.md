# spatchat-connect-dev

Spatchat Connectivity Developer

## Raster surface conversion utility

Use `convert_surfaces.py` to transform a raster between suitability, resistance (cost), and conductance representations.

```bash
python convert_surfaces.py \
    --input path/to/input.tif \
    --input-type suitability \
    --output path/to/output.tif \
    --output-type resistance
```

### Options

* `--eps`: Small epsilon added to avoid division by zero (default: `1e-6`).
* `--max-resistance`: Optional cap applied to resistance values before conversion.
* `--suitability-cap`: Upper limit for suitability outputs (set to `None` to disable).
* `--output-nodata`: Override the output nodata value.
* `--force-dtype`: Force the output datatype (`float32` or `float64`).
