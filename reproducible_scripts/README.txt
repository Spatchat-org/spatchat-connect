These are standalone reproducibility scripts for reviewer verification.

What should be in the reproducibility bundle:
- repro_config.json with the session's current inputs and analysis settings
- input_raster.tif and, when relevant, analysis_cost.tif
- nodes.csv with the node_id / longitude / latitude columns used in the session
- method-specific `.txt` scripts relevant to analyses prepared in the session

- Scripts are distributed as `.txt` files so reviewers can inspect them easily.
- Rename a script to `.py` to run it directly.
- By default scripts read values from `repro_config.json` in the same folder.
- Scripts do not import SpaTChat app modules or LLM components.

Python requirements (varies per script):
- pandas, numpy, rasterio, pyproj, scikit-image, scipy, networkx

Available scripts:
- convert_reproduce.txt
- lcp_reproduce.txt
- crk_reproduce.txt
- circuit_reproduce.txt
- graph_reproduce.txt
