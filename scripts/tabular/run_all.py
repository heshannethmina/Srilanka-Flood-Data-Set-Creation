"""Run the full build pipeline end-to-end (after 01_download.py has cached raw data)."""
import runpy, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for step in ["02_features.py", "03_labels_splits.py", "04_graph.py", "05_validate.py"]:
    print(f"\n{'='*70}\n>>> {step}\n{'='*70}")
    runpy.run_path(os.path.join(HERE, step), run_name="__main__")
print("\nPipeline complete. See data/processed/flood_dataset.parquet")
