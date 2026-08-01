"""Run the Kalu Ganga tabular pipeline (steps 2-5) after 01_download.py."""
import runpy, os
HERE = os.path.dirname(os.path.abspath(__file__))
for step in ["02_features.py", "03_labels_splits.py", "04_graph.py", "05_validate.py"]:
    print(f"\n{'='*70}\n>>> {step}\n{'='*70}")
    runpy.run_path(os.path.join(HERE, step), run_name="__main__")
print("\nTabular pipeline complete -> data/processed/flood_dataset.parquet")
