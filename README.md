# Bioinformatics

This repository contains tools, notebooks, and scripts for processing electrophysiology data, with a focus on burst detection, data analysis, and visualization. 
It supports experiments involving neural recordings, including Parkinson’s disease models using MaxOne and MaxTwo arrays.

---

## Project Structure
```
bioinformatics/
│
├── analysis_libs/ 
│ └── burst_analysis/
│ ├── detection.py
│ ├── plots.py
│ ├── metrics.py
│ └── loader.py
│
├── projects/
│ └── parkinsons/ 
│ ├── environment/ # Environment/conda files
│ ├── notebooks/ # Jupyter notebooks
│ ├── results/ # Output (gitignored)
│ ├── scripts/ # Experiment-related code (empty right now)
│ └── orchestrator.py
│
├── data/ # Raw and processed data (gitignored)
│ └── extracted/
│ ├── exp1/
│ ├── maxone_run1/
│ └── maxtwo_newconfig1/
│
├── docs/ # Documentation
├── notebooks/ # Shared or exploratory notebooks
├── .gitignore
└── README.md
```
## Environment

To set up the environment:

```
bash
conda env create -f projects/parkinsons/environment/brain.yml
conda activate brain
```

## Git Ignored Content
The following are excluded from version control via .gitignore:
- Raw/processed data (/data/)
- Output files (*.npz, *.csv, *.pkl, etc.)
- Jupyter checkpoints
- Environment folders (/env/, environment/, etc.)
- Cache and build artifacts
