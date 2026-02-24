# Colab Workspace

This folder isolates Colab assets from the root project.

## Structure

- `notebooks/`: Colab notebooks for preprocessing and evaluations
- `env/`: Minimal `uv` project used by Colab notebooks

## Dependency model

Notebooks install dependencies with:

1. `uv sync --project colab/env --extra <group>`
2. Fallback to root `requirements.txt` if `uv` fails

Extra groups used by notebooks:

- `preprocessing` → `colab_wikipedia_preprocessing.ipynb`
- `evaluation` → `colab_evaluation.ipynb`
- `mitigation` → `colab_mitigation_evaluation.ipynb`

## Notes

- The notebooks still clone/use the main repo at `/content/AIST-FYP`.
- Script execution remains plain `python ...`; after `uv sync` the notebook prepends `colab/env/.venv/bin` to `PATH`.
- Keep root `requirements.txt` as compatibility fallback for unstable Colab images.
