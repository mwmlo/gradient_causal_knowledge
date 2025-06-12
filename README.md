# Analysing relationships and disagreements between attribution methods

This code repository contains experiments and evaluations for the final year project "Analysing relationships and disagreements between attribution methods", submitted to Imperial College London's Department of Computing.

Core functionality for our experimental framework is located in:

- `attribution_methods.py` (implementations of attribution methods such as activation patching and layerwise Integrated Gradients)
- `testing.py` (evaluation dataset and performance measurement functionality)
- `plotting.py` (useful visualition utility functions)

Experiments and graphs are contained in Jupyter notebooks. Key experiments detailed in the report are `1 - Aligned baselines`, `2 - Important Disagreements`, `4 - Latent components`, `5 - CounterFact`, `7 - Latent Circuits` and `10 - Verifying Disagreements`. The remaining notebooks contain supplementary experiments.

Evaluations are contained in submodules. Evaluations for circuit localisation on the Mechanistic Interpretability Benchmark are stored in `MIB-circuit-track`, and evaluations for model editing on the CounterFact benchmark are stored in `rome-evaluations`.

To set up and run these experiments, first create a Python virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
```

Then install the requirements using `pip install -r requirements.txt`. Note that this repository uses Python 3.12.3.
