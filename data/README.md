# Data

## Included (small, derived)
- `digitized/` literature values digitised from published figures, each named by source (Wang 2017, Niu 2024, Park 2024, Meng 2022, etc.).
- `parameters.csv`  the sampled parameter space with ranges, evidence class, and provenance notes.
- `niu2024_*.csv`, `abm_calibrated_params*.csv`  killing-model calibration inputs.

## Not included (public bulk datasets;; download separately)
These feed the gate-design engine (cancer-vs-normal enzyme expression) and are too
large to redistribute here. Download them from the official portals and place them
as shown:

- **DepMap** cancer expression -> `data/depmap/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`
  and `data/depmap/Model.csv`, from https://depmap.org/portal/download/ (Public release).
- **GTEx** normal-tissue median TPM -> `data/gtex/`, the gene median TPM GCT from
  https://gtexportal.org/home/downloads/adult-gtex .

Both are public. The thesis cites them; they are not redistributed in this repository.
