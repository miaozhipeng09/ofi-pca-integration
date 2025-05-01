# OFI-PCA-Integration

This repository implements a research-oriented framework for computing Order Flow Imbalance (OFI) across multiple order book depth levels, and integrates them via Principal Component Analysis (PCA) to generate predictive market signals.

The implementation is inspired by the methodology in the paper:
**"Cross-Impact of Order Flow Imbalance in Equity Markets"**  
(Cajueiro, Figueredo, Vasconcelos, 2022)

---

## 📌 Project Structure

- `ofi_pca_main.py`: Main script for parsing LOB data, computing multi-level OFI, and applying PCA to extract the Integrated OFI feature.
- `first_25000_rows.csv`: Sample LOBSTER-style Level-10 order book snapshot (AAPL).
- `ofi_features_output.csv`: Output file containing per-minute OFI features:
  - `ofi_best_level` (Level 1 only)
  - `ofi_level_1` ~ `ofi_level_10` (per depth level)
  - `ofi_integrated` (PCA weighted sum)

---

## 🚀 How to Run

### Requirements

```bash
pip install pandas numpy scikit-learn
# ofi-pca-integration
