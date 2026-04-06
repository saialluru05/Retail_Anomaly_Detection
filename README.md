# Inventory Anomaly Detection

A two-pronged anomaly detection system for warehouse inventory data, combining **rule-based reconciliation** and **ML-based pattern detection** to surface discrepancies, shrinkage, shipment spikes, and seasonal breaks.

---

## Project Structure

```
inventory-anomaly-detection/
│
├── notebooks/
│   ├── Rule_Based_Anomaly_Detection.ipynb    # Daily SOD/EOD reconciliation logic
│   └── ML_Based_Anomaly_Detection.ipynb      # Isolation Forest + STL-based detection
│
├── data/
│   ├── sample/                               # Small anonymized sample CSVs for testing
│   │   ├── Inventory_Snapshot_SOD_sample.csv
│   │   ├── Inventory_Snapshot_EOD_sample.csv
│   │   ├── Receiving_Transactions_sample.csv
│   │   ├── Shipping_Transactions_sample.csv
│   │   ├── Inventory_Daily_sample.csv
│   │   ├── Receiving_MultiDay_sample.csv
│   │   └── Shipping_MultiDay_sample.csv
│   └── .gitkeep                              # Keeps folder tracked; real data not committed
│
├── outputs/                                  # Generated reports (gitignored)
│   └── .gitkeep
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Notebooks

### 1. Rule-Based Anomaly Detection
**File:** `notebooks/Rule_Based_Anomaly_Detection.ipynb`

Performs a single-day inventory reconciliation using the formula:

```
Expected EOD = SOD Quantity + Received Quantity − Shipped Quantity
Variance     = Actual EOD − Expected EOD
```

Flags any item-warehouse combination where variance ≠ 0 and classifies severity:

| Severity | Condition              |
|----------|------------------------|
| NONE     | variance = 0           |
| LOW      | abs(variance) ≤ 5      |
| MEDIUM   | abs(variance) ≤ 20     |
| HIGH     | abs(variance) > 20     |

**Input files:** `Inventory_Snapshot_SOD.csv`, `Inventory_Snapshot_EOD.csv`, `Receiving_Transactions.csv`, `Shipping_Transactions.csv`

**Output:** `Inventory_Anomaly_Report.csv`

---

### 2. ML-Based Anomaly Detection
**File:** `notebooks/ML_Based_Anomaly_Detection.ipynb`

Applies unsupervised ML models over multi-day time series data to detect five anomaly types:

| Anomaly Type              | Method                          | Features Used                          |
|---------------------------|---------------------------------|----------------------------------------|
| Shipment Spikes/Drop-offs | Isolation Forest                | `ship_qty`, 7-day rolling avg/std      |
| Over-Receiving Trend      | Isolation Forest                | `received_qty`, 7-day rolling avg      |
| Inventory Shrinkage       | Isolation Forest                | 7-day inventory slope (polyfit)        |
| Inventory Hoarding        | Isolation Forest                | `on_hand_qty`, 7-day ship avg          |
| Seasonality Break         | STL Decomposition + Isolation Forest | Seasonal residuals from ship_qty  |

A combined `ml_anomaly_flag` is set when any individual anomaly detector fires.

**Input files:** `Inventory_Daily.csv`, `Receiving_MultiDay.csv`, `Shipping_MultiDay.csv`

**Output:** Annotated DataFrame with anomaly columns + `ML_Anomaly_Report.csv`

---

## Data Files

> **Real data is NOT committed to Git.** Only anonymized sample files live in `data/sample/`.

| File | Used In | Description |
|------|---------|-------------|
| `Inventory_Snapshot_SOD.csv` | Rule-Based | Item quantities at start of day |
| `Inventory_Snapshot_EOD.csv` | Rule-Based | Item quantities at end of day |
| `Receiving_Transactions.csv` | Rule-Based | Inbound receiving records |
| `Shipping_Transactions.csv` | Rule-Based | Outbound shipping records |
| `Inventory_Daily.csv` | ML-Based | Daily on-hand qty per item/warehouse |
| `Receiving_MultiDay.csv` | ML-Based | Multi-day receiving history |
| `Shipping_MultiDay.csv` | ML-Based | Multi-day shipping history |

To run locally, place your actual CSV files in the `data/` folder and update `BASE_PATH` in each notebook:

```python
BASE_PATH = "data"   # instead of "/content/drive/My Drive/Datasets"
```

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas
numpy
scikit-learn
statsmodels
jupyter
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/inventory-anomaly-detection.git
cd inventory-anomaly-detection

# Install dependencies
pip install -r requirements.txt

# Add your data files to data/
cp /your/path/*.csv data/

# Launch Jupyter
jupyter notebook
```

Open either notebook from the `notebooks/` folder and run all cells.

---

## Notes

- Both notebooks were originally developed in **Google Colab**. The `drive.mount()` and `BASE_PATH` lines should be updated when running locally (see above).
- The ML notebook uses `contamination=0.03` in all Isolation Forest models — tune this based on your expected anomaly rate.
- STL decomposition requires a minimum of 2 full seasonal periods (14+ days of data per SKU) to produce reliable residuals.
