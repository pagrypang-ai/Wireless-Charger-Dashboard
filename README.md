# Wireless Charger Price-Value Dashboard

Streamlit prototype for a wireless charger product value matrix.

## Files

- `streamlit_app.py` - Streamlit dashboard app.
- `requirements.txt` - Python dependencies for local or Streamlit Cloud deployment.
- `sample_wireless_charger_products.csv` - Sample data exported from the provided workbook.
- `.streamlit/secrets.toml.example` - Example secrets file.

## Data Source

The app loads data in this order:

1. A CSV uploaded in the sidebar.
2. `GOOGLE_SHEET_CSV_URL` from Streamlit Secrets.
3. The bundled sample CSV.

For Streamlit Cloud, add this secret:

```toml
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/.../pub?output=csv"
```

For local testing, create `.streamlit/secrets.toml` with the same value, or use the sidebar uploader.

## Field Handling

The app normalizes column names before processing, so minor naming differences and case differences should not break the dashboard.

Expected fields include:

- `Platform`
- `Brand`
- `URL of Image`
- `Pickup or not`
- `Style`
- `Number of Charging Devices` or `Type`
- `Price`
- `iPhone max`
- `Adapter included`
- `Magnetic or not`
- `Link`
- Date columns such as `2026-06-03`

Date columns are detected automatically. `Add` marks a product as available from that date, and `Unavailable` marks it unavailable from that date. Other lifecycle notes, such as `Clearance`, do not change availability status.

## Filters

The sidebar filters support multi-select by default. Use `Clear all` to remove every selected value across all filters, or `Select all` to restore every available filter value after clearing.

## Product Value Logic

Product Value is calculated as a 0-100 score:

- 50% from number of charging devices.
- 50% from iPhone max wireless charging power.

The chart labels the y-axis as:

`Number of charging devices + iPhone max charging power`

The device count parser supports values like `Phone+Watch+Earbud`, `Phone+Earbud`, `3-in-1`, `2-in-1`, `Pad`, and `Stand`.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy

Upload the files in this package to Streamlit Cloud, set `GOOGLE_SHEET_CSV_URL` in Secrets, and deploy `streamlit_app.py`.
