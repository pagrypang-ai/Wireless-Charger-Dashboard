# Wireless Charger Split Devices Prototype

Streamlit prototype for the revised wireless charger value matrix.

## Data Model

The old `Number of Charging Devices` field is split into two fields:

- `Simultaneous Charging Devices`: maximum number of devices the charger can charge at the same time.
- `Supported Device Types`: device categories the product can support, such as `Phone+Earbud`.

Product Value uses:

- 50% from `Simultaneous Charging Devices`
- 50% from `iPhone max` wireless charging power

`Supported Device Types` is used for filtering and detail display, but not for the quantity score.

## Data Source

The app reads data in this order:

1. Uploaded CSV from the sidebar.
2. Google Sheets CSV URL from Streamlit Secrets.
3. `sample_wireless_charger_products.csv` next to `streamlit_app.py`, if present.

For Streamlit Cloud + Google Sheets, add this secret in **Manage app > Settings > Secrets**:

```toml
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/export?format=csv&gid=YOUR_GID"
```

You can also paste a normal Google Sheets URL with `gid=...`; the app will convert it to a CSV export URL.

The Google Sheet must be accessible to Streamlit Cloud. The simplest setup is:

1. In Google Sheets, use **Share** and allow anyone with the link to view, or use **File > Share > Publish to web**.
2. Make sure the first row contains the dashboard column headers.
3. Set `GOOGLE_SHEET_CSV_URL` in Streamlit Secrets.
4. Restart or rerun the Streamlit app.

The app accepts the new split fields and can derive them from the legacy `Number of Charging Devices` column for quick testing, but the revised CSV is preferred.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
