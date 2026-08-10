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

## Default Data

By default, the app reads:

`../walmart_wireless_chargers_instore_serpapi_split_devices.csv`

You can also upload a CSV in the sidebar. The app accepts the new split fields and can derive them from the legacy `Number of Charging Devices` column for quick testing, but the revised CSV is preferred.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

