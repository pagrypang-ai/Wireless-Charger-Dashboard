from __future__ import annotations

import os
from pathlib import Path
import re

import altair as alt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Wireless Charger Value Matrix", layout="wide")

APP_TITLE = "Wireless Charger Price-Value Matrix"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "sample_wireless_charger_products.csv"
GOOGLE_SHEET_SECRET = "GOOGLE_SHEET_CSV_URL"
DATE_COLUMN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

VALUE_WEIGHTS = {
    "simultaneous_device_count": 0.50,
    "iphone_power_w": 0.50,
}

RENAME_MAP = {
    "Pickup or not": "Pickup or Not",
    "URL of Image": "Image URL",
    "Simultaneous Charging Devices": "Simultaneous Charging Devices",
    "Supported Device Types": "Supported Device Types",
    "Number of Charging Devices": "Number of Charging Devices",
}

REQUIRED_COLUMNS = [
    "Platform",
    "Brand",
    "Image URL",
    "Pickup or Not",
    "Sold by",
    "Style",
    "Simultaneous Charging Devices",
    "Supported Device Types",
    "Rating",
    "Number of Reviews",
    "Was Price",
    "Price",
    "iPhone max",
    "Watch max",
    "Earbud max",
    "Size",
    "Weight",
    "Warranty",
    "What's Included",
    "Adapter included",
    "Magnetic or not",
    "Link",
]


def parse_number(value) -> float | None:
    if pd.isna(value) or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", "").replace("$", ""))
    return float(match.group(0)) if match else None


def parse_power_watts(value) -> float | None:
    text = "" if pd.isna(value) else str(value)
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*w", text, flags=re.I)
    return max(float(match) for match in matches) if matches else None


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    date_columns = [str(column) for column in df.columns if DATE_COLUMN_RE.match(str(column))]
    return sorted(date_columns, key=lambda column: pd.to_datetime(column))


def availability_on_query_date(row: pd.Series, query_date, date_columns: list[str]) -> str:
    if not date_columns:
        return "Available"

    query_ts = pd.to_datetime(query_date)
    status = "Unavailable"
    for column in date_columns:
        column_ts = pd.to_datetime(column)
        if column_ts > query_ts:
            break

        cell_value = clean_text(row.get(column)).lower()
        if "unavailable" in cell_value:
            status = "Unavailable"
        elif "add" in cell_value or cell_value == "available":
            status = "Available"
    return status


def add_availability_column(df: pd.DataFrame, query_date) -> pd.DataFrame:
    date_columns = detect_date_columns(df)
    result = df.copy()
    result["Availability on Query Date"] = result.apply(
        lambda row: availability_on_query_date(row, query_date, date_columns),
        axis=1,
    )
    return result


def derive_supported_types(value) -> str:
    text = clean_text(value)
    if not text:
        return "Phone"
    tokens = []
    lower = text.lower()
    if "phone" in lower or "pad" in lower or "stand" in lower:
        tokens.append("Phone")
    if "watch" in lower:
        tokens.append("Watch")
    if any(word in lower for word in ["earbud", "earbuds", "airpod", "airpods", "buds"]):
        tokens.append("Earbud")
    return "+".join(dict.fromkeys(tokens)) or "Phone"


def derive_simultaneous_count(style, legacy_devices, supported_types) -> int:
    style_text = clean_text(style).lower()
    legacy_text = clean_text(legacy_devices).lower()
    for text in [style_text, legacy_text]:
        match = re.search(r"(\d+)\s*[- ]?in\s*[- ]?1", text)
        if match:
            return int(match.group(1))
    if style_text in {"pad", "stand", "wireless charger", "car mount"}:
        return 1
    if "phone+watch+earbud" in legacy_text:
        return 3
    if "phone+earbud" in legacy_text and style_text in {"2-in-1", "3-in-1", "4-in-1"}:
        return 2
    return 1


def normalize_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.rename(columns=RENAME_MAP).copy()
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    if "Supported Device Types" not in raw_df.columns and "Number of Charging Devices" in df.columns:
        df["Supported Device Types"] = df["Number of Charging Devices"].apply(derive_supported_types)
    else:
        df["Supported Device Types"] = df["Supported Device Types"].apply(lambda value: clean_text(value) or "Phone")

    if "Simultaneous Charging Devices" not in raw_df.columns and "Number of Charging Devices" in df.columns:
        df["Simultaneous Charging Devices"] = df.apply(
            lambda row: derive_simultaneous_count(
                row.get("Style"),
                row.get("Number of Charging Devices"),
                row.get("Supported Device Types"),
            ),
            axis=1,
        )

    df["Price Num"] = df["Price"].apply(parse_number)
    df["Was Price Num"] = df["Was Price"].apply(parse_number)
    df["Rating Num"] = df["Rating"].apply(parse_number)
    df["Reviews Num"] = df["Number of Reviews"].apply(parse_number)
    df["iPhone Power W"] = df["iPhone max"].apply(parse_power_watts)
    df["Simultaneous Count"] = df["Simultaneous Charging Devices"].apply(parse_number).fillna(1).clip(lower=1)
    df["Product Label"] = df.apply(product_label, axis=1)
    df["Product Value"] = product_value(df)
    return df


def product_label(row) -> str:
    parts = [
        clean_text(row.get("Brand")),
        clean_text(row.get("Style")),
        clean_text(row.get("iPhone max")),
    ]
    return " ".join(part for part in parts if part) or "Unnamed product"


def product_value(df: pd.DataFrame) -> pd.Series:
    max_devices = max(float(df["Simultaneous Count"].max(skipna=True) or 1), 1.0)
    max_power = max(float(df["iPhone Power W"].max(skipna=True) or 1), 1.0)
    device_score = df["Simultaneous Count"].fillna(0) / max_devices
    power_score = df["iPhone Power W"].fillna(0) / max_power
    score = (
        VALUE_WEIGHTS["simultaneous_device_count"] * device_score
        + VALUE_WEIGHTS["iphone_power_w"] * power_score
    ) * 100
    return score.round(1)


@st.cache_data(ttl=900)
def read_csv(path_or_file) -> pd.DataFrame:
    return pd.read_csv(path_or_file, dtype=str, keep_default_na=False)


def get_google_sheet_csv_url() -> str:
    secret_url = ""
    try:
        secret_url = st.secrets.get(GOOGLE_SHEET_SECRET, "")
    except (FileNotFoundError, KeyError):
        secret_url = ""
    return clean_text(secret_url or os.getenv(GOOGLE_SHEET_SECRET, ""))


def normalize_google_sheet_url(url: str) -> str:
    clean_url = clean_text(url)
    if "docs.google.com/spreadsheets" not in clean_url:
        return clean_url
    if any(marker in clean_url for marker in ["format=csv", "output=csv", "tqx=out:csv"]):
        return clean_url

    sheet_id_match = re.search(r"/spreadsheets/d/([^/]+)", clean_url)
    if not sheet_id_match:
        return clean_url
    gid_match = re.search(r"(?:[?#&]gid=)(\d+)", clean_url)
    gid = gid_match.group(1) if gid_match else "0"
    return f"https://docs.google.com/spreadsheets/d/{sheet_id_match.group(1)}/export?format=csv&gid={gid}"


def load_data(uploaded_file) -> tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        return read_csv(uploaded_file), "Uploaded CSV"

    google_sheet_url = normalize_google_sheet_url(get_google_sheet_csv_url())
    if google_sheet_url:
        return read_csv(google_sheet_url), "Google Sheets"

    if DEFAULT_CSV.exists():
        return read_csv(DEFAULT_CSV), str(DEFAULT_CSV)

    st.error(
        "No data source configured. Add GOOGLE_SHEET_CSV_URL in Streamlit Secrets, "
        "upload a CSV from the sidebar, or include sample_wireless_charger_products.csv next to streamlit_app.py."
    )
    st.stop()


def filter_options(df: pd.DataFrame, column: str) -> list[str]:
    values = sorted({clean_text(value) for value in df[column].dropna() if clean_text(value)})
    return values


def apply_multiselect(
    options_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    label: str,
    column: str,
    default_values: list[str] | None = None,
) -> pd.DataFrame:
    options = filter_options(options_df, column)
    if default_values is None:
        default = options
    else:
        default = [value for value in default_values if value in options]
    selected = st.sidebar.multiselect(label, options, default=default)
    if not selected:
        return filtered_df.iloc[0:0]
    return filtered_df[filtered_df[column].isin(selected)]


def money(value) -> str:
    number = parse_number(value)
    return f"${number:,.2f}" if number is not None else ""


def render_metrics(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shown Products", len(df))
    median_price = df["Price Num"].median(skipna=True) if not df.empty else None
    c2.metric("Median Price", f"${median_price:,.2f}" if pd.notna(median_price) else "N/A")
    max_sim = df["Simultaneous Count"].max(skipna=True) if not df.empty else None
    c3.metric("Max Simultaneous", f"{max_sim:g}" if pd.notna(max_sim) else "N/A")
    median_value = df["Product Value"].median(skipna=True) if not df.empty else None
    c4.metric("Median Value", f"{median_value:,.1f}" if pd.notna(median_value) else "N/A")


def render_chart(df: pd.DataFrame) -> None:
    chart_df = df.dropna(subset=["Price Num", "Product Value"]).copy()
    if chart_df.empty:
        st.info("No products match the current filters.")
        return

    tooltip = [
        alt.Tooltip("Product Label:N", title="Product"),
        alt.Tooltip("Brand:N"),
        alt.Tooltip("Style:N"),
        alt.Tooltip("Supported Device Types:N"),
        alt.Tooltip("Simultaneous Count:Q", title="Simultaneous devices"),
        alt.Tooltip("iPhone max:N"),
        alt.Tooltip("Watch max:N"),
        alt.Tooltip("Earbud max:N"),
        alt.Tooltip("Price Num:Q", title="Price", format="$,.2f"),
        alt.Tooltip("Product Value:Q", format=",.1f"),
        alt.Tooltip("Adapter included:N"),
        alt.Tooltip("Magnetic or not:N"),
    ]

    base = alt.Chart(chart_df).encode(
        x=alt.X("Price Num:Q", title="Price", scale=alt.Scale(zero=False), axis=alt.Axis(format="$,.0f")),
        y=alt.Y(
            "Product Value:Q",
            title="Simultaneous charging devices + iPhone max charging power",
            scale=alt.Scale(zero=False),
        ),
        tooltip=tooltip,
        href="Link:N",
    )
    image = base.mark_image(width=52, height=52, aspect=False).encode(url="Image URL:N")
    dots = base.mark_circle(size=180, opacity=0.75).encode(color=alt.Color("Brand:N", legend=None))
    labels = base.mark_text(dy=28, fontSize=10, opacity=0.55).encode(text="Brand:N")
    chart = (dots + image + labels).properties(height=620).interactive()
    st.altair_chart(chart, use_container_width=True)


def main() -> None:
    st.title(APP_TITLE)
    st.caption("Revised prototype: simultaneous charging capacity is separated from supported device compatibility.")

    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    raw_df, source = load_data(uploaded_file)
    df = normalize_columns(raw_df)

    st.sidebar.caption(f"Data source: {source}")

    date_columns = detect_date_columns(df)
    if date_columns:
        query_date = st.sidebar.selectbox("Query Date", date_columns, index=len(date_columns) - 1)
        df = add_availability_column(df, query_date)
    else:
        df["Availability on Query Date"] = "Available"

    filtered = df.copy()
    for label, column in [
        ("Channels", "Platform"),
        ("Availability on Query Date", "Availability on Query Date"),
        ("Brand", "Brand"),
        ("Style", "Style"),
        ("Supported Device Types", "Supported Device Types"),
        ("Simultaneous Charging Devices", "Simultaneous Charging Devices"),
        ("iPhone max", "iPhone max"),
        ("Adapter included", "Adapter included"),
        ("Magnetic or not", "Magnetic or not"),
    ]:
        default = ["Available"] if column == "Availability on Query Date" else None
        filtered = apply_multiselect(df, filtered, label, column, default)

    render_metrics(filtered)
    render_chart(filtered)


if __name__ == "__main__":
    main()
