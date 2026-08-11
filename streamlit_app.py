from __future__ import annotations

import hashlib
import math
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
BLANK_FILTER_LABEL = "(Blank)"
CHART_DISPLAY_IMAGES = "Product images"
CHART_DISPLAY_DOTS = "Brand dots"
CHART_DISPLAY_OPTIONS = [CHART_DISPLAY_IMAGES, CHART_DISPLAY_DOTS]

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

COLUMN_ALIASES = {
    "channel": "Platform",
    "channels": "Platform",
    "platform": "Platform",
    "retailer": "Platform",
    "url of image": "Image URL",
    "image url": "Image URL",
    "product image": "Image URL",
    "product image url": "Image URL",
    "pickup or not": "Pickup or Not",
    "pickup or not?": "Pickup or Not",
    "pickup": "Pickup or Not",
    "sold by": "Sold by",
    "seller": "Sold by",
    "number of charging devices": "Number of Charging Devices",
    "number of charging device": "Number of Charging Devices",
    "charging devices": "Number of Charging Devices",
    "simultaneous charging devices": "Simultaneous Charging Devices",
    "simultaneous devices": "Simultaneous Charging Devices",
    "simultaneous count": "Simultaneous Charging Devices",
    "supported device types": "Supported Device Types",
    "supported devices": "Supported Device Types",
    "supported device compatibility": "Supported Device Types",
    "iphone max": "iPhone max",
    "iphone max charging power": "iPhone max",
    "iphone max wireless charging power": "iPhone max",
    "iphone power": "iPhone max",
    "phone max": "iPhone max",
    "phone max charging power": "iPhone max",
    "watch max": "Watch max",
    "watch max charging power": "Watch max",
    "earbud max": "Earbud max",
    "earbuds max": "Earbud max",
    "earbud max charging power": "Earbud max",
    "adapter included": "Adapter included",
    "power adapter included": "Adapter included",
    "adapter include": "Adapter included",
    "magnetic or not": "Magnetic or not",
    "magnetic charging": "Magnetic or not",
    "magnetic": "Magnetic or not",
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


def header_key(value) -> str:
    return re.sub(r"[^a-z0-9?]+", " ", clean_text(value).lower()).strip()


def looks_like_url(value) -> bool:
    return clean_text(value).lower().startswith(("http://", "https://"))


def canonical_column_name(column) -> str:
    text = clean_text(column)
    return RENAME_MAP.get(text) or COLUMN_ALIASES.get(header_key(text), text)


def normalize_input_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    for source_column in list(df.columns):
        target_column = canonical_column_name(source_column)
        if target_column == source_column:
            continue

        if target_column in df.columns:
            target_is_blank = df[target_column].apply(clean_text).eq("")
            if target_column == "Image URL":
                source_has_url = df[source_column].apply(looks_like_url)
                target_is_blank = target_is_blank | (~df[target_column].apply(looks_like_url) & source_has_url)
            df[target_column] = df[target_column].where(~target_is_blank, df[source_column])
            df = df.drop(columns=[source_column])
        else:
            df = df.rename(columns={source_column: target_column})
    return df


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
    df = normalize_input_columns(raw_df)
    has_legacy_devices = "Number of Charging Devices" in df.columns
    has_simultaneous_devices = "Simultaneous Charging Devices" in df.columns
    has_supported_types = "Supported Device Types" in df.columns

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    if not has_supported_types and has_legacy_devices:
        df["Supported Device Types"] = df["Number of Charging Devices"].apply(derive_supported_types)
    else:
        df["Supported Device Types"] = df.apply(
            lambda row: clean_text(row.get("Supported Device Types"))
            or (derive_supported_types(row.get("Number of Charging Devices")) if has_legacy_devices else "Phone"),
            axis=1,
        )

    if not has_simultaneous_devices and has_legacy_devices:
        df["Simultaneous Charging Devices"] = df.apply(
            lambda row: derive_simultaneous_count(
                row.get("Style"),
                row.get("Number of Charging Devices"),
                row.get("Supported Device Types"),
            ),
            axis=1,
        )
    elif has_legacy_devices:
        df["Simultaneous Charging Devices"] = df.apply(
            lambda row: clean_text(row.get("Simultaneous Charging Devices"))
            or derive_simultaneous_count(
                row.get("Style"),
                row.get("Number of Charging Devices"),
                row.get("Supported Device Types"),
            ),
            axis=1,
        )

    df["Price Num"] = pd.to_numeric(df["Price"].apply(parse_number), errors="coerce")
    df["Was Price Num"] = pd.to_numeric(df["Was Price"].apply(parse_number), errors="coerce")
    df["Rating Num"] = pd.to_numeric(df["Rating"].apply(parse_number), errors="coerce")
    df["Reviews Num"] = pd.to_numeric(df["Number of Reviews"].apply(parse_number), errors="coerce")
    df["iPhone Power W"] = pd.to_numeric(df["iPhone max"].apply(parse_power_watts), errors="coerce")
    df["Simultaneous Count"] = (
        pd.to_numeric(df["Simultaneous Charging Devices"].apply(parse_number), errors="coerce")
        .fillna(1)
        .clip(lower=1)
    )
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
    simultaneous_count = pd.to_numeric(df["Simultaneous Count"], errors="coerce").fillna(1).clip(lower=1)
    iphone_power = pd.to_numeric(df["iPhone Power W"], errors="coerce").fillna(0).clip(lower=0)

    max_devices = simultaneous_count.max(skipna=True)
    if pd.isna(max_devices) or max_devices <= 0:
        max_devices = 1.0

    max_power = iphone_power.max(skipna=True)
    device_score = simultaneous_count / float(max_devices)
    if pd.isna(max_power) or max_power <= 0:
        power_score = iphone_power * 0
    else:
        power_score = iphone_power / float(max_power)

    score = (
        VALUE_WEIGHTS["simultaneous_device_count"] * device_score
        + VALUE_WEIGHTS["iphone_power_w"] * power_score
    ) * 100
    return score.round(1)


def add_plot_jitter(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Plot Price Num"] = result["Price Num"]
    result["Plot Product Value"] = result["Product Value"]

    valid_mask = result["Price Num"].notna() & result["Product Value"].notna()
    valid_df = result.loc[valid_mask]
    if valid_df.empty:
        return result

    price_range = float(valid_df["Price Num"].max() - valid_df["Price Num"].min())
    value_range = float(valid_df["Product Value"].max() - valid_df["Product Value"].min())
    x_radius = max(price_range * 0.004, 0.35)
    y_radius = max(value_range * 0.006, 0.75)

    grouped = valid_df.groupby(["Price Num", "Product Value"], dropna=False, sort=False)
    for _coordinates, indexes in grouped.groups.items():
        index_list = list(indexes)
        count = len(index_list)
        if count <= 1:
            continue

        for position, index in enumerate(index_list):
            angle = 2 * math.pi * position / count
            result.at[index, "Plot Price Num"] = result.at[index, "Price Num"] + math.cos(angle) * x_radius
            result.at[index, "Plot Product Value"] = result.at[index, "Product Value"] + math.sin(angle) * y_radius

    return result


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
    values = {clean_text(value) or BLANK_FILTER_LABEL for value in df[column].dropna()}
    if df[column].isna().any():
        values.add(BLANK_FILTER_LABEL)
    values = sorted(values)
    return values


def filter_widget_key(column: str, options: list[str]) -> str:
    option_signature = hashlib.md5("|".join(options).encode("utf-8")).hexdigest()[:10]
    return f"filter_{header_key(column)}_{option_signature}"


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
    try:
        selected = st.sidebar.multiselect(label, options, default=default, key=filter_widget_key(column, options))
    except TypeError:
        selected = st.sidebar.multiselect(label, options, default=default)
    if not selected:
        return filtered_df.iloc[0:0]
    filter_values = filtered_df[column].apply(lambda value: clean_text(value) or BLANK_FILTER_LABEL)
    return filtered_df[filter_values.isin(selected)]


def default_filter_values(column: str) -> list[str] | None:
    return None


def chart_layers_for_mode(display_mode: str) -> tuple[str, ...]:
    if display_mode == CHART_DISPLAY_DOTS:
        return ("dots", "labels")
    return ("image",)


def select_chart_display_mode() -> str:
    return st.sidebar.radio("Chart Display", CHART_DISPLAY_OPTIONS, index=0, horizontal=True)


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


def render_chart(df: pd.DataFrame, display_mode: str) -> None:
    chart_df = df.dropna(subset=["Price Num", "Product Value"]).copy()
    if chart_df.empty:
        st.info("No products match the current filters.")
        return
    chart_df = add_plot_jitter(chart_df)

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
        x=alt.X("Plot Price Num:Q", title="Price", scale=alt.Scale(zero=False), axis=alt.Axis(format="$,.0f")),
        y=alt.Y(
            "Plot Product Value:Q",
            title="Simultaneous charging devices + iPhone max charging power",
            scale=alt.Scale(zero=False),
        ),
        tooltip=tooltip,
        href="Link:N",
    )
    image = base.mark_image(width=52, height=52, aspect=False).encode(url="Image URL:N")
    dots = base.mark_circle(size=180, opacity=0.75).encode(color=alt.Color("Brand:N", legend=None))
    labels = base.mark_text(dy=28, fontSize=10, opacity=0.55).encode(text="Brand:N")

    layer_map = {"image": image, "dots": dots, "labels": labels}
    layers = [layer_map[name] for name in chart_layers_for_mode(display_mode)]
    chart = layers[0]
    for layer in layers[1:]:
        chart = chart + layer
    chart = chart.properties(height=620).interactive()
    st.altair_chart(chart, use_container_width=True)


def main() -> None:
    st.title(APP_TITLE)
    st.caption("Revised prototype: simultaneous charging capacity is separated from supported device compatibility.")

    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    raw_df, source = load_data(uploaded_file)
    df = normalize_columns(raw_df)

    st.sidebar.caption(f"Data source: {source}")
    chart_display_mode = select_chart_display_mode()

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
        filtered = apply_multiselect(df, filtered, label, column, default_filter_values(column))

    render_metrics(filtered)
    render_chart(filtered, chart_display_mode)


if __name__ == "__main__":
    main()
