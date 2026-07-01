from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Iterable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "Wireless Charger Price-Value Matrix"
SAMPLE_CSV = Path(__file__).with_name("sample_wireless_charger_products.csv")
VALUE_WEIGHTS = {
    "charging_device_count": 0.50,
    "iphone_power_w": 0.50,
}

FIELD_ALIASES = {
    "platform": "platform",
    "brand": "brand",
    "model number": "model_number",
    "model": "model_number",
    "product image": "product_image",
    "url of image": "image_url",
    "image url": "image_url",
    "image": "image_url",
    "pickup or not": "pickup_or_not",
    "pickup": "pickup_or_not",
    "sold by": "sold_by",
    "seller": "sold_by",
    "asin": "sku",
    "sku": "sku",
    "style": "style",
    "type": "number_of_charging_devices",
    "number of charging devices": "number_of_charging_devices",
    "charging devices": "number_of_charging_devices",
    "rating": "rating",
    "number of reviews": "number_of_reviews",
    "reviews": "number_of_reviews",
    "was price": "was_price",
    "list price": "was_price",
    "price": "price",
    "color": "color",
    "colour": "color",
    "iphone max": "iphone_max",
    "iphone max charging power": "iphone_max",
    "watch max": "watch_max",
    "earbud max": "earbud_max",
    "size": "size",
    "weight": "weight",
    "warranty": "warranty",
    "whats included": "whats_included",
    "what s included": "whats_included",
    "adapter included": "adapter_included",
    "magnetic or not": "magnetic_or_not",
    "magnetic": "magnetic_or_not",
    "link": "link",
    "product link": "link",
}

REQUIRED_TEXT_COLUMNS = [
    "platform",
    "brand",
    "image_url",
    "pickup_or_not",
    "sold_by",
    "style",
    "number_of_charging_devices",
    "iphone_max",
    "watch_max",
    "earbud_max",
    "size",
    "weight",
    "warranty",
    "whats_included",
    "adapter_included",
    "magnetic_or_not",
    "link",
]

FILTERS = [
    ("Platform", "platform"),
    ("Brand", "brand"),
    ("Pickup or Not", "pickup_or_not"),
    ("Availability on Query Date", "availability_status"),
    ("Style", "style"),
    ("iPhone max", "iphone_max"),
    ("Adapter included", "adapter_included"),
    ("Magnetic or not", "magnetic_or_not"),
]

FILTER_STATE_PREFIX = "filter__"


def clean_key(value: object) -> str:
    text = "" if value is None else str(value).strip()
    text = text.replace("\ufeff", "")
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return re.sub(r"\s+", " ", text)


def parse_date_header(value: object) -> date | None:
    if value is None or str(value).strip() == "":
        return None

    text = str(value).strip()
    if re.fullmatch(r"\d+(\.0)?", text):
        serial = float(text)
        if 30000 <= serial <= 70000:
            return (pd.Timestamp("1899-12-30") + pd.to_timedelta(serial, unit="D")).date()
        return None

    if not re.search(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", text):
        return None

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    renamed: dict[object, str] = {}
    date_columns: list[str] = []
    used: set[str] = set()

    for original in df.columns:
        parsed_date = parse_date_header(original)
        if parsed_date:
            canonical = parsed_date.isoformat()
            date_columns.append(canonical)
        else:
            canonical = FIELD_ALIASES.get(clean_key(original), clean_key(original).replace(" ", "_"))
            if not canonical:
                canonical = "unnamed"

        base = canonical
        counter = 2
        while canonical in used:
            canonical = f"{base}_{counter}"
            counter += 1
        used.add(canonical)
        renamed[original] = canonical

    normalized = df.rename(columns=renamed)
    normalized = normalized.dropna(axis=1, how="all")

    for col in REQUIRED_TEXT_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = ""

    return normalized, [col for col in date_columns if col in normalized.columns]


def read_csv_safely(source) -> pd.DataFrame:
    return pd.read_csv(source, dtype=str, keep_default_na=False)


@st.cache_data(ttl=900, show_spinner=False)
def load_remote_csv(url: str) -> pd.DataFrame:
    return read_csv_safely(url)


@st.cache_data(show_spinner=False)
def load_sample_csv() -> pd.DataFrame:
    return read_csv_safely(SAMPLE_CSV)


def load_data(uploaded_file) -> tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        return read_csv_safely(uploaded_file), "Uploaded CSV"

    sheet_url = st.secrets.get("GOOGLE_SHEET_CSV_URL", "")
    if sheet_url:
        return load_remote_csv(sheet_url), "Google Sheets CSV"

    return load_sample_csv(), "Bundled sample CSV"


def extract_numbers(value: object) -> list[float]:
    text = "" if value is None else str(value)
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return [float(match) for match in matches]


def parse_money(value: object, choose: str = "last") -> float:
    numbers = extract_numbers(value)
    if not numbers:
        return np.nan
    return numbers[0] if choose == "first" else numbers[-1]


def parse_power_watts(value: object) -> float:
    text = "" if value is None else str(value)
    watt_matches = re.findall(r"(\d+(?:\.\d+)?)\s*w", text, flags=re.IGNORECASE)
    if watt_matches:
        return max(float(match) for match in watt_matches)
    return np.nan


def parse_device_count(value: object) -> float:
    text = "" if value is None else str(value).strip()
    lower = text.lower()
    if not lower:
        return np.nan

    in_one = re.search(r"(\d+)\s*[- ]?\s*in\s*[- ]?\s*1", lower)
    if in_one:
        return float(in_one.group(1))

    plus_tokens = [token.strip() for token in re.split(r"[+/,&]", lower) if token.strip()]
    device_tokens = [
        token
        for token in plus_tokens
        if any(word in token for word in ["phone", "watch", "earbud", "airpod", "buds"])
    ]
    if device_tokens:
        return float(len(device_tokens))

    if any(word in lower for word in ["phone", "pad", "stand"]):
        return 1.0

    numbers = extract_numbers(lower)
    return numbers[0] if numbers else np.nan


def normalize_yes_no(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return "Unknown"
    lower = text.lower()
    if lower in {"yes", "y", "true", "included"}:
        return "Yes"
    if lower in {"no", "n", "false", "not included"}:
        return "No"
    return text


def prepare_data(raw_df: pd.DataFrame, query_date: date | None = None) -> tuple[pd.DataFrame, list[str]]:
    df, date_columns = normalize_columns(raw_df.copy())

    for col in REQUIRED_TEXT_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    for col in ["price", "was_price", "rating", "number_of_reviews"]:
        if col not in df.columns:
            df[col] = ""

    df["price"] = df["price"].apply(parse_money)
    df["was_price"] = df["was_price"].apply(lambda value: parse_money(value, choose="first"))
    df["rating"] = df["rating"].apply(parse_money)
    df["number_of_reviews"] = df["number_of_reviews"].apply(parse_money)
    df["iphone_power_w"] = df["iphone_max"].apply(parse_power_watts)

    raw_device_count = df["number_of_charging_devices"].apply(parse_device_count)
    style_device_count = df["style"].apply(parse_device_count)
    df["charging_device_count"] = raw_device_count.fillna(style_device_count).fillna(1.0)

    df["adapter_included"] = df["adapter_included"].apply(normalize_yes_no)
    df["magnetic_or_not"] = df["magnetic_or_not"].apply(normalize_yes_no)
    df["availability_status"] = df.apply(
        lambda row: availability_on_date(row, date_columns, query_date), axis=1
    )

    df["product_label"] = df.apply(build_product_label, axis=1)
    df["product_value"] = calculate_value_score(df)
    df["key_specs"] = df.apply(build_key_specs, axis=1)
    df["full_product_info"] = df.apply(build_full_product_info, axis=1)
    return df, date_columns


def availability_on_date(row: pd.Series, date_columns: Iterable[str], query_date: date | None) -> str:
    if not date_columns or query_date is None:
        pickup = str(row.get("pickup_or_not", "")).lower()
        return "unavailable" if "unavailable" in pickup else "available"

    state_events: list[tuple[date, str]] = []
    for col in date_columns:
        event_date = parse_date_header(col)
        if event_date is None:
            continue
        event_text = str(row.get(col, "")).strip().lower()
        if not event_text:
            continue
        if "unavailable" in event_text:
            state_events.append((event_date, "unavailable"))
        elif "add" in event_text:
            state_events.append((event_date, "available"))

    past_events = [(event_date, status) for event_date, status in state_events if event_date <= query_date]
    if past_events:
        return sorted(past_events, key=lambda item: item[0])[-1][1]

    if state_events:
        return "unavailable"

    pickup = str(row.get("pickup_or_not", "")).lower()
    return "unavailable" if "unavailable" in pickup else "available"


def calculate_value_score(df: pd.DataFrame) -> pd.Series:
    max_devices = max(float(df["charging_device_count"].max(skipna=True) or 1), 1.0)
    max_power = max(float(df["iphone_power_w"].max(skipna=True) or 1), 1.0)
    device_score = df["charging_device_count"].fillna(0) / max_devices
    power_score = df["iphone_power_w"].fillna(0) / max_power
    score = (
        VALUE_WEIGHTS["charging_device_count"] * device_score
        + VALUE_WEIGHTS["iphone_power_w"] * power_score
    ) * 100
    return score.round(1)


def build_product_label(row: pd.Series) -> str:
    parts = [
        str(row.get("brand", "")).strip(),
        str(row.get("model_number", "")).strip(),
        str(row.get("style", "")).strip(),
    ]
    label = " ".join(part for part in parts if part)
    return label or "Unnamed product"


def build_key_specs(row: pd.Series) -> str:
    specs = [
        f"{row.get('charging_device_count', 0):g} devices",
        str(row.get("iphone_max", "")).strip(),
        f"Watch: {row.get('watch_max', '')}".strip(),
        f"Earbud: {row.get('earbud_max', '')}".strip(),
    ]
    return " | ".join(spec for spec in specs if spec and not spec.endswith(":"))


def build_full_product_info(row: pd.Series) -> str:
    display_columns = [
        "platform",
        "brand",
        "product_image",
        "pickup_or_not",
        "sold_by",
        "style",
        "number_of_charging_devices",
        "rating",
        "number_of_reviews",
        "was_price",
        "price",
        "iphone_max",
        "watch_max",
        "earbud_max",
        "size",
        "weight",
        "warranty",
        "whats_included",
        "adapter_included",
        "magnetic_or_not",
    ]
    lines = []
    for column in display_columns:
        if column not in row:
            continue
        value = row.get(column)
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text == "":
            continue
        label = column.replace("_", " ").title()
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


def format_currency(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"${value:,.2f}"


def sorted_options(series: pd.Series) -> list[str]:
    values = [str(value).strip() for value in series.dropna().unique() if str(value).strip()]
    return sorted(values, key=lambda item: item.lower())


def filter_state_key(column: str) -> str:
    return f"{FILTER_STATE_PREFIX}{column}"


def ensure_filter_state(df: pd.DataFrame) -> None:
    for _, column in FILTERS:
        options = sorted_options(df[column])
        key = filter_state_key(column)
        if key not in st.session_state:
            st.session_state[key] = options
            continue

        valid = set(options)
        st.session_state[key] = [value for value in st.session_state[key] if value in valid]


def set_all_filter_values(df: pd.DataFrame, selected: bool) -> None:
    for _, column in FILTERS:
        st.session_state[filter_state_key(column)] = sorted_options(df[column]) if selected else []


def apply_multiselect(df: pd.DataFrame, label: str, column: str) -> pd.DataFrame:
    options = sorted_options(df[column])
    if not options:
        return df
    key = filter_state_key(column)
    selected = st.sidebar.multiselect(
        label,
        options=options,
        key=key,
        help="Choose one or more values. Leave empty to exclude all values for this filter.",
    )
    if not selected:
        return df.iloc[0:0]
    return df[df[column].isin(selected)]


def make_chart_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "product_label",
        "platform",
        "brand",
        "style",
        "pickup_or_not",
        "availability_status",
        "price",
        "was_price",
        "product_value",
        "charging_device_count",
        "iphone_power_w",
        "iphone_max",
        "watch_max",
        "earbud_max",
        "adapter_included",
        "magnetic_or_not",
        "rating",
        "number_of_reviews",
        "key_specs",
        "full_product_info",
        "image_url",
        "link",
    ]
    chart_df = df.loc[:, columns].copy()
    chart_df = chart_df.dropna(subset=["price", "product_value"])
    chart_df = chart_df[chart_df["image_url"].astype(str).str.len() > 0]

    text_cols = [
        "product_label",
        "platform",
        "brand",
        "style",
        "pickup_or_not",
        "availability_status",
        "iphone_max",
        "watch_max",
        "earbud_max",
        "adapter_included",
        "magnetic_or_not",
        "key_specs",
        "full_product_info",
        "image_url",
        "link",
    ]
    for col in text_cols:
        chart_df[col] = chart_df[col].fillna("").astype(str)

    numeric_cols = [
        "price",
        "was_price",
        "product_value",
        "charging_device_count",
        "iphone_power_w",
        "rating",
        "number_of_reviews",
    ]
    for col in numeric_cols:
        chart_df[col] = pd.to_numeric(chart_df[col], errors="coerce").astype(float)

    return chart_df.reset_index(drop=True)


def build_value_matrix(chart_df: pd.DataFrame) -> alt.Chart:
    price_median = float(chart_df["price"].median()) if not chart_df.empty else 0.0
    value_median = float(chart_df["product_value"].median()) if not chart_df.empty else 0.0

    base = alt.Chart(chart_df)
    images = base.mark_image(width=46, height=46).encode(
        x=alt.X("price:Q", title="Price", scale=alt.Scale(zero=False), axis=alt.Axis(format="$,.0f")),
        y=alt.Y(
            "product_value:Q",
            title="Number of charging devices + iPhone max charging power",
            scale=alt.Scale(zero=False),
        ),
        url="image_url:N",
        href="link:N",
        tooltip=[
            alt.Tooltip("product_label:N", title="Product"),
            alt.Tooltip("platform:N", title="Platform"),
            alt.Tooltip("brand:N", title="Brand"),
            alt.Tooltip("style:N", title="Style"),
            alt.Tooltip("pickup_or_not:N", title="Pickup or Not"),
            alt.Tooltip("availability_status:N", title="Availability"),
            alt.Tooltip("price:Q", title="Price", format="$,.2f"),
            alt.Tooltip("was_price:Q", title="Was Price", format="$,.2f"),
            alt.Tooltip("charging_device_count:Q", title="Charging Devices", format=".0f"),
            alt.Tooltip("iphone_power_w:Q", title="iPhone Max Power (W)", format=".1f"),
            alt.Tooltip("iphone_max:N", title="iPhone max"),
            alt.Tooltip("watch_max:N", title="Watch max"),
            alt.Tooltip("earbud_max:N", title="Earbud max"),
            alt.Tooltip("adapter_included:N", title="Adapter included"),
            alt.Tooltip("magnetic_or_not:N", title="Magnetic"),
            alt.Tooltip("rating:Q", title="Rating", format=".1f"),
            alt.Tooltip("number_of_reviews:Q", title="Reviews", format=",.0f"),
            alt.Tooltip("key_specs:N", title="Key Specs"),
            alt.Tooltip("full_product_info:N", title="Full Product Info"),
        ],
    )

    price_rule = alt.Chart(pd.DataFrame({"median_price": [price_median]})).mark_rule(
        color="#9CA3AF", strokeDash=[6, 4], opacity=0.75
    ).encode(x="median_price:Q")

    value_rule = alt.Chart(pd.DataFrame({"median_value": [value_median]})).mark_rule(
        color="#9CA3AF", strokeDash=[6, 4], opacity=0.75
    ).encode(y="median_value:Q")

    return (price_rule + value_rule + images).properties(height=620)


def render_metric_cards(df: pd.DataFrame, query_date: date | None) -> None:
    shown = len(df)
    available = int((df["availability_status"] == "available").sum()) if shown else 0
    median_price = df["price"].median(skipna=True) if shown else np.nan

    if shown:
        min_devices = df["charging_device_count"].min(skipna=True)
        max_devices = df["charging_device_count"].max(skipna=True)
        min_power = df["iphone_power_w"].min(skipna=True)
        max_power = df["iphone_power_w"].max(skipna=True)
        spec_range = f"{min_devices:g}-{max_devices:g} devices / {min_power:g}-{max_power:g}W"
    else:
        spec_range = "n/a"

    card1, card2, card3, card4 = st.columns(4)
    card1.metric("Shown Products", f"{shown:,}")
    card2.metric("Available on Query Date", f"{available:,}")
    card3.metric("Median Price", format_currency(median_price))
    card4.metric("Spec Range", spec_range)

    if query_date is not None:
        st.caption(f"Availability is calculated for {query_date.isoformat()}.")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("A scalable dashboard prototype for wireless charger assortment tracking.")

    with st.sidebar:
        st.header("Data Source")
        uploaded_file = st.file_uploader("Upload local CSV", type=["csv"])
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    raw_df, source_label = load_data(uploaded_file)
    initial_df, date_columns = prepare_data(raw_df)

    date_options = [parse_date_header(col) for col in date_columns]
    date_options = sorted(option for option in date_options if option is not None)
    default_query_date = date_options[-1] if date_options else date.today()

    with st.sidebar:
        st.header("Query")
        query_date = st.date_input("Query Date", value=default_query_date)

    df, date_columns = prepare_data(raw_df, query_date=query_date)

    with st.sidebar:
        st.header("Filters")
        ensure_filter_state(df)
        clear_col, select_col = st.columns(2)
        if clear_col.button("Clear all", use_container_width=True):
            set_all_filter_values(df, selected=False)
            st.rerun()
        if select_col.button("Select all", use_container_width=True):
            set_all_filter_values(df, selected=True)
            st.rerun()
        st.caption("Filters support multi-select. Use Clear all to remove every selected value.")
        filtered_df = df.copy()
        for label, column in FILTERS:
            filtered_df = apply_multiselect(filtered_df, label, column)

    st.info(f"Source: {source_label}. Rows loaded: {len(df):,}. Date columns detected: {len(date_columns):,}.")
    render_metric_cards(filtered_df, query_date)

    chart_df = make_chart_dataframe(filtered_df)
    if chart_df.empty:
        st.warning("No products match the current filters or chart requirements.")
    else:
        st.altair_chart(build_value_matrix(chart_df), use_container_width=True)


if __name__ == "__main__":
    main()
