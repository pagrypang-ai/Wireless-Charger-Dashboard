from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Iterable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


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


def apply_multiselect(df: pd.DataFrame, options_df: pd.DataFrame, label: str, column: str) -> pd.DataFrame:
    options = sorted_options(options_df[column])
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


def finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number) or not np.isfinite(number):
        return None
    return number


def compact_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def detail_pair(label: str, value: object, kind: str = "text") -> dict[str, str] | None:
    if kind == "money":
        number = finite_number(value)
        text = format_currency(number) if number is not None else ""
    elif kind == "number":
        number = finite_number(value)
        text = f"{number:g}" if number is not None else ""
    elif kind == "rating":
        number = finite_number(value)
        text = f"{number:.1f}" if number is not None else ""
    else:
        text = compact_text(value)

    if not text or text == "n/a":
        return None
    return {"label": label, "value": text}


def component_rows(chart_df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, row in chart_df.iterrows():
        price = finite_number(row.get("price"))
        product_value = finite_number(row.get("product_value"))
        image_url = compact_text(row.get("image_url"))
        if price is None or product_value is None or not image_url:
            continue

        detail_candidates = [
            detail_pair("Platform", row.get("platform")),
            detail_pair("Brand", row.get("brand")),
            detail_pair("Style", row.get("style")),
            detail_pair("Pickup or Not", row.get("pickup_or_not")),
            detail_pair("Availability", row.get("availability_status")),
            detail_pair("Price", row.get("price"), "money"),
            detail_pair("Was Price", row.get("was_price"), "money"),
            detail_pair("Charging Devices", row.get("charging_device_count"), "number"),
            detail_pair("iPhone Max Power (W)", row.get("iphone_power_w"), "number"),
            detail_pair("iPhone max", row.get("iphone_max")),
            detail_pair("Watch max", row.get("watch_max")),
            detail_pair("Earbud max", row.get("earbud_max")),
            detail_pair("Adapter included", row.get("adapter_included")),
            detail_pair("Magnetic", row.get("magnetic_or_not")),
            detail_pair("Rating", row.get("rating"), "rating"),
            detail_pair("Reviews", row.get("number_of_reviews"), "number"),
            detail_pair("Key Specs", row.get("key_specs")),
        ]

        rows.append(
            {
                "product": compact_text(row.get("product_label")) or "Unnamed product",
                "price": price,
                "productValue": product_value,
                "imageUrl": image_url,
                "link": compact_text(row.get("link")),
                "availability": compact_text(row.get("availability_status")),
                "fullDetails": [item for item in detail_candidates if item],
            }
        )
    return rows


def render_value_matrix_component(chart_df: pd.DataFrame, domain_df: pd.DataFrame) -> None:
    payload = {
        "rows": component_rows(chart_df),
        "domainRows": component_rows(domain_df),
    }
    data_json = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")

    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17202a; background: #fff; }
    .chart-wrap { position: relative; width: 100%; overflow: hidden; }
    svg { display: block; width: 100%; height: 640px; }
    .product-marker { cursor: pointer; transition: transform .12s ease; }
    .point-image { cursor: pointer; }
    .tooltip {
      position: fixed;
      z-index: 20;
      width: min(520px, calc(100vw - 32px));
      max-height: min(600px, calc(100vh - 32px));
      overflow: auto;
      pointer-events: none;
      background: rgba(17, 24, 39, .96);
      color: white;
      border-radius: 8px;
      padding: 12px 13px;
      box-shadow: 0 18px 45px rgba(15, 23, 42, .28);
      opacity: 0;
      transform: translateY(4px);
      transition: opacity .12s ease, transform .12s ease;
      font-size: 12px;
      line-height: 1.4;
    }
    .tooltip.visible { opacity: 1; transform: translateY(0); }
    .tooltip .title { font-size: 13px; font-weight: 750; margin-bottom: 7px; }
    .tooltip .row { display: flex; justify-content: space-between; gap: 12px; border-top: 1px solid rgba(255,255,255,.12); padding-top: 5px; margin-top: 5px; }
    .tooltip strong { text-align: right; }
  </style>
</head>
<body>
  <div class="chart-wrap">
    <svg id="chart" role="img" aria-label="Wireless charger value matrix"></svg>
    <div class="tooltip" id="tooltip"></div>
  </div>
  <script id="dashboard-data" type="application/json">__DATA__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("dashboard-data").textContent);
    const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
    const money = value => Number.isFinite(value) ? `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "n/a";
    const median = values => {
      const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
      if (!sorted.length) return NaN;
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    };

    function renderChart(data, domainData) {
      const svg = document.getElementById("chart");
      const tooltip = document.getElementById("tooltip");
      svg.innerHTML = "";
      const box = svg.getBoundingClientRect();
      const width = Math.max(760, Math.floor(box.width || 760));
      const height = Math.floor(box.height || 640);
      const margin = { top: 22, right: 28, bottom: 62, left: 76 };
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

      const chartRows = data.filter(row => Number.isFinite(row.price) && Number.isFinite(row.productValue) && row.imageUrl);
      const domainRows = (domainData.length ? domainData : chartRows).filter(row => Number.isFinite(row.price) && Number.isFinite(row.productValue));
      const domainSource = domainRows.length ? domainRows : [{ price: 0, productValue: 0 }, { price: 100, productValue: 100 }];
      const xVals = domainSource.map(row => row.price).filter(Number.isFinite);
      const yVals = domainSource.map(row => row.productValue).filter(Number.isFinite);
      const xMin = Math.min(...xVals), xMax = Math.max(...xVals);
      const yMin = Math.min(...yVals), yMax = Math.max(...yVals);
      const xPad = Math.max((xMax - xMin) * 0.08, 8);
      const yPad = Math.max((yMax - yMin) * 0.08, 4);
      const x0 = Math.max(0, xMin - xPad), x1 = xMax + xPad;
      const y0 = Math.max(0, yMin - yPad), y1 = Math.min(105, yMax + yPad);
      const sx = value => margin.left + ((value - x0) / (x1 - x0 || 1)) * innerW;
      const sy = value => margin.top + innerH - ((value - y0) / (y1 - y0 || 1)) * innerH;
      const ns = "http://www.w3.org/2000/svg";
      const add = (name, attrs, parent = svg) => {
        const el = document.createElementNS(ns, name);
        Object.entries(attrs || {}).forEach(([key, value]) => el.setAttribute(key, value));
        parent.appendChild(el);
        return el;
      };
      const text = (value, x, y, attrs = {}) => {
        const el = add("text", { x, y, ...attrs });
        el.textContent = value;
        return el;
      };

      add("rect", { x: margin.left, y: margin.top, width: innerW, height: innerH, fill: "#ffffff" });
      const xTicks = 5, yTicks = 5;
      for (let i = 0; i <= xTicks; i++) {
        const value = x0 + (x1 - x0) * i / xTicks;
        const x = sx(value);
        add("line", { x1: x, x2: x, y1: margin.top, y2: margin.top + innerH, stroke: "#e6eaf0", "stroke-dasharray": "3 5" });
        text(`$${Math.round(value)}`, x, height - 28, { "text-anchor": "middle", fill: "#536073", "font-size": 12 });
      }
      for (let i = 0; i <= yTicks; i++) {
        const value = y0 + (y1 - y0) * i / yTicks;
        const y = sy(value);
        add("line", { x1: margin.left, x2: margin.left + innerW, y1: y, y2: y, stroke: "#e6eaf0", "stroke-dasharray": "3 5" });
        text(Math.round(value), 48, y + 4, { "text-anchor": "end", fill: "#536073", "font-size": 12 });
      }
      add("line", { x1: margin.left, x2: margin.left + innerW, y1: margin.top + innerH, y2: margin.top + innerH, stroke: "#9aa4b2" });
      add("line", { x1: margin.left, x2: margin.left, y1: margin.top, y2: margin.top + innerH, stroke: "#9aa4b2" });
      text("Price", margin.left + innerW / 2, height - 8, { "text-anchor": "middle", fill: "#344054", "font-size": 13, "font-weight": 700 });
      const yLabel = text("Number of charging devices + iPhone max charging power", 18, margin.top + innerH / 2, { "text-anchor": "middle", fill: "#344054", "font-size": 13, "font-weight": 700 });
      yLabel.setAttribute("transform", `rotate(-90 18 ${margin.top + innerH / 2})`);

      if (domainRows.length) {
        const medianPrice = median(domainRows.map(row => row.price));
        const medianValue = median(domainRows.map(row => row.productValue));
        add("line", { x1: sx(medianPrice), x2: sx(medianPrice), y1: margin.top, y2: margin.top + innerH, stroke: "#9CA3AF", "stroke-dasharray": "7 6" });
        add("line", { x1: margin.left, x2: margin.left + innerW, y1: sy(medianValue), y2: sy(medianValue), stroke: "#9CA3AF", "stroke-dasharray": "7 6" });
      }

      const markerLayer = add("g", { class: "marker-layer" });
      const hoverLayer = add("g", { class: "hover-layer" });
      let activeMarker = null;
      let activeRestore = null;
      const resetActiveMarker = () => {
        if (activeRestore) activeRestore();
        activeMarker = null;
        activeRestore = null;
        tooltip.classList.remove("visible");
      };

      chartRows.forEach((row, index) => {
        const x = sx(row.price), y = sy(row.productValue);
        const size = 42;
        const baseTransform = `translate(${x} ${y})`;
        const hoverTransform = `translate(${x} ${y}) scale(1.1)`;
        const g = add("g", { tabindex: "0", role: "link", "aria-label": row.product, class: "product-marker", transform: baseTransform }, markerLayer);
        add("rect", { x: -size / 2, y: -size / 2, width: size, height: size, rx: 6, fill: "#fff" }, g);
        add("image", { href: row.imageUrl, x: -size / 2 + 2, y: -size / 2 + 2, width: size - 4, height: size - 4, preserveAspectRatio: "xMidYMid meet", class: "point-image" }, g);
        const restore = () => {
          if (g.parentNode !== markerLayer) markerLayer.insertBefore(g, markerLayer.children[index] || null);
          g.setAttribute("transform", baseTransform);
        };
        const lift = () => {
          if (activeMarker && activeMarker !== g && activeRestore) activeRestore();
          activeMarker = g;
          activeRestore = restore;
          if (g.parentNode !== hoverLayer) hoverLayer.appendChild(g);
          g.setAttribute("transform", hoverTransform);
        };
        const show = () => {
          lift();
          const detailRows = (row.fullDetails || []).map(item => `
            <div class="row"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></div>
          `).join("");
          tooltip.innerHTML = `
            <div class="title">${escapeHtml(row.product)}</div>
            <div class="row"><span>Availability on Query Date</span><strong>${escapeHtml(row.availability)}</strong></div>
            <div class="row"><span>Product Value</span><strong>${escapeHtml(row.productValue)}</strong></div>
            ${detailRows}`;
          const svgRect = svg.getBoundingClientRect();
          const pointScreenX = svgRect.left + (x / width) * svgRect.width;
          const pointScreenY = svgRect.top + (y / height) * svgRect.height;
          const tooltipRect = tooltip.getBoundingClientRect();
          const tooltipWidth = tooltipRect.width || 520;
          const tooltipHeight = tooltipRect.height || 320;
          const lowerHalf = y > margin.top + innerH / 2;
          const desiredLeft = pointScreenX + 18;
          const desiredTop = lowerHalf ? pointScreenY - tooltipHeight - 18 : pointScreenY + 18;
          const left = Math.min(window.innerWidth - tooltipWidth - 12, desiredLeft);
          const top = Math.min(window.innerHeight - tooltipHeight - 12, desiredTop);
          tooltip.style.left = `${Math.max(12, left)}px`;
          tooltip.style.top = `${Math.max(12, top)}px`;
          tooltip.classList.add("visible");
        };
        const hide = () => {
          if (activeMarker === g) resetActiveMarker();
          else restore();
        };
        g.addEventListener("mouseenter", show);
        g.addEventListener("mousemove", show);
        g.addEventListener("mouseleave", hide);
        g.addEventListener("focus", show);
        g.addEventListener("blur", hide);
        g.addEventListener("click", () => row.link && window.open(row.link, "_blank", "noopener"));
        g.addEventListener("keydown", event => {
          if ((event.key === "Enter" || event.key === " ") && row.link) window.open(row.link, "_blank", "noopener");
        });
      });
      svg.onmouseleave = resetActiveMarker;
      svg.onmousemove = event => {
        if (!event.target.closest?.(".product-marker")) resetActiveMarker();
      };
    }

    renderChart(DATA.rows, DATA.domainRows);
    window.addEventListener("resize", () => renderChart(DATA.rows, DATA.domainRows));
  </script>
</body>
</html>
""".replace("__DATA__", data_json)

    components.html(html, height=700, scrolling=False)


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
            filtered_df = apply_multiselect(filtered_df, df, label, column)

    st.info(f"Source: {source_label}. Rows loaded: {len(df):,}. Date columns detected: {len(date_columns):,}.")
    render_metric_cards(filtered_df, query_date)

    chart_df = make_chart_dataframe(filtered_df)
    domain_chart_df = make_chart_dataframe(df)
    if chart_df.empty:
        st.warning("No products match the current filters or chart requirements.")
    render_value_matrix_component(chart_df, domain_chart_df)


if __name__ == "__main__":
    main()
