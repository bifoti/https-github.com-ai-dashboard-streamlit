import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import json
import os
import calendar
import hashlib
import html
import math
from datetime import datetime, timedelta
import telebot
from sklearn.ensemble import RandomForestRegressor
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

# =========================
# LOAD DATA
# =========================
model = joblib.load("sales_forecasting_model.pkl")
latest_values = joblib.load("latest_values.pkl")

df = pd.read_csv("AyamSerayu_3Years_Transaction_Data.csv")
df["Tanggal & Waktu"] = pd.to_datetime(df["Tanggal & Waktu"])

QTY_COLUMN = (
    "Jumlah Produk"
    if "Jumlah Produk" in df.columns
    else "Jumlah"
    if "Jumlah" in df.columns
    else None
)

DATA_MIN_DATE = df["Tanggal & Waktu"].min().normalize()
DATA_MAX_DATE = df["Tanggal & Waktu"].max().normalize()


# =========================
# TELEGRAM CONFIG
# =========================
# Replace this token with your real token from @BotFather.
# The bot file will register users and save their chat_id into users.json.
BOT_TOKEN = "8876275131:AAFqLliTehn630SesjjHaV9J4f4K18EGkC0"
USERS_FILE = "users.json"
LATEST_RESTOCK_FILE = "latest_restocking_alert.json"
AUTO_ALERT_STATE_FILE = "auto_alert_state.json"
ALERT_CENTER_FILE = "alert_center.json"

try:
    telegram_bot = telebot.TeleBot(BOT_TOKEN)
except Exception:
    telegram_bot = None


def ensure_json_file(filename, default_data):
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(default_data, f, indent=4)


def load_registered_users():
    ensure_json_file(USERS_FILE, {})
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def send_telegram_alert(chat_id, message):
    if BOT_TOKEN == "YOUR_BOT_TOKEN":
        st.warning("Please set BOT_TOKEN first.")
        return False

    if telegram_bot is None:
        st.warning("Telegram bot is not ready.")
        return False

    try:
        telegram_bot.send_message(chat_id, message)
        return True
    except Exception as e:
        st.error(f"Telegram alert failed: {e}")
        return False


def save_latest_restock_alert(data):
    with open(LATEST_RESTOCK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)


def load_auto_alert_state():
    ensure_json_file(AUTO_ALERT_STATE_FILE, {})
    try:
        with open(AUTO_ALERT_STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except json.JSONDecodeError:
        state = {}

    return state if isinstance(state, dict) else {}


def save_auto_alert_state(state):
    with open(AUTO_ALERT_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, default=str)


def load_alert_center():
    ensure_json_file(ALERT_CENTER_FILE, [])
    try:
        with open(ALERT_CENTER_FILE, "r", encoding="utf-8") as f:
            alerts = json.load(f)
    except json.JSONDecodeError:
        alerts = []

    return alerts if isinstance(alerts, list) else []


def save_alert_center(alerts):
    with open(ALERT_CENTER_FILE, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=4, default=str)


def add_alert_center_event(company, title, status, channel, severity, message, items=None, dedupe_key=None):
    alerts = load_alert_center()
    if dedupe_key is not None:
        for alert in alerts:
            if alert.get("dedupe_key") == dedupe_key and alert.get("status") == status:
                return alert.get("id")

    alert_id = hashlib.sha256(
        f"{company}|{title}|{status}|{channel}|{datetime.now().isoformat()}".encode("utf-8")
    ).hexdigest()[:12]
    alerts.insert(0, {
        "id": alert_id,
        "dedupe_key": dedupe_key,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "company": company,
        "title": title,
        "status": status,
        "channel": channel,
        "severity": severity,
        "message": message,
        "items": items or []
    })
    save_alert_center(alerts[:100])
    return alert_id

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Restaurant Analytics",
    page_icon=":fork_and_knife:",
    layout="wide"
)

# =========================
# CURRENCY SETTING
# =========================
IDR_TO_MYR = 0.00029


def convert_idr_to_myr(value):
    return value * IDR_TO_MYR


def format_currency(value_idr):
    if st.session_state.get("currency_mode", "MYR") == "MYR":
        return f"RM {convert_idr_to_myr(value_idr):,.2f}"
    return f"Rp {value_idr:,.0f}"


def format_myr(value):
    return f"RM {value:,.2f}"


def format_idr(value):
    return f"Rp {value:,.0f}"


def rm(value):
    return format_currency(value)


def safe_html(value):
    return html.escape(str(value))


def compact_number(value):
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def compact_currency(value_idr):
    if st.session_state.get("currency_mode", "MYR") == "MYR":
        value_myr = convert_idr_to_myr(value_idr)
        if value_myr >= 1_000_000:
            return f"RM {value_myr / 1_000_000:.1f}M"
        if value_myr >= 1_000:
            return f"RM {value_myr / 1_000:.1f}K"
        return f"RM {value_myr:,.0f}"
    return f"Rp {value_idr / 1_000_000:,.1f}M"


def render_kpi_card(title, value, note="", accent="fire"):
    st.markdown(
        f"""
        <div class="kpi-card accent-{accent}">
            <div class="kpi-title">{safe_html(title)}</div>
            <div class="kpi-value">{safe_html(value)}</div>
            <div class="kpi-note">{safe_html(note)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def style_axis(ax):
    ax.set_facecolor("#111820")
    ax.figure.set_facecolor("#0b0f14")
    ax.tick_params(colors="#c8d0d9")
    ax.title.set_color("#f7f2ea")
    ax.xaxis.label.set_color("#c8d0d9")
    ax.yaxis.label.set_color("#c8d0d9")
    for spine in ax.spines.values():
        spine.set_color("#2b333d")
    ax.grid(True, color="#2b333d", linewidth=0.6, alpha=0.45)
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor("#111820")
        legend.get_frame().set_edgecolor("#2b333d")
        for text in legend.get_texts():
            text.set_color("#f7f2ea")

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0b0f14;
    color: #f7f2ea;
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(180deg, rgba(12,17,22,0.96), rgba(9,12,16,1)),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.025) 0 1px, transparent 1px 64px);
}
section[data-testid="stSidebar"] {
    background: #10161d;
    border-right: 1px solid rgba(255,255,255,0.08);
}
.ops-hero {
    background:
        linear-gradient(135deg, rgba(255,90,61,0.18), rgba(47,214,163,0.08) 48%, rgba(242,184,75,0.12)),
        #111820;
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 8px;
    padding: 24px;
    box-shadow: 0 18px 42px rgba(0,0,0,0.28);
    margin-bottom: 20px;
}
.ops-kicker {
    color: #2fd6a3;
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0;
    margin-bottom: 8px;
}
.main-title {
    font-size: 46px;
    font-weight: 900;
    color: #fff7ef;
    line-height: 1.05;
}
.subtitle {
    color: #c8d0d9;
    font-size: 17px;
    margin-top: 10px;
    margin-bottom: 18px;
    max-width: 900px;
}
.service-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
}
.service-chip {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 8px;
    padding: 12px 14px;
}
.service-chip span {
    display: block;
    color: #9aa7b5;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0;
}
.service-chip strong {
    display: block;
    color: #f7f2ea;
    font-size: 17px;
    margin-top: 4px;
}
.kpi-card, .forecast-card {
    background: rgba(17,24,32,0.96);
    border: 1px solid rgba(255,255,255,0.10);
    border-top: 4px solid #ff5a3d;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 14px 32px rgba(0,0,0,0.22);
    min-height: 120px;
}
.kpi-title {
    color: #aab5c1;
    font-size: 14px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0;
}
.kpi-value {
    color: #fff7ef;
    font-size: 34px;
    font-weight: 900;
    margin-top: 8px;
    overflow-wrap: anywhere;
}
.kpi-note {
    color: #8f9ba8;
    font-size: 13px;
    margin-top: 8px;
}
.accent-fire {
    border-top-color: #ff5a3d;
}
.accent-mint {
    border-top-color: #2fd6a3;
}
.accent-amber {
    border-top-color: #f2b84b;
}
.accent-steel {
    border-top-color: #7aa7ff;
}
.forecast-title {
    font-size: 24px;
    font-weight: 800;
    color: #fff7ef;
}
.forecast-desc {
    color: #aab5c1;
    font-size: 14px;
}
.forecast-value {
    color: #2fd6a3;
    font-size: 36px;
    font-weight: 900;
}
.badge {
    display: inline-block;
    background: rgba(255,90,61,0.16);
    color: #ffb19f;
    padding: 6px 10px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 15px;
    text-transform: uppercase;
    letter-spacing: 0;
}
.section-label {
    color: #f2b84b;
    font-size: 15px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0;
    margin: 8px 0 12px 0;
}
.menu-board {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-bottom: 12px;
}
.menu-tile {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 8px;
    padding: 12px;
}
.menu-tile span {
    display: block;
    color: #9aa7b5;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
}
.menu-tile strong {
    display: block;
    color: #f7f2ea;
    font-size: 18px;
    margin-top: 5px;
}
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg,#ff5a3d,#f2b84b);
    color: #16130f;
    border: none;
    border-radius: 8px;
    height: 50px;
    font-weight: bold;
    font-size: 16px;
}
.stButton>button:hover {
    border: none;
    color: #16130f;
    filter: brightness(1.04);
}
[data-testid="stMetric"] {
    background: rgba(17,24,32,0.86);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 14px;
}
@media (max-width: 700px) {
    .main-title {
        font-size: 34px;
    }
    .service-strip, .menu-board {
        grid-template-columns: 1fr;
    }
    .kpi-value, .forecast-value {
        font-size: 28px;
    }
}
</style>
""", unsafe_allow_html=True)

# =========================
# FUNCTIONS
# =========================
@st.cache_data(show_spinner=False)
def get_daily_sales_history(data_rows, min_date, max_date):
    daily_sales = (
        df.groupby(df["Tanggal & Waktu"].dt.normalize())["Total"]
        .sum()
        .sort_index()
        .asfreq("D")
        .fillna(0)
    )
    return daily_sales


@st.cache_data(show_spinner=False)
def get_outlet_share(data_rows, min_date, max_date):
    share = df.groupby("Outlet")["Total"].sum() / df["Total"].sum()
    return share.to_dict()


def series_value(series, date, fallback):
    date = pd.to_datetime(date).normalize()
    if date in series.index and pd.notna(series.loc[date]):
        return float(series.loc[date])
    return float(fallback)


def create_input(date, sales_history=None):
    date = pd.to_datetime(date).normalize()

    if sales_history is None:
        sales_history = get_daily_sales_history(
            len(df),
            str(DATA_MIN_DATE.date()),
            str(DATA_MAX_DATE.date())
        )

    trailing_sales = sales_history[sales_history.index < date].tail(7)
    rolling_mean_7 = (
        float(trailing_sales.mean())
        if len(trailing_sales) > 0
        else float(latest_values["rolling_mean_7"])
    )
    rolling_std_7 = (
        float(trailing_sales.std())
        if len(trailing_sales) > 1
        else float(latest_values["rolling_std_7"])
    )

    return pd.DataFrame({
        "month": [date.month],
        "day": [date.day],
        "dayofweek": [date.dayofweek],
        "weekofyear": [date.isocalendar().week],
        "quarter": [date.quarter],
        "lag_1": [
            series_value(
                sales_history,
                date - pd.Timedelta(days=1),
                latest_values["last_lag_1"]
            )
        ],
        "lag_7": [
            series_value(
                sales_history,
                date - pd.Timedelta(days=7),
                latest_values["last_lag_7"]
            )
        ],
        "rolling_mean_7": [rolling_mean_7],
        "rolling_std_7": [rolling_std_7]
    })


def predict_total_sales(date):
    return predict_total_sales_for_dates([date])[0]


def predict_total_sales_for_dates(dates):
    normalized_dates = [pd.to_datetime(date).normalize() for date in dates]
    if len(normalized_dates) == 0:
        return []

    sales_history = get_daily_sales_history(
        len(df),
        str(DATA_MIN_DATE.date()),
        str(DATA_MAX_DATE.date())
    ).copy()

    predictions = {}
    last_known_date = sales_history.index.max()

    for target_date in sorted(set(normalized_dates)):
        if target_date <= last_known_date:
            input_data = create_input(target_date, sales_history)
            predictions[target_date] = max(float(model.predict(input_data)[0]), 0)
            continue

        current_date = last_known_date + pd.Timedelta(days=1)
        while current_date <= target_date:
            input_data = create_input(current_date, sales_history)
            prediction = max(float(model.predict(input_data)[0]), 0)
            sales_history.loc[current_date] = prediction
            predictions[current_date] = prediction
            last_known_date = current_date
            current_date = current_date + pd.Timedelta(days=1)

    return [predictions[date] for date in normalized_dates]


def get_outlet_prediction(total_prediction, outlet):
    outlet_share = get_outlet_share(
        len(df),
        str(DATA_MIN_DATE.date()),
        str(DATA_MAX_DATE.date())
    )
    return total_prediction * outlet_share.get(outlet, 0)


def forecast_horizon_note(date):
    forecast_date = pd.to_datetime(date).normalize()
    horizon_days = (forecast_date - DATA_MAX_DATE).days
    if horizon_days <= 0:
        return "Historical range"
    if horizon_days <= 30:
        return f"{horizon_days} days beyond data - near-term"
    if horizon_days <= 120:
        return f"{horizon_days} days beyond data - medium horizon"
    return f"{horizon_days} days beyond data - scenario forecast"


@st.cache_data(show_spinner=False)
def build_historical_model_frame(data_rows, min_date, max_date):
    daily_sales = get_daily_sales_history(data_rows, min_date, max_date)
    model_df = pd.DataFrame({
        "sales": daily_sales
    })
    model_df["month"] = model_df.index.month
    model_df["day"] = model_df.index.day
    model_df["dayofweek"] = model_df.index.dayofweek
    model_df["weekofyear"] = model_df.index.isocalendar().week.astype(int)
    model_df["quarter"] = model_df.index.quarter
    model_df["lag_1"] = model_df["sales"].shift(1)
    model_df["lag_7"] = model_df["sales"].shift(7)
    model_df["rolling_mean_7"] = model_df["sales"].shift(1).rolling(7).mean()
    model_df["rolling_std_7"] = model_df["sales"].shift(1).rolling(7).std()
    return model_df.dropna()


def evaluate_sales_model(evaluation_days):
    feature_cols = list(getattr(model, "feature_names_in_", []))
    model_df = build_historical_model_frame(
        len(df),
        str(DATA_MIN_DATE.date()),
        str(DATA_MAX_DATE.date())
    )
    eval_df = model_df.tail(int(evaluation_days)).copy()
    eval_df["Predicted Sales IDR"] = model.predict(eval_df[feature_cols])
    eval_df["Actual Sales IDR"] = eval_df["sales"]
    eval_df["Error IDR"] = eval_df["Actual Sales IDR"] - eval_df["Predicted Sales IDR"]
    eval_df["Abs Error IDR"] = eval_df["Error IDR"].abs()
    eval_df["APE"] = eval_df["Abs Error IDR"] / eval_df["Actual Sales IDR"].replace(0, pd.NA)

    actual = eval_df["Actual Sales IDR"]
    predicted = eval_df["Predicted Sales IDR"]
    mae = float(eval_df["Abs Error IDR"].mean())
    mape = float(eval_df["APE"].dropna().mean() * 100)
    ss_res = float(((actual - predicted) ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    eval_df = eval_df.reset_index()
    eval_df = eval_df.rename(columns={eval_df.columns[0]: "Date"})
    return eval_df, mae, mape, r2


def model_health_label(mape):
    if mape <= 10:
        return "Strong"
    if mape <= 18:
        return "Good"
    if mape <= 28:
        return "Needs Attention"
    return "High Risk"


@st.cache_data(show_spinner=False)
def build_market_basket_matrix(data_rows, min_date, max_date):
    basket_source = (
        df[["ID Struk", "Nama Produk"]]
        .dropna()
        .drop_duplicates()
    )
    basket = pd.crosstab(
        basket_source["ID Struk"],
        basket_source["Nama Produk"]
    ).astype(bool)
    return basket


@st.cache_data(show_spinner=False)
def mine_association_rules(algorithm, min_support, min_confidence, min_lift, data_rows, min_date, max_date):
    basket = build_market_basket_matrix(data_rows, min_date, max_date)

    if algorithm == "Apriori":
        frequent_itemsets = apriori(
            basket,
            min_support=float(min_support),
            use_colnames=True,
            max_len=2
        )
    else:
        frequent_itemsets = fpgrowth(
            basket,
            min_support=float(min_support),
            use_colnames=True,
            max_len=2
        )

    if frequent_itemsets.empty:
        return pd.DataFrame()

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=float(min_confidence)
    )

    if rules.empty:
        return pd.DataFrame()

    rules = rules[rules["lift"] >= float(min_lift)].copy()
    rules = rules[
        (rules["antecedents"].apply(len) == 1) &
        (rules["consequents"].apply(len) == 1)
    ].copy()

    if rules.empty:
        return pd.DataFrame()

    rules["Antecedent"] = rules["antecedents"].apply(lambda items: next(iter(items)))
    rules["Recommended Combo Item"] = rules["consequents"].apply(lambda items: next(iter(items)))
    rules["Support %"] = rules["support"] * 100
    rules["Confidence %"] = rules["confidence"] * 100
    rules["Lift"] = rules["lift"]
    rules["Rule Strength"] = rules["Lift"] * rules["confidence"]
    rules["Algorithm"] = algorithm

    return rules[
        [
            "Antecedent",
            "Recommended Combo Item",
            "Support %",
            "Confidence %",
            "Lift",
            "Rule Strength",
            "Algorithm"
        ]
    ].sort_values(
        ["Rule Strength", "Lift", "Confidence %"],
        ascending=False
    )


def get_frequency_combo_fallback(product_name, top_n=5):
    basket = build_market_basket_matrix(
        len(df),
        str(DATA_MIN_DATE.date()),
        str(DATA_MAX_DATE.date())
    )

    if product_name not in basket.columns:
        return pd.DataFrame(columns=[
            "Recommended Combo Item",
            "Frequency",
            "Support %",
            "Confidence %",
            "Lift",
            "Rule Strength",
            "Algorithm"
        ])

    product_mask = basket[product_name]
    product_count = int(product_mask.sum())
    total_receipts = len(basket)
    rows = []

    for item in basket.columns:
        if item == product_name:
            continue

        item_mask = basket[item]
        pair_count = int((product_mask & item_mask).sum())
        if pair_count == 0:
            continue

        item_support = float(item_mask.sum() / total_receipts)
        confidence = float(pair_count / product_count) if product_count > 0 else 0
        support = float(pair_count / total_receipts)
        lift = confidence / item_support if item_support > 0 else 0

        rows.append({
            "Recommended Combo Item": item,
            "Frequency": pair_count,
            "Support %": support * 100,
            "Confidence %": confidence * 100,
            "Lift": lift,
            "Rule Strength": lift * confidence,
            "Algorithm": "Frequency fallback"
        })

    if len(rows) == 0:
        return pd.DataFrame(columns=[
            "Recommended Combo Item",
            "Frequency",
            "Support %",
            "Confidence %",
            "Lift",
            "Rule Strength",
            "Algorithm"
        ])

    return (
        pd.DataFrame(rows)
        .sort_values(["Rule Strength", "Lift", "Frequency"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def get_product_combo(product_name, top_n=5, algorithm="FP-Growth", min_support=0.01, min_confidence=0.05, min_lift=0.8):
    rules = mine_association_rules(
        algorithm,
        min_support,
        min_confidence,
        min_lift,
        len(df),
        str(DATA_MIN_DATE.date()),
        str(DATA_MAX_DATE.date())
    )

    if not rules.empty:
        product_rules = rules[rules["Antecedent"] == product_name].copy()
        if not product_rules.empty:
            product_rules["Frequency"] = (
                product_rules["Support %"] / 100 * df["ID Struk"].nunique()
            ).round().astype(int)
            result = product_rules[
                [
                    "Recommended Combo Item",
                    "Frequency",
                    "Support %",
                    "Confidence %",
                    "Lift",
                    "Rule Strength",
                    "Algorithm"
                ]
            ].head(top_n).reset_index(drop=True)
            for metric_col in ["Support %", "Confidence %", "Lift", "Rule Strength"]:
                result[metric_col] = result[metric_col].round(3)
            return result

    result = get_frequency_combo_fallback(product_name, top_n)
    for metric_col in ["Support %", "Confidence %", "Lift", "Rule Strength"]:
        if metric_col in result.columns:
            result[metric_col] = result[metric_col].round(3)
    return result


def apply_discount_simulation(base_sales, discount_rate, uplift_multiplier):
    estimated_uplift = discount_rate * uplift_multiplier
    discounted_sales = base_sales * (1 - discount_rate / 100)
    final_sales = discounted_sales * (1 + estimated_uplift / 100)
    return discounted_sales, final_sales


def product_sales_summary(product_name):
    product_df = df[df["Nama Produk"] == product_name]
    total_sales_idr = product_df["Total"].sum()
    total_sales_myr = convert_idr_to_myr(total_sales_idr)
    total_qty = product_df[QTY_COLUMN].sum() if QTY_COLUMN is not None else len(product_df)
    total_transactions = product_df["ID Struk"].nunique()
    avg_basket_idr = total_sales_idr / total_transactions if total_transactions > 0 else 0
    avg_basket_myr = convert_idr_to_myr(avg_basket_idr)

    return {
        "total_sales_idr": total_sales_idr,
        "total_sales_myr": total_sales_myr,
        "total_qty": total_qty,
        "total_transactions": total_transactions,
        "avg_basket_myr": avg_basket_myr
    }


def top_product_by_quantity():
    if QTY_COLUMN is not None:
        return df.groupby("Nama Produk")[QTY_COLUMN].sum().idxmax()
    return df["Nama Produk"].value_counts().idxmax()


def demand_pressure_label(forecast_value, baseline_value):
    if baseline_value <= 0:
        return "Normal"

    change_percent = ((forecast_value - baseline_value) / baseline_value) * 100
    if change_percent >= 20:
        return "High Prep"
    if change_percent >= 8:
        return "Watch"
    if change_percent <= -12:
        return "Light Prep"
    return "Normal"


def build_manager_checklist(forecast_value, baseline_value, peak_day, selected_combo, selected_product):
    pressure = demand_pressure_label(forecast_value, baseline_value)
    checklist = []

    if pressure == "High Prep":
        checklist.append("Increase chicken, rice, and drink prep before peak service.")
        checklist.append("Assign extra staff for cashier and packing during busy hours.")
    elif pressure == "Watch":
        checklist.append("Prepare a moderate stock buffer and monitor rush-hour orders.")
    elif pressure == "Light Prep":
        checklist.append("Keep production lean to avoid over-prep and wastage.")
    else:
        checklist.append("Follow normal prep levels and watch live sales movement.")

    checklist.append(f"Push combo suggestion: {selected_product} + {selected_combo}.")
    checklist.append(f"Use {peak_day} as the main staffing and procurement checkpoint.")
    return pressure, checklist


# =========================
# AI RESTOCKING MODEL FUNCTIONS
# =========================
INGREDIENT_COLUMNS = [
    "Chicken_kg",
    "Rice_kg",
    "Drink_units",
    "Chili_kg",
    "Oil_liter",
    "Egg_units"
]


def format_ingredient_name(name):
    return (
        name.replace("_kg", " (kg)")
        .replace("_units", " (units)")
        .replace("_liter", " (liter)")
        .replace("_", " ")
    )


def estimate_recipe_from_product_name(product_name):
    """
    Recipe mapping assumption per 1 item sold.
    This is only used to convert product sales into ingredient usage.
    The monthly demand itself is predicted by RandomForestRegressor.
    """
    name = str(product_name).lower()

    recipe = {
        "Chicken_kg": 0.0,
        "Rice_kg": 0.0,
        "Drink_units": 0.0,
        "Chili_kg": 0.0,
        "Oil_liter": 0.0,
        "Egg_units": 0.0
    }

    if any(word in name for word in ["ayam", "chicken", "geprek", "bakar", "crispy"]):
        recipe["Chicken_kg"] = 0.22
        recipe["Chili_kg"] = 0.015
        recipe["Oil_liter"] = 0.02

    if any(word in name for word in ["nasi", "rice", "paket"]):
        recipe["Rice_kg"] = 0.18

    if any(word in name for word in ["teh", "es", "air", "kopi", "jeruk", "drink", "minum"]):
        recipe["Drink_units"] = 1.0

    if any(word in name for word in ["telur", "egg"]):
        recipe["Egg_units"] = 1.0

    return recipe


def build_recipe_map(data):
    products = sorted(data["Nama Produk"].dropna().unique())
    return {
        product: estimate_recipe_from_product_name(product)
        for product in products
    }


def create_monthly_ingredient_usage(data, recipe_map):
    working_df = data.copy()
    working_df["Tanggal & Waktu"] = pd.to_datetime(working_df["Tanggal & Waktu"])
    working_df["year"] = working_df["Tanggal & Waktu"].dt.year
    working_df["month"] = working_df["Tanggal & Waktu"].dt.month

    working_df["_ForecastQty"] = (
        working_df[QTY_COLUMN].fillna(0)
        if QTY_COLUMN is not None
        else 1
    )

    for ingredient in INGREDIENT_COLUMNS:
        working_df[ingredient] = working_df.apply(
            lambda row: recipe_map.get(row["Nama Produk"], {}).get(ingredient, 0) * row["_ForecastQty"],
            axis=1
        )

    monthly_usage = (
        working_df.groupby(["year", "month"])[INGREDIENT_COLUMNS]
        .sum()
        .reset_index()
        .sort_values(["year", "month"])
    )

    return monthly_usage


def add_monthly_ai_features(monthly_usage):
    monthly_df = monthly_usage.copy()

    for ingredient in INGREDIENT_COLUMNS:
        monthly_df[f"lag_1_{ingredient}"] = monthly_df[ingredient].shift(1)
        monthly_df[f"rolling_mean_3_{ingredient}"] = monthly_df[ingredient].rolling(3).mean()

    monthly_df["month_sin"] = monthly_df["month"].apply(
        lambda x: math.sin(2 * math.pi * x / 12)
    )
    monthly_df["month_cos"] = monthly_df["month"].apply(
        lambda x: math.cos(2 * math.pi * x / 12)
    )

    monthly_df = monthly_df.dropna()
    return monthly_df


@st.cache_resource
def train_inventory_model(data_rows, min_date, max_date):
    recipe_map = build_recipe_map(df)
    monthly_usage = create_monthly_ingredient_usage(df, recipe_map)
    monthly_features = add_monthly_ai_features(monthly_usage)

    if len(monthly_features) < 3:
        return None, recipe_map, monthly_usage, None, None

    feature_cols = ["year", "month", "month_sin", "month_cos"]

    for ingredient in INGREDIENT_COLUMNS:
        feature_cols.append(f"lag_1_{ingredient}")
        feature_cols.append(f"rolling_mean_3_{ingredient}")

    X = monthly_features[feature_cols]
    y = monthly_features[INGREDIENT_COLUMNS]

    inventory_model = RandomForestRegressor(
        n_estimators=250,
        random_state=42
    )

    inventory_model.fit(X, y)

    latest_month = monthly_usage.iloc[-1]

    return inventory_model, recipe_map, monthly_usage, latest_month, feature_cols


def detect_seasonal_event(simulated_today, forecast_month):
    simulated_today = pd.to_datetime(simulated_today)

    event_name = "Normal Demand"
    multiplier = 1.0
    confidence = 70

    if forecast_month in [3, 4]:
        event_name = "Hari Raya Seasonal Demand"
        multiplier = 1.45
        confidence = 88
    elif simulated_today.day >= 25:
        event_name = "Salary Week Demand Surge"
        multiplier = 1.25
        confidence = 82
    elif simulated_today.weekday() in [4, 5]:
        event_name = "Weekend Demand Spike"
        multiplier = 1.15
        confidence = 76

    return event_name, multiplier, confidence


def predict_monthly_ingredient_demand(inventory_model, monthly_usage, latest_month, feature_cols, forecast_year, forecast_month, seasonal_multiplier):
    recent_months = monthly_usage.tail(3)
    input_data = {
        "year": [forecast_year],
        "month": [forecast_month],
        "month_sin": [math.sin(2 * math.pi * forecast_month / 12)],
        "month_cos": [math.cos(2 * math.pi * forecast_month / 12)]
    }

    for ingredient in INGREDIENT_COLUMNS:
        latest_value = latest_month[ingredient]
        rolling_value = recent_months[ingredient].mean()
        input_data[f"lag_1_{ingredient}"] = [latest_value]
        input_data[f"rolling_mean_3_{ingredient}"] = [rolling_value]

    X_future = pd.DataFrame(input_data)[feature_cols]
    prediction = inventory_model.predict(X_future)[0]

    result = {}
    for ingredient, value in zip(INGREDIENT_COLUMNS, prediction):
        result[ingredient] = max(round(value * seasonal_multiplier, 2), 0)

    return result


PRIORITY_RANK = {
    "SUFFICIENT": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3
}


AUTO_ALERT_THRESHOLDS = {
    "HIGH only": 3,
    "MEDIUM and HIGH": 2,
    "Any shortage": 1
}


def get_auto_alert_items(result_df, threshold_label):
    minimum_rank = AUTO_ALERT_THRESHOLDS[threshold_label]
    working_df = result_df.copy()
    working_df["_priority_rank"] = working_df["Priority"].map(PRIORITY_RANK).fillna(0)
    alert_items = working_df[
        (working_df["_priority_rank"] >= minimum_rank) &
        (working_df["Recommended Restock Quantity"] > 0)
    ].copy()
    return alert_items.sort_values(
        ["_priority_rank", "Recommended Restock Quantity"],
        ascending=[False, False]
    ).drop(columns=["_priority_rank"])


def build_auto_alert_key(company, simulated_today, forecast_year, forecast_month, threshold_label, alert_items):
    payload = {
        "company": company,
        "simulated_today": str(pd.to_datetime(simulated_today).date()),
        "forecast_year": int(forecast_year),
        "forecast_month": int(forecast_month),
        "threshold": threshold_label,
        "items": alert_items[
            ["Ingredient", "Recommended Restock Quantity", "Priority"]
        ].to_dict("records")
    }
    raw_key = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:20]

# =========================
# HEADER
# =========================
data_window = f"{DATA_MIN_DATE.strftime('%d %b %Y')} - {DATA_MAX_DATE.strftime('%d %b %Y')}"
total_orders = df["ID Struk"].nunique()
total_items = df[QTY_COLUMN].sum() if QTY_COLUMN is not None else len(df)

st.markdown(
    f"""
    <div class="ops-hero">
        <div class="ops-kicker">Ayam Serayu restaurant demand cockpit</div>
        <div class="main-title">Kitchen Demand Forecasting</div>
        <div class="subtitle">
            Sales forecast, combo planning, and ingredient restocking in one operations view.
        </div>
        <div class="service-strip">
            <div class="service-chip"><span>Data Window</span><strong>{safe_html(data_window)}</strong></div>
            <div class="service-chip"><span>Outlets</span><strong>{df["Outlet"].nunique()}</strong></div>
            <div class="service-chip"><span>Menu Items</span><strong>{df["Nama Produk"].nunique()}</strong></div>
            <div class="service-chip"><span>Items Sold</span><strong>{safe_html(compact_number(total_items))}</strong></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# SIDEBAR NAVIGATION
# =========================
page_labels = {
    "Dashboard": "Demand Kitchen",
    "AI Manager Report": "AI Manager Brief",
    "Model Performance": "Model Performance",
    "Combo Control": "Combo Counter",
    "AI Restocking Demo": "Stock Planner",
    "Alert Center": "Alert Center"
}

if "page_nav" not in st.session_state:
    st.session_state["page_nav"] = "Dashboard"

LANDING_PAGE_URL = "http://127.0.0.1:5173/"

st.sidebar.link_button(
    "Home",
    LANDING_PAGE_URL,
    icon=":material/home:",
    width="stretch"
)
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "AI Manager Report",
        "Model Performance",
        "Combo Control",
        "AI Restocking Demo",
        "Alert Center"
    ],
    key="page_nav",
    format_func=lambda value: page_labels[value]
)

st.sidebar.caption(f"Current Page: {page_labels[page]}")

st.sidebar.divider()
st.sidebar.caption("Currency Mode")

currency_mode = st.sidebar.radio(
    "Choose Currency",
    ["MYR", "IDR"],
    horizontal=True
)

st.session_state["currency_mode"] = currency_mode

if currency_mode == "MYR":
    st.sidebar.success("Showing values in MYR")
    st.sidebar.write(f"Rate used: {IDR_TO_MYR}")
else:
    st.sidebar.info("Showing values in Rupiah")

# =========================
# DASHBOARD PAGE
# =========================
if page == "Dashboard":

    daily_history = get_daily_sales_history(
        len(df),
        str(DATA_MIN_DATE.date()),
        str(DATA_MAX_DATE.date())
    )
    top_outlet = df.groupby("Outlet")["Total"].sum().idxmax()
    top_product = (
        df.groupby("Nama Produk")[QTY_COLUMN].sum().idxmax()
        if QTY_COLUMN is not None
        else df["Nama Produk"].value_counts().idxmax()
    )
    avg_daily_sales = daily_history.mean()

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        render_kpi_card(
            "Orders Served",
            compact_number(total_orders),
            "Unique receipts in dataset",
            "fire"
        )

    with k2:
        render_kpi_card(
            "Avg Daily Demand",
            compact_currency(avg_daily_sales),
            "Across all outlets",
            "mint"
        )

    with k3:
        render_kpi_card(
            "Busiest Outlet",
            top_outlet.replace("AYAM SERAYU - ", ""),
            "Highest historical sales",
            "amber"
        )

    with k4:
        render_kpi_card(
            "Top Menu Pull",
            top_product,
            "By item quantity sold",
            "steel"
        )

    st.divider()
    st.markdown('<div class="section-label">Service Forecast Control</div>', unsafe_allow_html=True)

    outlets = sorted(df["Outlet"].dropna().unique())
    selected_outlet = st.selectbox("Select Outlet", outlets)

    prediction_options = st.multiselect(
        "Prediction Type",
        ["Daily Sales", "Monthly Sales"],
        default=["Daily Sales"]
    )

    st.divider()
    st.markdown('<div class="section-label">Promo Counter</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        discount_rate = st.slider("Discount (%)", 0, 50, 10)

    with c2:
        uplift_multiplier = st.slider("Uplift Multiplier", 0.1, 2.0, 0.6, 0.1)

    with c3:
        mba_algorithm = st.selectbox(
            "MBA Algorithm",
            ["FP-Growth", "Apriori"],
            key="dashboard_mba_algorithm"
        )

    with c4:
        top_combo_n = st.slider("No. of Combo Suggestions", 3, 10, 5)

    with st.expander("Association Rule Settings"):
        r1, r2, r3 = st.columns(3)

        with r1:
            min_support_percent = st.slider(
                "Minimum Support (%)",
                0.1,
                10.0,
                1.0,
                0.1,
                key="dashboard_min_support"
            )

        with r2:
            min_confidence_percent = st.slider(
                "Minimum Confidence (%)",
                1.0,
                80.0,
                5.0,
                1.0,
                key="dashboard_min_confidence"
            )

        with r3:
            min_lift = st.slider(
                "Minimum Lift",
                0.5,
                5.0,
                0.8,
                0.1,
                key="dashboard_min_lift"
            )

    c4, c5 = st.columns(2)

    with c4:
        categories = sorted(df["Kategori"].dropna().unique())
        selected_category = st.selectbox("Category", categories)

    with c5:
        products = sorted(
            df[df["Kategori"] == selected_category]["Nama Produk"].dropna().unique()
        )
        selected_product = st.selectbox("Promo Product", products)

    combo_items = get_product_combo(
        selected_product,
        top_combo_n,
        mba_algorithm,
        min_support_percent / 100,
        min_confidence_percent / 100,
        min_lift
    )

    st.markdown(f"""
    <div class="forecast-card accent-amber">
        <div class="badge">Market Basket</div>
        <div class="forecast-title">Combo pairing for {safe_html(selected_product)}</div>
        <p class="forecast-desc">{safe_html(mba_algorithm)} association rules ranked by support, confidence, and lift.</p>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(combo_items, width="stretch")

    if combo_items.empty:
        st.warning("No combo rules found. Lower support/confidence/lift thresholds.")
        selected_combo = "No combo found"
    else:
        if combo_items["Lift"].max() < 1:
            st.info("Lift is below 1 for the current product, so these are the strongest available pairings but not strong positive associations.")
        selected_combo = st.selectbox(
            "Choose Combo Item",
            combo_items["Recommended Combo Item"].tolist()
        )

    preview_start = max(pd.Timestamp(datetime.today()).normalize(), DATA_MAX_DATE + pd.Timedelta(days=1))
    preview_dates = pd.date_range(preview_start, periods=14, freq="D")
    preview_total_sales = predict_total_sales_for_dates(preview_dates)
    preview_outlet_sales = [
        get_outlet_prediction(value, selected_outlet)
        for value in preview_total_sales
    ]
    preview_df = pd.DataFrame({
        "Date": preview_dates,
        "Forecast": preview_outlet_sales
    })
    peak_row = preview_df.loc[preview_df["Forecast"].idxmax()]
    calm_row = preview_df.loc[preview_df["Forecast"].idxmin()]
    preview_total_label = compact_currency(preview_df["Forecast"].sum())
    peak_day_label = peak_row["Date"].strftime("%d %b")
    calm_day_label = calm_row["Date"].strftime("%d %b")

    st.markdown(f"""
    <div class="menu-board">
        <div class="menu-tile"><span>Next 14 Days</span><strong>{safe_html(preview_total_label)}</strong></div>
        <div class="menu-tile"><span>Peak Prep Day</span><strong>{safe_html(peak_day_label)}</strong></div>
        <div class="menu-tile"><span>Lightest Day</span><strong>{safe_html(calm_day_label)}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(
        preview_df["Date"],
        preview_df["Forecast"].apply(convert_idr_to_myr),
        color="#ff5a3d",
        marker="o",
        linewidth=2.4,
        label="Outlet forecast"
    )
    ax.fill_between(
        preview_df["Date"],
        preview_df["Forecast"].apply(convert_idr_to_myr),
        color="#ff5a3d",
        alpha=0.14
    )
    ax.set_title(f"14-Day Service Demand Preview - {selected_outlet}")
    ax.set_xlabel("Service Date")
    ax.set_ylabel("Forecast (MYR)")
    style_axis(ax)
    plt.xticks(rotation=25)
    st.pyplot(fig)

    st.divider()
    st.markdown('<div class="section-label">Kitchen Forecast Modules</div>', unsafe_allow_html=True)

    daily_col, monthly_col = st.columns(2)

    with daily_col:
        st.markdown("""
        <div class="forecast-card accent-mint">
            <div class="badge">DAILY AI</div>
            <div class="forecast-title">Daily Outlet Forecast</div>
            <p class="forecast-desc">Predict daily sales with combo promotion simulation.</p>
        </div>
        """, unsafe_allow_html=True)

        if "Daily Sales" in prediction_options:
            daily_date = st.date_input("Daily Forecast Date")

            if st.button("Predict Daily"):
                total_prediction = predict_total_sales(daily_date)
                outlet_prediction = get_outlet_prediction(total_prediction, selected_outlet)
                horizon_note = forecast_horizon_note(daily_date)

                discounted_sales, final_sales = apply_discount_simulation(
                    outlet_prediction,
                    discount_rate,
                    uplift_multiplier
                )

                impact = final_sales - outlet_prediction

                st.markdown(f"""
                <div class="forecast-card accent-mint">
                    <div class="badge">RESULT</div>
                    <div class="forecast-title">{safe_html(selected_outlet)}</div>
                    <hr>
                    <p class="forecast-desc">Base Forecast</p>
                    <div class="forecast-value">{rm(outlet_prediction)}</div>
                    <br>
                    <p class="forecast-desc">Forecast After Promotion</p>
                    <div class="forecast-value">{rm(final_sales)}</div>
                    <hr>
                    <p class="forecast-desc">Forecast Horizon: <b>{safe_html(horizon_note)}</b></p>
                    <p class="forecast-desc">Promo Product: <b>{safe_html(selected_product)}</b></p>
                    <p class="forecast-desc">Combo Product: <b>{safe_html(selected_combo)}</b></p>
                    <p class="forecast-desc">Discount: <b>{discount_rate}%</b></p>
                    <p class="forecast-desc">Uplift Multiplier: <b>{uplift_multiplier}</b></p>
                </div>
                """, unsafe_allow_html=True)

                if impact > 0:
                    st.success(f"Estimated positive impact: {rm(impact)}")
                else:
                    st.warning(f"Estimated reduction: {rm(abs(impact))}")

    with monthly_col:
        st.markdown("""
        <div class="forecast-card accent-fire">
            <div class="badge">MONTHLY AI</div>
            <div class="forecast-title">Monthly Outlet Forecast</div>
            <p class="forecast-desc">Predict monthly revenue with promotion strategy.</p>
        </div>
        """, unsafe_allow_html=True)

        if "Monthly Sales" in prediction_options:
            m1, m2 = st.columns(2)

            with m1:
                selected_month = st.selectbox("Month", range(1, 13))

            with m2:
                selected_year = st.number_input("Year", 2025, 2030, 2026)

            if st.button("Predict Monthly"):
                days_in_month = calendar.monthrange(int(selected_year), int(selected_month))[1]
                dates = pd.date_range(
                    start=f"{selected_year}-{selected_month}-01",
                    periods=days_in_month,
                    freq="D"
                )
                total_predictions = predict_total_sales_for_dates(dates)

                base_predictions = []
                final_predictions = []

                for total_prediction in total_predictions:
                    outlet_prediction = get_outlet_prediction(total_prediction, selected_outlet)

                    discounted_sales, final_sales = apply_discount_simulation(
                        outlet_prediction,
                        discount_rate,
                        uplift_multiplier
                    )

                    base_predictions.append(outlet_prediction)
                    final_predictions.append(final_sales)

                monthly_df = pd.DataFrame({
                    "Date": dates,
                    "Base Forecast IDR": base_predictions,
                    "After Promotion IDR": final_predictions
                })

                monthly_df["Base Forecast MYR"] = monthly_df["Base Forecast IDR"].apply(convert_idr_to_myr)
                monthly_df["After Promotion MYR"] = monthly_df["After Promotion IDR"].apply(convert_idr_to_myr)

                total_base = monthly_df["Base Forecast IDR"].sum()
                total_final = monthly_df["After Promotion IDR"].sum()
                impact = total_final - total_base
                horizon_note = forecast_horizon_note(dates.max())

                st.markdown(f"""
                <div class="forecast-card accent-fire">
                    <div class="badge">MONTHLY RESULT</div>
                    <div class="forecast-title">{safe_html(selected_outlet)}</div>
                    <hr>
                    <p class="forecast-desc">Monthly Forecast</p>
                    <div class="forecast-value">{rm(total_base)}</div>
                    <br>
                    <p class="forecast-desc">Forecast After Promotion</p>
                    <div class="forecast-value">{rm(total_final)}</div>
                    <hr>
                    <p class="forecast-desc">Forecast Horizon: <b>{safe_html(horizon_note)}</b></p>
                    <p class="forecast-desc">Promo Product: <b>{safe_html(selected_product)}</b></p>
                    <p class="forecast-desc">Combo Product: <b>{safe_html(selected_combo)}</b></p>
                    <p class="forecast-desc">Discount: <b>{discount_rate}%</b></p>
                    <p class="forecast-desc">Uplift Multiplier: <b>{uplift_multiplier}</b></p>
                </div>
                """, unsafe_allow_html=True)

                if impact > 0:
                    st.success(f"Estimated positive impact: {rm(impact)}")
                else:
                    st.warning(f"Estimated reduction: {rm(abs(impact))}")

                display_df = monthly_df[["Date", "Base Forecast MYR", "After Promotion MYR"]].copy()
                display_df["Base Forecast MYR"] = display_df["Base Forecast MYR"].apply(format_myr)
                display_df["After Promotion MYR"] = display_df["After Promotion MYR"].apply(format_myr)

                st.dataframe(display_df, width="stretch")

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(monthly_df["Date"], monthly_df["Base Forecast MYR"], marker="o", label="Base Forecast")
                ax.plot(monthly_df["Date"], monthly_df["After Promotion MYR"], marker="o", label="After Promotion")
                ax.set_title(f"Monthly Forecast - {selected_outlet}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Sales (MYR)")
                ax.legend()
                style_axis(ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)


# =========================
# AI MANAGER REPORT PAGE
# =========================
if page == "AI Manager Report":

    st.header("AI Manager Brief")
    st.write("Daily operation brief for sales demand, staffing pressure, combo action, and procurement focus.")

    st.divider()

    outlets = sorted(df["Outlet"].dropna().unique())
    products = sorted(df["Nama Produk"].dropna().unique())
    default_product = top_product_by_quantity()
    default_product_index = products.index(default_product) if default_product in products else 0

    b1, b2, b3 = st.columns(3)

    with b1:
        brief_outlet = st.selectbox("Brief Outlet", outlets)

    with b2:
        brief_date = st.date_input("Brief Date", value=datetime.today())

    with b3:
        focus_product = st.selectbox("Focus Menu Item", products, index=default_product_index)

    combo_df = get_product_combo(focus_product, 5)
    focus_combo = combo_df["Recommended Combo Item"].iloc[0] if len(combo_df) > 0 else "No combo found"

    brief_dates = pd.date_range(pd.to_datetime(brief_date).normalize(), periods=7, freq="D")
    total_forecast = predict_total_sales_for_dates(brief_dates)
    outlet_forecast = [
        get_outlet_prediction(value, brief_outlet)
        for value in total_forecast
    ]
    brief_df = pd.DataFrame({
        "Date": brief_dates,
        "Outlet Forecast IDR": outlet_forecast
    })
    brief_df["Outlet Forecast MYR"] = brief_df["Outlet Forecast IDR"].apply(convert_idr_to_myr)

    daily_history = get_daily_sales_history(
        len(df),
        str(DATA_MIN_DATE.date()),
        str(DATA_MAX_DATE.date())
    )
    outlet_share = get_outlet_share(
        len(df),
        str(DATA_MIN_DATE.date()),
        str(DATA_MAX_DATE.date())
    ).get(brief_outlet, 0)
    outlet_baseline = daily_history.mean() * outlet_share
    today_forecast = brief_df["Outlet Forecast IDR"].iloc[0]
    peak_row = brief_df.loc[brief_df["Outlet Forecast IDR"].idxmax()]
    pressure_label, checklist = build_manager_checklist(
        today_forecast,
        outlet_baseline,
        peak_row["Date"].strftime("%d %b"),
        focus_combo,
        focus_product
    )

    _, _, brief_mape, _ = evaluate_sales_model(90)
    confidence_label = model_health_label(brief_mape)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Today Forecast", rm(today_forecast))

    with m2:
        st.metric("Prep Pressure", pressure_label)

    with m3:
        st.metric("Peak Day", peak_row["Date"].strftime("%d %b"))

    with m4:
        st.metric("Model Health", confidence_label)

    st.markdown(f"""
    <div class="forecast-card accent-mint">
        <div class="badge">AI Daily Brief</div>
        <div class="forecast-title">{safe_html(brief_outlet)}</div>
        <p class="forecast-desc">Expected demand: <b>{rm(today_forecast)}</b></p>
        <p class="forecast-desc">Recommended combo: <b>{safe_html(focus_product)} + {safe_html(focus_combo)}</b></p>
        <p class="forecast-desc">Forecast horizon: <b>{safe_html(forecast_horizon_note(brief_date))}</b></p>
        <p class="forecast-desc">Confidence signal: <b>{safe_html(confidence_label)} based on latest replay MAPE {brief_mape:.2f}%</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Manager Action Checklist")
    for item in checklist:
        st.checkbox(item, value=False)

    st.subheader("7-Day Outlet Forecast")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        brief_df["Date"],
        brief_df["Outlet Forecast MYR"],
        marker="o",
        color="#2fd6a3",
        linewidth=2.4,
        label="Outlet forecast"
    )
    ax.set_title(f"7-Day Manager Forecast - {brief_outlet}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales (MYR)")
    ax.legend()
    style_axis(ax)
    plt.xticks(rotation=25)
    st.pyplot(fig)

    brief_display = brief_df[["Date", "Outlet Forecast MYR"]].copy()
    brief_display["Outlet Forecast MYR"] = brief_display["Outlet Forecast MYR"].apply(format_myr)
    st.dataframe(brief_display, width="stretch")

    checklist_text = "\n- ".join(checklist)
    brief_message = f"""
AI Manager Brief

Outlet: {brief_outlet}
Date: {pd.to_datetime(brief_date).strftime('%d %b %Y')}
Today Forecast: {rm(today_forecast)}
Prep Pressure: {pressure_label}
Peak Day: {peak_row['Date'].strftime('%d %b %Y')}
Combo Action: {focus_product} + {focus_combo}
Model Health: {confidence_label} ({brief_mape:.2f}% replay MAPE)

Checklist:
- {checklist_text}
"""

    st.text_area("Manager Brief Message", brief_message, height=260)

    if st.button("Save Manager Brief To Alert Center"):
        add_alert_center_event(
            brief_outlet,
            "AI Manager Brief",
            "SAVED",
            "Dashboard",
            pressure_label,
            brief_message,
            [{"Action": item} for item in checklist]
        )
        st.success("Manager brief saved to Alert Center.")


# =========================
# MODEL PERFORMANCE PAGE
# =========================
if page == "Model Performance":

    st.header("Model Performance")
    st.write("Historical replay view for checking how close the sales forecasting model is to actual demand.")

    st.divider()

    evaluation_days = st.slider("Evaluation Window (days)", 30, 180, 90, 15)
    eval_df, mae, mape, r2 = evaluate_sales_model(evaluation_days)
    health = model_health_label(mape)

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.metric("Replay MAPE", f"{mape:.2f}%")

    with p2:
        st.metric("MAE", rm(mae))

    with p3:
        st.metric("R2 Score", f"{r2:.3f}")

    with p4:
        st.metric("Model Health", health)

    eval_df["Actual Sales MYR"] = eval_df["Actual Sales IDR"].apply(convert_idr_to_myr)
    eval_df["Predicted Sales MYR"] = eval_df["Predicted Sales IDR"].apply(convert_idr_to_myr)
    eval_df["Abs Error MYR"] = eval_df["Abs Error IDR"].apply(convert_idr_to_myr)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(eval_df["Date"], eval_df["Actual Sales MYR"], color="#2fd6a3", label="Actual")
    ax.plot(eval_df["Date"], eval_df["Predicted Sales MYR"], color="#ff5a3d", label="Predicted")
    ax.set_title(f"Actual vs Predicted Sales - Latest {evaluation_days} Days")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales (MYR)")
    ax.legend()
    style_axis(ax)
    plt.xticks(rotation=25)
    st.pyplot(fig)

    st.subheader("Worst Error Days")
    worst_days = eval_df.sort_values("Abs Error IDR", ascending=False).head(10)[
        ["Date", "Actual Sales MYR", "Predicted Sales MYR", "Abs Error MYR"]
    ].copy()
    for col in ["Actual Sales MYR", "Predicted Sales MYR", "Abs Error MYR"]:
        worst_days[col] = worst_days[col].apply(format_myr)
    st.dataframe(worst_days, width="stretch")

    st.info(
        "Use this page during viva to defend the model with replay metrics. "
        "For final report, mention that a true holdout split is stronger if the model is retrained."
    )


# =========================
# COMBO CONTROL PAGE
# =========================
if page == "Combo Control":

    st.header("Combo Counter")
    st.write("Build restaurant bundles from actual receipt pairing and promo assumptions.")

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Combo Setup")

        combo_categories = sorted(df["Kategori"].dropna().unique())
        combo_category = st.selectbox("Choose Product Category", combo_categories)

        combo_products = sorted(
            df[df["Kategori"] == combo_category]["Nama Produk"].dropna().unique()
        )
        main_product = st.selectbox("Main Product", combo_products)

        cb1, cb2 = st.columns(2)

        with cb1:
            combo_algorithm = st.selectbox(
                "MBA Algorithm",
                ["FP-Growth", "Apriori"],
                key="combo_mba_algorithm"
            )

        with cb2:
            top_n = st.slider("Number of AI Combo Suggestions", 3, 15, 5)

        with st.expander("Association Rule Settings"):
            ar1, ar2, ar3 = st.columns(3)

            with ar1:
                combo_min_support_percent = st.slider(
                    "Minimum Support (%)",
                    0.1,
                    10.0,
                    1.0,
                    0.1,
                    key="combo_min_support"
                )

            with ar2:
                combo_min_confidence_percent = st.slider(
                    "Minimum Confidence (%)",
                    1.0,
                    80.0,
                    5.0,
                    1.0,
                    key="combo_min_confidence"
                )

            with ar3:
                combo_min_lift = st.slider(
                    "Minimum Lift",
                    0.5,
                    5.0,
                    0.8,
                    0.1,
                    key="combo_min_lift"
                )

        ai_combo_df = get_product_combo(
            main_product,
            top_n,
            combo_algorithm,
            combo_min_support_percent / 100,
            combo_min_confidence_percent / 100,
            combo_min_lift
        )

        combo_mode = st.radio(
            "Combo Selection Mode",
            ["AI Recommended Combo", "Manual Combo"]
        )

        if combo_mode == "AI Recommended Combo" and not ai_combo_df.empty:
            combo_product = st.selectbox(
                "Choose AI Combo Product",
                ai_combo_df["Recommended Combo Item"].tolist()
            )
        else:
            if combo_mode == "AI Recommended Combo":
                st.warning("No association rules found. Manual combo selection is shown instead.")
            all_products = sorted(df["Nama Produk"].dropna().unique())
            combo_product = st.selectbox("Choose Manual Combo Product", all_products)

    with right:
        st.subheader("Promotion Control")

        discount_rate = st.slider("Discount (%)", 0, 70, 10)
        expected_uplift = st.slider("Expected Uplift (%)", 0, 100, 20)
        target_bundle_qty = st.number_input("Target Combo Quantity", min_value=1, value=100, step=10)

        main_summary = product_sales_summary(main_product)
        combo_summary = product_sales_summary(combo_product)

        avg_main_price_idr = (
            main_summary["total_sales_idr"] / main_summary["total_qty"]
            if main_summary["total_qty"] > 0 else 0
        )

        avg_combo_price_idr = (
            combo_summary["total_sales_idr"] / combo_summary["total_qty"]
            if combo_summary["total_qty"] > 0 else 0
        )

        bundle_price_idr = avg_main_price_idr + avg_combo_price_idr
        discounted_bundle_price_idr = bundle_price_idr * (1 - discount_rate / 100)
        estimated_revenue_idr = discounted_bundle_price_idr * target_bundle_qty
        estimated_revenue_after_uplift_idr = estimated_revenue_idr * (1 + expected_uplift / 100)
        discount_loss_idr = (bundle_price_idr - discounted_bundle_price_idr) * target_bundle_qty
        uplift_gain_idr = estimated_revenue_after_uplift_idr - estimated_revenue_idr
        net_impact_idr = uplift_gain_idr - discount_loss_idr

    st.divider()

    st.subheader("AI Combo Recommendation")
    st.dataframe(ai_combo_df, width="stretch")
    if not ai_combo_df.empty and ai_combo_df["Lift"].max() < 1:
        st.info("Lift is below 1 for this product, so the table shows the strongest available pairings rather than strong positive associations.")

    st.divider()

    st.subheader("Selected Combo Summary")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Main Product Avg Price", rm(avg_main_price_idr))

    with c2:
        st.metric("Combo Product Avg Price", rm(avg_combo_price_idr))

    with c3:
        st.metric("Bundle Price Before Discount", rm(bundle_price_idr))

    with c4:
        st.metric("Bundle Price After Discount", rm(discounted_bundle_price_idr))

    c5, c6, c7 = st.columns(3)

    with c5:
        st.metric("Estimated Revenue", rm(estimated_revenue_idr))

    with c6:
        st.metric("Revenue After Uplift", rm(estimated_revenue_after_uplift_idr))

    with c7:
        st.metric("Net Promo Impact", rm(net_impact_idr))

    st.divider()

    st.subheader("Combo Strategy Card")

    st.markdown(f"""
    <div class="forecast-card accent-amber">
        <div class="badge">COMBO STRATEGY</div>
        <div class="forecast-title">{safe_html(main_product)} + {safe_html(combo_product)}</div>
        <br>
        <p class="forecast-desc">Discount: <b>{discount_rate}%</b></p>
        <p class="forecast-desc">Expected Uplift: <b>{expected_uplift}%</b></p>
        <p class="forecast-desc">Target Combo Quantity: <b>{target_bundle_qty:,}</b></p>
        <p class="forecast-desc">Estimated Revenue After Uplift: <b>{rm(estimated_revenue_after_uplift_idr)}</b></p>
        <p class="forecast-desc">Net Promo Impact: <b>{rm(net_impact_idr)}</b></p>
    </div>
    """, unsafe_allow_html=True)

    if net_impact_idr > 0:
        st.success("This combo promotion is estimated to give a positive impact.")
    elif net_impact_idr < 0:
        st.warning("This combo promotion may reduce revenue. Try lowering discount or increasing target quantity/uplift.")
    else:
        st.info("This combo promotion is estimated to break even.")

    st.divider()

    st.subheader("Combo Revenue Chart")

    chart_df = pd.DataFrame({
        "Metric": [
            "Before Discount",
            "After Discount",
            "After Uplift"
        ],
        "Revenue MYR": [
            convert_idr_to_myr(bundle_price_idr * target_bundle_qty),
            convert_idr_to_myr(estimated_revenue_idr),
            convert_idr_to_myr(estimated_revenue_after_uplift_idr)
        ]
    })

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(chart_df["Metric"], chart_df["Revenue MYR"], color=["#7aa7ff", "#f2b84b", "#2fd6a3"])
    ax.set_title("Combo Promotion Revenue Simulation")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Revenue (MYR)")
    style_axis(ax)
    st.pyplot(fig)

    st.divider()

    st.subheader("Product Sales Detail")

    detail_df = pd.DataFrame({
        "Product": [main_product, combo_product],
        "Total Sales MYR": [
            main_summary["total_sales_myr"],
            combo_summary["total_sales_myr"]
        ],
        "Transactions": [
            main_summary["total_transactions"],
            combo_summary["total_transactions"]
        ],
        "Average Basket MYR": [
            main_summary["avg_basket_myr"],
            combo_summary["avg_basket_myr"]
        ]
    })

    detail_df["Total Sales MYR"] = detail_df["Total Sales MYR"].apply(format_myr)
    detail_df["Average Basket MYR"] = detail_df["Average Basket MYR"].apply(format_myr)

    st.dataframe(detail_df, width="stretch")


# =========================
# AI RESTOCKING DEMO PAGE
# =========================
if page == "AI Restocking Demo":

    st.header("Stock Planner")
    st.write(
        "This demo lets users enter current inventory stock and a simulated date. "
        "The system trains an AI monthly ingredient demand model from the transaction dataset, "
        "detects seasonal timing, suggests restocking date and quantity, then sends Telegram alerts."
    )

    st.divider()

    registered_users = load_registered_users()

    st.subheader("Telegram Registered User")

    if len(registered_users) == 0:
        st.warning("No Telegram user registered yet. Open the Telegram bot and register first.")
        st.code("/register AyamSerayu 0123456789")
        selected_company = "Not Registered"
        selected_chat_id = None
    else:
        selected_company = st.selectbox(
            "Select Registered Company",
            list(registered_users.keys())
        )
        selected_chat_id = registered_users[selected_company]["chat_id"]
        st.success(f"Telegram connected for: {selected_company}")

    st.divider()

    st.subheader("Auto Telegram Alert")

    a1, a2 = st.columns(2)

    with a1:
        auto_alert_enabled = st.checkbox(
            "Auto send when risk is detected",
            value=False
        )

    with a2:
        auto_alert_threshold = st.selectbox(
            "Auto Alert Threshold",
            list(AUTO_ALERT_THRESHOLDS.keys()),
            index=1
        )

    with st.expander("Auto alert history"):
        auto_alert_state = load_auto_alert_state()
        st.write(f"Sent scenarios stored: {len(auto_alert_state)}")
        if st.button("Reset Auto Alert History"):
            save_auto_alert_state({})
            st.success("Auto alert history reset. The next matching risk scenario can send again.")

    st.divider()

    st.subheader("1. Simulated Timeframe")

    tf1, tf2, tf3 = st.columns(3)

    with tf1:
        simulated_today = st.date_input(
            "Simulated Today",
            value=datetime.today()
        )

    with tf2:
        forecast_month = st.selectbox(
            "Forecast Month",
            list(range(1, 13)),
            index=datetime.today().month - 1
        )

    with tf3:
        forecast_year = st.number_input(
            "Forecast Year",
            min_value=2025,
            max_value=2035,
            value=2026
        )

    supplier_lead_time = st.slider(
        "Supplier Lead Time (days)",
        min_value=1,
        max_value=30,
        value=5
    )

    st.divider()

    st.subheader("2. Current Inventory Stock")

    s1, s2, s3 = st.columns(3)

    with s1:
        current_chicken = st.number_input("Chicken Stock (kg)", min_value=0.0, value=80.0, step=5.0)
        current_rice = st.number_input("Rice Stock (kg)", min_value=0.0, value=50.0, step=5.0)

    with s2:
        current_drink = st.number_input("Drink Stock (units)", min_value=0.0, value=150.0, step=10.0)
        current_chili = st.number_input("Chili Stock (kg)", min_value=0.0, value=20.0, step=2.0)

    with s3:
        current_oil = st.number_input("Oil Stock (liter)", min_value=0.0, value=30.0, step=2.0)
        current_egg = st.number_input("Egg Stock (units)", min_value=0.0, value=200.0, step=10.0)

    current_stock = {
        "Chicken_kg": current_chicken,
        "Rice_kg": current_rice,
        "Drink_units": current_drink,
        "Chili_kg": current_chili,
        "Oil_liter": current_oil,
        "Egg_units": current_egg
    }

    st.divider()

    st.subheader("3. AI Inventory Demand Model")

    inventory_model, recipe_map, monthly_usage, latest_month, feature_cols = train_inventory_model(
        len(df),
        str(df["Tanggal & Waktu"].min()),
        str(df["Tanggal & Waktu"].max())
    )

    with st.expander("View Auto Ingredient Mapping"):
        mapping_rows = []
        for product, recipe in recipe_map.items():
            row = {"Product": product}
            row.update(recipe)
            mapping_rows.append(row)
        st.dataframe(pd.DataFrame(mapping_rows), width="stretch")

    with st.expander("View Monthly Ingredient Usage Generated From Dataset"):
        st.dataframe(monthly_usage, width="stretch")

    if inventory_model is None:
        st.error("Not enough monthly data to train inventory forecasting model.")
    else:
        st.success("AI monthly inventory forecasting model trained successfully from historical transaction data.")

        seasonal_event, seasonal_multiplier, seasonal_confidence = detect_seasonal_event(
            simulated_today,
            forecast_month
        )

        predicted_demand = predict_monthly_ingredient_demand(
            inventory_model,
            monthly_usage,
            latest_month,
            feature_cols,
            int(forecast_year),
            int(forecast_month),
            seasonal_multiplier
        )

        suggested_restock_date = pd.to_datetime(simulated_today) + timedelta(days=supplier_lead_time)

        result_rows = []

        for ingredient in INGREDIENT_COLUMNS:
            restock_qty = max(
                round(predicted_demand[ingredient] - current_stock[ingredient], 2),
                0
            )

            shortage_percent = 0
            if predicted_demand[ingredient] > 0:
                shortage_percent = (restock_qty / predicted_demand[ingredient]) * 100

            if shortage_percent >= 50:
                priority = "HIGH"
            elif shortage_percent >= 20:
                priority = "MEDIUM"
            elif restock_qty > 0:
                priority = "LOW"
            else:
                priority = "SUFFICIENT"

            result_rows.append({
                "Ingredient": format_ingredient_name(ingredient),
                "Current Stock": round(current_stock[ingredient], 2),
                "AI Predicted Monthly Demand": round(predicted_demand[ingredient], 2),
                "Recommended Restock Quantity": restock_qty,
                "Priority": priority
            })

        result_df = pd.DataFrame(result_rows)

        st.divider()
        st.subheader("4. AI Restocking Recommendation")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric("Detected Seasonal Event", seasonal_event)

        with r2:
            st.metric("Seasonal Confidence", f"{seasonal_confidence}%")

        with r3:
            st.metric("Suggested Restocking Date", suggested_restock_date.strftime("%d %b %Y"))

        st.dataframe(result_df, width="stretch")

        if len(result_df[result_df["Priority"] == "HIGH"]) > 0:
            st.error("High inventory risk detected. Immediate procurement action is recommended.")
        elif len(result_df[result_df["Priority"] == "MEDIUM"]) > 0:
            st.warning("Medium inventory risk detected. Restocking should be planned soon.")
        else:
            st.success("Inventory is mostly sufficient for the predicted monthly demand.")

        st.divider()
        st.subheader("5. Demand vs Stock Chart")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(result_df["Ingredient"], result_df["AI Predicted Monthly Demand"], color="#ff5a3d", label="AI Predicted Demand")
        ax.plot(result_df["Ingredient"], result_df["Current Stock"], color="#2fd6a3", marker="o", linewidth=2.4, label="Current Stock")
        ax.set_title("Monthly Ingredient Demand vs Current Stock")
        ax.set_xlabel("Ingredient")
        ax.set_ylabel("Quantity")
        ax.legend()
        style_axis(ax)
        plt.xticks(rotation=20)
        st.pyplot(fig)

        st.divider()
        st.subheader("6. Telegram AI Restocking Alert")

        risky_items = result_df[result_df["Recommended Restock Quantity"] > 0].sort_values(
            "Recommended Restock Quantity",
            ascending=False
        )

        if len(risky_items) > 0:
            main_suggestion = f"Restock {risky_items.iloc[0]['Ingredient']} first because it has the highest shortage risk."
        else:
            main_suggestion = "No urgent restocking required for this selected timeframe."

        auto_alert_items = get_auto_alert_items(result_df, auto_alert_threshold)

        restock_lines = ""
        for _, row in result_df.iterrows():
            restock_lines += (
                f"\n{row['Ingredient']}\n"
                f"Current Stock: {row['Current Stock']}\n"
                f"AI Demand: {row['AI Predicted Monthly Demand']}\n"
                f"Recommended Restock: +{row['Recommended Restock Quantity']}\n"
                f"Priority: {row['Priority']}\n"
            )

        telegram_message = f"""
🤖 AI Seasonal Procurement Alert

Company:
{selected_company}

Simulated Today:
{pd.to_datetime(simulated_today).strftime('%d %b %Y')}

Forecast Period:
{forecast_month}/{forecast_year}

Detected Event:
{seasonal_event}

Seasonal Confidence:
{seasonal_confidence}%

Suggested Restocking Date:
{suggested_restock_date.strftime('%d %b %Y')}

AI Restocking Recommendation:
{restock_lines}

Main Suggestion:
{main_suggestion}
"""

        latest_alert = {
            "company": selected_company,
            "simulated_today": str(simulated_today),
            "forecast_month": int(forecast_month),
            "forecast_year": int(forecast_year),
            "seasonal_event": seasonal_event,
            "seasonal_confidence": seasonal_confidence,
            "suggested_restock_date": str(suggested_restock_date.date()),
            "recommendations": result_rows,
            "auto_alert_enabled": auto_alert_enabled,
            "auto_alert_threshold": auto_alert_threshold,
            "auto_alert_triggered_items": auto_alert_items.to_dict("records"),
            "main_suggestion": main_suggestion
        }

        save_latest_restock_alert(latest_alert)

        if auto_alert_enabled:
            if selected_chat_id is None:
                st.warning("Auto alert is enabled, but no Telegram user is registered yet.")
                if len(auto_alert_items) > 0:
                    auto_alert_key = build_auto_alert_key(
                        selected_company,
                        simulated_today,
                        forecast_year,
                        forecast_month,
                        auto_alert_threshold,
                        auto_alert_items
                    )
                    add_alert_center_event(
                        selected_company,
                        "Auto procurement alert pending",
                        "PENDING",
                        "Dashboard",
                        auto_alert_items["Priority"].iloc[0],
                        telegram_message,
                        auto_alert_items.to_dict("records"),
                        dedupe_key=f"pending-{auto_alert_key}"
                    )
            elif len(auto_alert_items) == 0:
                st.success("Auto alert is enabled. No item reached the selected risk threshold.")
            else:
                auto_alert_key = build_auto_alert_key(
                    selected_company,
                    simulated_today,
                    forecast_year,
                    forecast_month,
                    auto_alert_threshold,
                    auto_alert_items
                )
                auto_alert_state = load_auto_alert_state()

                if auto_alert_key in auto_alert_state:
                    previous_alert = auto_alert_state[auto_alert_key]
                    st.info(
                        "Auto alert already sent for this simulated scenario "
                        f"at {previous_alert.get('sent_at', 'previous run')}."
                    )
                else:
                    auto_telegram_message = (
                        telegram_message
                        + f"\nAuto Trigger Threshold:\n{auto_alert_threshold}\n"
                    )
                    sent = send_telegram_alert(selected_chat_id, auto_telegram_message)
                    if sent:
                        auto_alert_state[auto_alert_key] = {
                            "company": selected_company,
                            "simulated_today": str(pd.to_datetime(simulated_today).date()),
                            "forecast_month": int(forecast_month),
                            "forecast_year": int(forecast_year),
                            "threshold": auto_alert_threshold,
                            "sent_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "items": auto_alert_items.to_dict("records")
                        }
                        save_auto_alert_state(auto_alert_state)
                        add_alert_center_event(
                            selected_company,
                            "Auto procurement alert sent",
                            "SENT",
                            "Telegram",
                            auto_alert_items["Priority"].iloc[0],
                            auto_telegram_message,
                            auto_alert_items.to_dict("records"),
                            dedupe_key=f"sent-{auto_alert_key}"
                        )
                        st.success("Auto Telegram alert sent because risk threshold was reached.")
                    else:
                        add_alert_center_event(
                            selected_company,
                            "Telegram failed - fallback alert saved",
                            "FAILED",
                            "Telegram",
                            auto_alert_items["Priority"].iloc[0],
                            auto_telegram_message,
                            auto_alert_items.to_dict("records"),
                            dedupe_key=f"failed-{auto_alert_key}"
                        )
                        st.warning("Telegram failed, so the alert was saved in Alert Center.")
        else:
            st.info("Auto alert is off. Manual Telegram sending is still available.")

        st.text_area("Telegram Message Preview", telegram_message, height=350)

        if selected_chat_id is not None:
            if st.button("Send Telegram AI Restocking Alert"):
                sent = send_telegram_alert(selected_chat_id, telegram_message)
                if sent:
                    add_alert_center_event(
                        selected_company,
                        "Manual procurement alert sent",
                        "SENT",
                        "Telegram",
                        risky_items["Priority"].iloc[0] if len(risky_items) > 0 else "INFO",
                        telegram_message,
                        risky_items.to_dict("records")
                    )
                    st.success("Telegram AI restocking alert sent successfully.")
                else:
                    add_alert_center_event(
                        selected_company,
                        "Manual Telegram failed - fallback alert saved",
                        "FAILED",
                        "Telegram",
                        risky_items["Priority"].iloc[0] if len(risky_items) > 0 else "INFO",
                        telegram_message,
                        risky_items.to_dict("records")
                    )
                    st.warning("Telegram failed, so the manual alert was saved in Alert Center.")
        else:
            st.info("Register a Telegram user first before sending alert.")


# =========================
# ALERT CENTER PAGE
# =========================
if page == "Alert Center":

    st.header("Alert Center")
    st.write("Fallback inbox for procurement alerts, Telegram failures, saved manager briefs, and resend actions.")

    st.divider()

    alerts = load_alert_center()

    if len(alerts) == 0:
        st.success("No alerts stored yet.")
    else:
        alert_df = pd.DataFrame(alerts)

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.metric("Total Alerts", len(alert_df))

        with s2:
            st.metric("Pending", len(alert_df[alert_df["status"] == "PENDING"]))

        with s3:
            st.metric("Failed", len(alert_df[alert_df["status"] == "FAILED"]))

        with s4:
            st.metric("Sent/Saved", len(alert_df[alert_df["status"].isin(["SENT", "SAVED", "RESENT"])]))

        status_options = ["All"] + sorted(alert_df["status"].dropna().unique().tolist())
        selected_status = st.selectbox("Filter Status", status_options)

        display_alerts = alert_df.copy()
        if selected_status != "All":
            display_alerts = display_alerts[display_alerts["status"] == selected_status]

        st.dataframe(
            display_alerts[
                ["created_at", "company", "title", "status", "channel", "severity", "id"]
            ],
            width="stretch"
        )

        alert_options = [
            f"{alert['id']} | {alert['status']} | {alert['title']}"
            for alert in alerts
        ]
        selected_alert_label = st.selectbox("Open Alert", alert_options)
        selected_alert_id = selected_alert_label.split(" | ")[0]
        selected_alert = next(alert for alert in alerts if alert["id"] == selected_alert_id)

        st.markdown(f"""
        <div class="forecast-card accent-steel">
            <div class="badge">{safe_html(selected_alert.get('status', 'ALERT'))}</div>
            <div class="forecast-title">{safe_html(selected_alert.get('title', 'Alert'))}</div>
            <p class="forecast-desc">Company: <b>{safe_html(selected_alert.get('company', '-'))}</b></p>
            <p class="forecast-desc">Channel: <b>{safe_html(selected_alert.get('channel', '-'))}</b></p>
            <p class="forecast-desc">Severity: <b>{safe_html(selected_alert.get('severity', '-'))}</b></p>
            <p class="forecast-desc">Created: <b>{safe_html(selected_alert.get('created_at', '-'))}</b></p>
        </div>
        """, unsafe_allow_html=True)

        st.text_area("Alert Message", selected_alert.get("message", ""), height=320)

        selected_items = selected_alert.get("items", [])
        if len(selected_items) > 0:
            st.subheader("Alert Items")
            st.dataframe(pd.DataFrame(selected_items), width="stretch")

        st.divider()
        st.subheader("Fallback Actions")

        registered_users = load_registered_users()
        resendable = selected_alert.get("status") in ["FAILED", "PENDING"]

        if resendable:
            if len(registered_users) == 0:
                st.warning("No Telegram user registered. The alert remains available inside the dashboard.")
            else:
                default_company = selected_alert.get("company")
                user_names = list(registered_users.keys())
                default_index = user_names.index(default_company) if default_company in user_names else 0
                resend_company = st.selectbox("Resend To", user_names, index=default_index)

                if st.button("Resend Selected Alert To Telegram"):
                    sent = send_telegram_alert(
                        registered_users[resend_company]["chat_id"],
                        selected_alert.get("message", "")
                    )
                    if sent:
                        for alert in alerts:
                            if alert["id"] == selected_alert_id:
                                alert["status"] = "RESENT"
                                alert["channel"] = "Telegram"
                                alert["resent_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                alert["company"] = resend_company
                                break
                        save_alert_center(alerts)
                        st.success("Alert resent successfully.")
                    else:
                        st.error("Telegram resend failed. Alert is still saved in Alert Center.")
        else:
            st.info("This alert does not need a resend action.")

        if st.button("Clear Alert Center"):
            save_alert_center([])
            st.success("Alert Center cleared.")
