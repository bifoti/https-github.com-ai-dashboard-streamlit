import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime, timedelta
import telebot
from sklearn.ensemble import RandomForestRegressor

# =========================
# LOAD DATA
# =========================
model = joblib.load("sales_forecasting_model.pkl")
latest_values = joblib.load("latest_values.pkl")

df = pd.read_csv("AyamSerayu_3Years_Transaction_Data.csv")
df["Tanggal & Waktu"] = pd.to_datetime(df["Tanggal & Waktu"])


# =========================
# TELEGRAM CONFIG
# =========================
# Replace this token with your real token from @BotFather.
# The bot file will register users and save their chat_id into users.json.
BOT_TOKEN = "8876275131:AAFqLliTehn630SesjjHaV9J4f4K18EGkC0"
USERS_FILE = "users.json"
LATEST_RESTOCK_FILE = "latest_restocking_alert.json"

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

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Restaurant Analytics",
    page_icon="🤖",
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

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #070b14;
    color: white;
    font-family: 'Segoe UI';
}
.main-title {
    font-size: 50px;
    font-weight: 900;
    background: linear-gradient(90deg,#00F5FF,#7B61FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    color: #8b98b5;
    font-size: 18px;
    margin-bottom: 20px;
}
.kpi-card, .forecast-card {
    background: linear-gradient(145deg, rgba(0,245,255,0.10), rgba(123,97,255,0.08));
    border: 1px solid rgba(0,245,255,0.25);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 0 25px rgba(0,245,255,0.10);
}
.kpi-title {
    color: #94a3b8;
    font-size: 14px;
}
.kpi-value {
    color: #00F5FF;
    font-size: 36px;
    font-weight: 900;
}
.forecast-title {
    font-size: 28px;
    font-weight: 800;
    color: white;
}
.forecast-desc {
    color: #94a3b8;
    font-size: 14px;
}
.forecast-value {
    color: #00F5FF;
    font-size: 42px;
    font-weight: 900;
}
.badge {
    display: inline-block;
    background: rgba(0,245,255,0.15);
    color: #00F5FF;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 15px;
}
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg,#00F5FF,#7B61FF);
    color: white;
    border: none;
    border-radius: 12px;
    height: 50px;
    font-weight: bold;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# FUNCTIONS
# =========================
def create_input(date):
    date = pd.to_datetime(date)
    return pd.DataFrame({
        "month": [date.month],
        "day": [date.day],
        "dayofweek": [date.dayofweek],
        "weekofyear": [date.isocalendar().week],
        "quarter": [date.quarter],
        "lag_1": [latest_values["last_lag_1"]],
        "lag_7": [latest_values["last_lag_7"]],
        "rolling_mean_7": [latest_values["rolling_mean_7"]],
        "rolling_std_7": [latest_values["rolling_std_7"]]
    })


def predict_total_sales(date):
    input_data = create_input(date)
    return model.predict(input_data)[0]


def get_outlet_prediction(total_prediction, outlet):
    outlet_share = df.groupby("Outlet")["Total"].sum() / df["Total"].sum()
    return total_prediction * outlet_share[outlet]


def get_product_combo(product_name, top_n=5):
    receipt_with_product = df[df["Nama Produk"] == product_name]["ID Struk"].unique()

    combo_df = df[
        (df["ID Struk"].isin(receipt_with_product)) &
        (df["Nama Produk"] != product_name)
    ]

    combo_result = (
        combo_df["Nama Produk"]
        .value_counts()
        .head(top_n)
        .reset_index()
    )

    combo_result.columns = ["Recommended Combo Item", "Frequency"]
    return combo_result


def apply_discount_simulation(base_sales, discount_rate, uplift_multiplier):
    estimated_uplift = discount_rate * uplift_multiplier
    discounted_sales = base_sales * (1 - discount_rate / 100)
    final_sales = discounted_sales * (1 + estimated_uplift / 100)
    return discounted_sales, final_sales


def product_sales_summary(product_name):
    product_df = df[df["Nama Produk"] == product_name]
    total_sales_idr = product_df["Total"].sum()
    total_sales_myr = convert_idr_to_myr(total_sales_idr)
    total_qty = product_df["Jumlah"].sum() if "Jumlah" in product_df.columns else len(product_df)
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

    if "Jumlah" not in working_df.columns:
        working_df["Jumlah"] = 1

    for ingredient in INGREDIENT_COLUMNS:
        working_df[ingredient] = working_df.apply(
            lambda row: recipe_map.get(row["Nama Produk"], {}).get(ingredient, 0) * row["Jumlah"],
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
        lambda x: __import__("math").sin(2 * __import__("math").pi * x / 12)
    )
    monthly_df["month_cos"] = monthly_df["month"].apply(
        lambda x: __import__("math").cos(2 * __import__("math").pi * x / 12)
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


def predict_monthly_ingredient_demand(inventory_model, latest_month, feature_cols, forecast_year, forecast_month, seasonal_multiplier):
    import math

    input_data = {
        "year": [forecast_year],
        "month": [forecast_month],
        "month_sin": [math.sin(2 * math.pi * forecast_month / 12)],
        "month_cos": [math.cos(2 * math.pi * forecast_month / 12)]
    }

    for ingredient in INGREDIENT_COLUMNS:
        latest_value = latest_month[ingredient]
        input_data[f"lag_1_{ingredient}"] = [latest_value]
        input_data[f"rolling_mean_3_{ingredient}"] = [latest_value]

    X_future = pd.DataFrame(input_data)[feature_cols]
    prediction = inventory_model.predict(X_future)[0]

    result = {}
    for ingredient, value in zip(INGREDIENT_COLUMNS, prediction):
        result[ingredient] = max(round(value * seasonal_multiplier, 2), 0)

    return result

# =========================
# HEADER
# =========================
st.markdown(
    '<div class="main-title">AI Restaurant Analytics Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Forecasting + Market Basket Analysis + Combo Control</div>',
    unsafe_allow_html=True
)

# =========================
# SIDEBAR NAVIGATION
# =========================
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Combo Control", "AI Restocking Demo"]
)

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

    k1, k2, k3 = st.columns(3)

    with k1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Total Transactions</div>
            <div class="kpi-value">626K+</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Model Accuracy</div>
            <div class="kpi-value">86.4%</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">AI Engine</div>
            <div class="kpi-value">MBA + ML</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Prediction Control Panel")

    outlets = sorted(df["Outlet"].dropna().unique())
    selected_outlet = st.selectbox("Select Outlet", outlets)

    prediction_options = st.multiselect(
        "Prediction Type",
        ["Daily Sales", "Monthly Sales"],
        default=["Daily Sales"]
    )

    st.divider()
    st.subheader("Promotion & Combo Simulator")

    c1, c2, c3 = st.columns(3)

    with c1:
        discount_rate = st.slider("Discount (%)", 0, 50, 10)

    with c2:
        uplift_multiplier = st.slider("Uplift Multiplier", 0.1, 2.0, 0.6, 0.1)

    with c3:
        top_combo_n = st.slider("No. of Combo Suggestions", 3, 10, 5)

    c4, c5 = st.columns(2)

    with c4:
        categories = sorted(df["Kategori"].dropna().unique())
        selected_category = st.selectbox("Category", categories)

    with c5:
        products = sorted(
            df[df["Kategori"] == selected_category]["Nama Produk"].dropna().unique()
        )
        selected_product = st.selectbox("Promo Product", products)

    combo_items = get_product_combo(selected_product, top_combo_n)

    st.markdown("""
    <div class="forecast-card">
        <div class="badge">MARKET BASKET ANALYSIS</div>
        <h3>Recommended Combo Products</h3>
        <p class="forecast-desc">AI detected products frequently purchased together.</p>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(combo_items, use_container_width=True)

    selected_combo = st.selectbox(
        "Choose Combo Item",
        combo_items["Recommended Combo Item"].tolist()
    )

    st.divider()
    st.header("Forecast Modules")

    daily_col, monthly_col = st.columns(2)

    with daily_col:
        st.markdown("""
        <div class="forecast-card">
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

                discounted_sales, final_sales = apply_discount_simulation(
                    outlet_prediction,
                    discount_rate,
                    uplift_multiplier
                )

                impact = final_sales - outlet_prediction

                st.markdown(f"""
                <div class="forecast-card">
                    <div class="badge">RESULT</div>
                    <div class="forecast-title">{selected_outlet}</div>
                    <hr>
                    <p class="forecast-desc">Base Forecast</p>
                    <div class="forecast-value">{rm(outlet_prediction)}</div>
                    <br>
                    <p class="forecast-desc">Forecast After Promotion</p>
                    <div class="forecast-value">{rm(final_sales)}</div>
                    <hr>
                    <p class="forecast-desc">Promo Product: <b>{selected_product}</b></p>
                    <p class="forecast-desc">Combo Product: <b>{selected_combo}</b></p>
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
        <div class="forecast-card">
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
                dates = pd.date_range(
                    start=f"{selected_year}-{selected_month}-01",
                    periods=31,
                    freq="D"
                )
                dates = dates[dates.month == selected_month]

                base_predictions = []
                final_predictions = []

                for d in dates:
                    total_prediction = predict_total_sales(d)
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

                st.markdown(f"""
                <div class="forecast-card">
                    <div class="badge">MONTHLY RESULT</div>
                    <div class="forecast-title">{selected_outlet}</div>
                    <hr>
                    <p class="forecast-desc">Monthly Forecast</p>
                    <div class="forecast-value">{rm(total_base)}</div>
                    <br>
                    <p class="forecast-desc">Forecast After Promotion</p>
                    <div class="forecast-value">{rm(total_final)}</div>
                    <hr>
                    <p class="forecast-desc">Promo Product: <b>{selected_product}</b></p>
                    <p class="forecast-desc">Combo Product: <b>{selected_combo}</b></p>
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

                st.dataframe(display_df, use_container_width=True)

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(monthly_df["Date"], monthly_df["Base Forecast MYR"], marker="o", label="Base Forecast")
                ax.plot(monthly_df["Date"], monthly_df["After Promotion MYR"], marker="o", label="After Promotion")
                ax.set_title(f"Monthly Forecast - {selected_outlet}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Sales (MYR)")
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)

# =========================
# COMBO CONTROL PAGE
# =========================
if page == "Combo Control":

    st.header("🛒 Combo Control Page")
    st.write("Control promo product, combo item, discount, quantity, and expected uplift.")

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

        top_n = st.slider("Number of AI Combo Suggestions", 3, 15, 5)
        ai_combo_df = get_product_combo(main_product, top_n)

        combo_mode = st.radio(
            "Combo Selection Mode",
            ["AI Recommended Combo", "Manual Combo"]
        )

        if combo_mode == "AI Recommended Combo":
            combo_product = st.selectbox(
                "Choose AI Combo Product",
                ai_combo_df["Recommended Combo Item"].tolist()
            )
        else:
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
    st.dataframe(ai_combo_df, use_container_width=True)

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
    <div class="forecast-card">
        <div class="badge">COMBO STRATEGY</div>
        <div class="forecast-title">{main_product} + {combo_product}</div>
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
    ax.bar(chart_df["Metric"], chart_df["Revenue MYR"])
    ax.set_title("Combo Promotion Revenue Simulation")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Revenue (MYR)")
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

    st.dataframe(detail_df, use_container_width=True)


# =========================
# AI RESTOCKING DEMO PAGE
# =========================
if page == "AI Restocking Demo":

    st.header("📦 AI Monthly Restocking Demo")
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
        st.dataframe(pd.DataFrame(mapping_rows), use_container_width=True)

    with st.expander("View Monthly Ingredient Usage Generated From Dataset"):
        st.dataframe(monthly_usage, use_container_width=True)

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

        st.dataframe(result_df, use_container_width=True)

        if len(result_df[result_df["Priority"] == "HIGH"]) > 0:
            st.error("High inventory risk detected. Immediate procurement action is recommended.")
        elif len(result_df[result_df["Priority"] == "MEDIUM"]) > 0:
            st.warning("Medium inventory risk detected. Restocking should be planned soon.")
        else:
            st.success("Inventory is mostly sufficient for the predicted monthly demand.")

        st.divider()
        st.subheader("5. Demand vs Stock Chart")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(result_df["Ingredient"], result_df["AI Predicted Monthly Demand"], label="AI Predicted Demand")
        ax.plot(result_df["Ingredient"], result_df["Current Stock"], marker="o", label="Current Stock")
        ax.set_title("Monthly Ingredient Demand vs Current Stock")
        ax.set_xlabel("Ingredient")
        ax.set_ylabel("Quantity")
        ax.legend()
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
            "main_suggestion": main_suggestion
        }

        save_latest_restock_alert(latest_alert)

        st.text_area("Telegram Message Preview", telegram_message, height=350)

        if selected_chat_id is not None:
            if st.button("Send Telegram AI Restocking Alert"):
                sent = send_telegram_alert(selected_chat_id, telegram_message)
                if sent:
                    st.success("Telegram AI restocking alert sent successfully.")
        else:
            st.info("Register a Telegram user first before sending alert.")
