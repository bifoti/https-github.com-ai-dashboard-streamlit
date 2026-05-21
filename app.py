import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import telebot
import json
import os

# =========================
# LOAD DATA
# =========================
model = joblib.load("sales_forecasting_model.pkl")
latest_values = joblib.load("latest_values.pkl")

df = pd.read_csv("AyamSerayu_3Years_Transaction_Data.csv")
df["Tanggal & Waktu"] = pd.to_datetime(df["Tanggal & Waktu"])

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
# TELEGRAM BOT SETTING
# =========================
# 1. Get BOT_TOKEN from @BotFather
# 2. Get CHAT_ID from https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
BOT_TOKEN = "8876275131:AAFqLliTehn630SesjjHaV9J4f4K18EGkC0"
CHAT_ID = "7816636283"
BOT_STATE_FILE = "telegram_bot_state.json"


def get_telegram_bot():
    if BOT_TOKEN == "YOUR_BOT_TOKEN":
        return None
    return telebot.TeleBot(BOT_TOKEN)


def send_telegram_notification(message):
    bot = get_telegram_bot()

    if bot is None or CHAT_ID == "YOUR_CHAT_ID":
        st.error("Please set BOT_TOKEN and CHAT_ID first.")
        return False

    bot.send_message(CHAT_ID, message)
    return True


def save_telegram_state(data):
    with open(BOT_STATE_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

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
    ["Dashboard", "Combo Control"]
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

    # =========================
    # AI PROMOTION SUGGESTION FOR TELEGRAM
    # =========================
    if net_impact_idr > 0:
        promo_status = "PROFITABLE"
        ai_suggestion = (
            "This combo promotion is recommended because the estimated net impact is positive. "
            "The business can test this combo during peak hours or lunch/dinner promotion."
        )
    elif net_impact_idr < 0:
        promo_status = "RISKY"
        ai_suggestion = (
            "This combo promotion may reduce revenue. Try lowering the discount, increasing target quantity, "
            "or choosing another combo item with higher purchase frequency."
        )
    else:
        promo_status = "BREAK EVEN"
        ai_suggestion = (
            "This combo promotion is estimated to break even. Adjust the discount or expected uplift "
            "to improve the promotion impact."
        )

    telegram_message = f"""
🤖 AI Combo Promotion Suggestion

Status: {promo_status}

Main Product:
{main_product}

Combo Product:
{combo_product}

Discount:
{discount_rate}%

Expected Uplift:
{expected_uplift}%

Target Combo Quantity:
{target_bundle_qty:,}

Bundle Price Before Discount:
{rm(bundle_price_idr)}

Bundle Price After Discount:
{rm(discounted_bundle_price_idr)}

Estimated Revenue:
{rm(estimated_revenue_idr)}

Revenue After Uplift:
{rm(estimated_revenue_after_uplift_idr)}

Net Promo Impact:
{rm(net_impact_idr)}

AI Suggestion:
{ai_suggestion}
"""

    latest_bot_state = {
        "promo_status": promo_status,
        "main_product": main_product,
        "combo_product": combo_product,
        "discount_rate": discount_rate,
        "expected_uplift": expected_uplift,
        "target_bundle_qty": int(target_bundle_qty),
        "bundle_price_before_discount": rm(bundle_price_idr),
        "bundle_price_after_discount": rm(discounted_bundle_price_idr),
        "estimated_revenue": rm(estimated_revenue_idr),
        "revenue_after_uplift": rm(estimated_revenue_after_uplift_idr),
        "net_promo_impact": rm(net_impact_idr),
        "ai_suggestion": ai_suggestion,
        "top_combos": ai_combo_df.to_dict(orient="records")
    }

    save_telegram_state(latest_bot_state)

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

    st.divider()

    st.subheader("Telegram AI Notification")

    st.write(
        "Send the latest combo promotion analysis and AI suggestion to Telegram."
    )

    if st.button("Send AI Suggestion To Telegram"):
        sent = send_telegram_notification(telegram_message)

        if sent:
            st.success("Telegram notification sent successfully.")
