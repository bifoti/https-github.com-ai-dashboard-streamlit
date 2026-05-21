import telebot
import json
import os

# =========================
# TELEGRAM BOT CONFIG
# =========================
# Get token from @BotFather
BOT_TOKEN = "YOUR_BOT_TOKEN"
BOT_STATE_FILE = "telegram_bot_state.json"

bot = telebot.TeleBot(BOT_TOKEN)


def load_latest_state():
    if not os.path.exists(BOT_STATE_FILE):
        return None

    with open(BOT_STATE_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(
        message,
        """
🤖 Welcome to AI Restaurant Assistant Bot

This bot is connected to your Streamlit dashboard.

Commands:
/status - View latest promotion status
/suggest - Get latest AI suggestion
/topcombo - View top combo recommendation
/help - Show command list
"""
    )


@bot.message_handler(commands=["help"])
def help_command(message):
    bot.reply_to(
        message,
        """
Available Commands:

/status
Show latest combo promotion result.

/suggest
Show AI suggestion based on the latest Streamlit analysis.

/topcombo
Show latest AI combo recommendations.
"""
    )


@bot.message_handler(commands=["status"])
def status(message):
    state = load_latest_state()

    if state is None:
        bot.reply_to(
            message,
            "No Streamlit data found yet. Please open the Streamlit app and use Combo Control first."
        )
        return

    reply = f"""
📊 Latest Promo Status

Status:
{state.get("promo_status")}

Main Product:
{state.get("main_product")}

Combo Product:
{state.get("combo_product")}

Discount:
{state.get("discount_rate")}%

Expected Uplift:
{state.get("expected_uplift")}%

Target Quantity:
{state.get("target_bundle_qty")}

Estimated Revenue:
{state.get("estimated_revenue")}

Revenue After Uplift:
{state.get("revenue_after_uplift")}

Net Promo Impact:
{state.get("net_promo_impact")}
"""
    bot.reply_to(message, reply)


@bot.message_handler(commands=["suggest"])
def suggest(message):
    state = load_latest_state()

    if state is None:
        bot.reply_to(
            message,
            "No AI suggestion found yet. Please use Combo Control in Streamlit first."
        )
        return

    reply = f"""
🤖 AI Business Suggestion

Recommended Combo:
{state.get("main_product")} + {state.get("combo_product")}

Status:
{state.get("promo_status")}

Suggestion:
{state.get("ai_suggestion")}
"""
    bot.reply_to(message, reply)


@bot.message_handler(commands=["topcombo"])
def topcombo(message):
    state = load_latest_state()

    if state is None:
        bot.reply_to(
            message,
            "No combo data found yet. Please use Combo Control in Streamlit first."
        )
        return

    combos = state.get("top_combos", [])

    if not combos:
        bot.reply_to(message, "No combo recommendation available.")
        return

    reply = "🛒 Latest AI Combo Recommendations\n\n"

    for index, item in enumerate(combos, start=1):
        product = item.get("Recommended Combo Item", "Unknown")
        frequency = item.get("Frequency", 0)
        reply += f"{index}. {product} - Frequency: {frequency}\n"

    bot.reply_to(message, reply)


print("Telegram bot is running...")
bot.infinity_polling()
