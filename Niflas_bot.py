import os
import logging
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    ConversationHandler,
    filters,
)
import pandas as pd
import numpy as np
import requests

# --- Load environment variables ---
load_dotenv()
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
twelve_api_key = os.getenv("TWELVE_DATA_API_KEY")

# --- Enable logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Conversation states ---
SELECT_PAIR, SELECT_EXPIRY = range(2)

# --- RSI CALCULATION ---
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# --- FETCH DATA FROM TWELVE DATA ---
def get_market_data(symbol="EUR/USD"):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&outputsize=100&apikey={twelve_api_key}"
    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        raise ValueError(f"No data returned for {symbol}. Check symbol or API key.")

    df = pd.DataFrame(data["values"])
    df = df.rename(columns={
        "datetime": "time",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close"
    }).astype({"open": float, "high": float, "low": float, "close": float})
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)

    # Indicators
    df['sma_fast'] = df['close'].rolling(window=10).mean()
    df['sma_slow'] = df['close'].rolling(window=50).mean()
    df['rsi'] = compute_rsi(df['close'])

    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()

    df['sma20'] = df['close'].rolling(window=20).mean()
    df['stddev'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['sma20'] + 2 * df['stddev']
    df['lower_band'] = df['sma20'] - 2 * df['stddev']

    df['high_diff'] = df['high'].diff()
    df['low_diff'] = -df['low'].diff()
    df['plus_dm'] = np.where((df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0), df['high_diff'], 0)
    df['minus_dm'] = np.where((df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0), df['low_diff'], 0)

    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df['atr'] = df['tr'].rolling(window=14).mean()
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=14).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=14).mean() / df['atr'])
    df['dx'] = (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])) * 100
    df['adx'] = df['dx'].rolling(window=14).mean()

    return df

# --- /start COMMAND ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome! Use /signal to get started.")

# --- /signal COMMAND ---
async def signal_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("EURUSD", callback_data="EURUSD"),
         InlineKeyboardButton("USDJPY", callback_data="USDJPY")],
        [InlineKeyboardButton("GBPUSD", callback_data="GBPUSD"),
         InlineKeyboardButton("USDZAR", callback_data="USDZAR")],
        [InlineKeyboardButton("AUDCAD", callback_data="AUDCAD"),
         InlineKeyboardButton("EURJPY", callback_data="EURJPY")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Select a currency pair:", reply_markup=reply_markup)
    return SELECT_PAIR

# --- Handle pair selection ---
async def handle_pair_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    pair = query.data
    context.user_data["pair"] = pair
    await query.edit_message_text(f"Selected pair: {pair}\nNow enter expiry time (1, 3, or 5 minutes):")
    return SELECT_EXPIRY

# --- Handle expiry time ---
async def handle_expiry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        expiry = int(update.message.text.strip())
        if expiry not in [1, 3, 5]:
            raise ValueError("Invalid expiry")

        pair = context.user_data["pair"]
        symbol = f"{pair[:3]}/{pair[3:]}"
        df = get_market_data(symbol)
        last_3 = df.tail(3)
        last = last_3.iloc[-1]

        otc_pairs = {"USDZAR", "USDTRY", "EURMXN", "USDTHB", "USDPLN"}
        is_otc = f" _(OTC Market)_" if pair in otc_pairs else ""

        message = f"*Signal for {symbol}*{is_otc}\n"
        message += f"🕒 Expiry: {expiry} minute(s)\n"
        message += f"_Last 3 candles: {', '.join(str(t) for t in last_3.index)}_\n"

        # --- Trend Detection ---
        trend_diff = last['sma_fast'] - last['sma_slow']
        trend_pct = abs(trend_diff / last['sma_slow']) * 100

        if trend_diff > 0 and trend_pct > 0.05:
            message += "\n📈 *Uptrend detected.*"
            if 30 < last['rsi'] < 50:
                message += "\nRSI suggests pullback → *Potential BUY signal.*"
            else:
                message += "\nRSI neutral, wait for better setup."
        elif trend_diff < 0 and trend_pct > 0.05:
            message += "\n📉 *Downtrend detected.*"
            if 50 < last['rsi'] < 70:
                message += "\nRSI suggests pullback → *Potential SELL signal.*"
            else:
                message += "\nRSI neutral, wait for better setup."
        else:
            message += "\n➖ *Sideways market — no strong trend.*"

        # MACD
        if last['macd'] > last['signal_line']:
            message += "\nMACD: *Bullish crossover.*"
        else:
            message += "\nMACD: *Bearish crossover.*"

        # Bollinger Bands
        if last['close'] > last['upper_band']:
            message += "\nAbove upper Bollinger Band → *Possible overbought*"
        elif last['close'] < last['lower_band']:
            message += "\nBelow lower Bollinger Band → *Possible oversold*"
        else:
            message += "\nPrice within Bollinger Bands → *Neutral*"

        # ADX
        if last['adx'] > 25:
            message += f"\nADX: *{last['adx']:.2f} → Strong trend strength*"
        else:
            message += f"\nADX: *{last['adx']:.2f} → Weak or sideways market*"

        await update.message.reply_text(message, parse_mode='Markdown')
        return ConversationHandler.END

    except Exception as e:
        logger.error(f"Error in expiry handler: {e}")
        await update.message.reply_text("⚠️ Invalid expiry time. Please enter 1, 3, or 5.")
        return ConversationHandler.END

# --- APP INIT ---
app = ApplicationBuilder().token(bot_token).build()

conv_handler = ConversationHandler(
    entry_points=[CommandHandler("signal", signal_start)],
    states={
        SELECT_PAIR: [CallbackQueryHandler(handle_pair_selection)],
        SELECT_EXPIRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_expiry)],
    },
    fallbacks=[],
)

app.add_handler(CommandHandler("start", start))
app.add_handler(conv_handler)

print("✅ Bot is running and waiting for commands...")
app.run_polling()
