import telebot
import detect_intent
import intent
import os;

intent.ingest_csv_to_chroma('./data/intent.csv')

bot = telebot.TeleBot(os.environ.get('TG_API_KEY'));

@bot.message_handler()
def echo_message(message):
    bot.reply_to(message, detect_intent.detect_intent_with_context(message.text))

bot.infinity_polling()