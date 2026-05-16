import os
import logging
import random
import re
import asyncio
import io
import torch
from datetime import datetime, timedelta

from google import genai
from google.genai import types
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    filters, 
    ContextTypes, 
    CallbackQueryHandler
)
from dotenv import load_dotenv
import aiosqlite
# from TTS.api import TTS
from f5_tts.api import F5TTS
from num2words import num2words
from cached_path import cached_path
from ruaccent import RUAccent

# --- Загрузка и конфигурация ---
load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TEXT_MODEL = "models/gemma-4-31b-it"

if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
    raise ValueError("Необходимо установить TELEGRAM_BOT_TOKEN и GEMINI_API_KEY в .env файле")

client = genai.Client(api_key=GEMINI_API_KEY)

# --- Локальная озвучка F5-TTS ---
class TTSManager:
    _instance = None
    _model = None
    _accentizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TTSManager, cls).__new__(cls)
        return cls._instance

    def get_model(self):
        if self._model is None:
            logger.info("Загрузка модели F5-TTS (Russian fine-tuned)...")
            
            # Скачиваем файлы модели и вокабуляра локально через cached_path
            ckpt_url = "hf://Misha24-10/F5-TTS_RUSSIAN/F5TTS_v1_Base_v2/model_last_inference.safetensors"
            vocab_url = "hf://Misha24-10/F5-TTS_RUSSIAN/F5TTS_v1_Base/vocab.txt"
            
            try:
                local_ckpt = str(cached_path(ckpt_url))
                local_vocab = str(cached_path(vocab_url))
                
                self._model = F5TTS(
                    model="F5TTS_v1_Base",
                    ckpt_file=local_ckpt,
                    vocab_file=local_vocab
                )
                logger.info("Модель успешно загружена из локального кеша.")
            except Exception as e:
                logger.error(f"Ошибка при загрузке модели через cached_path: {e}")
                logger.info("Пробуем загрузить стандартную модель...")
                self._model = F5TTS()
                
        return self._model

    def get_accentizer(self):
        if self._accentizer is None:
            logger.info("Загрузка RUAccent...")
            self._accentizer = RUAccent()
            self._accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True)
            logger.info("RUAccent успешно загружен.")
        return self._accentizer

    def unload_accentizer(self):
        if self._accentizer is not None:
            logger.info("Выгрузка RUAccent из памяти...")
            self._accentizer = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("RUAccent выгружен.")

tts_manager = TTSManager()

# --- Пресеты личности ---
PRESETS = {
    "assistant": {
        "name": "Ассистент 🤖",
        "prompt": "Ты — максимально вежливый и человечный Ассистент. Твой стиль общения максимально похож на человеческий. Отвечай кратко, до 3-4 предложений. Будь всегда полезным и крайне вежливым. ВАЖНО: Никогда не называй пользователя по имени в своих ответах."
    },
    "scientific": {
        "name": "Научный ассистент 🧬",
        "prompt": "Ты — Научный ассистент. Общайся, используя научные термины, и давай подробные, глубокие объяснения. Твой стиль — академический, точный и высокоинтеллектуальный. ВАЖНО: Никогда не называй пользователя по имени в своих ответах."
    }
}

TTS_MODES = {
    0: "❌ Выкл",
    1: "🎤 Только голос",
    2: "📝 Голос + Текст"
}

BOT_NAME = "Ассистент"
PRIMARY_VOICE_NAME = "Тони Старк 🦾"
PRIMARY_VOICE_FILE = "VOICES/tony_stark.mp3"

# --- База данных ---
class Database:
    def __init__(self, db_path="assistant.db"):
        self.db_path = db_path

    async def init(self):
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("PRAGMA table_info(chats)") as cursor:
                chat_columns = [row[1] for row in await cursor.fetchall()]

            if chat_columns and "voice" in chat_columns:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS chats_new (
                        chat_id INTEGER PRIMARY KEY,
                        preset TEXT DEFAULT 'assistant',
                        reply_chance REAL DEFAULT 0.3,
                        tts_mode INTEGER DEFAULT 0,
                        stable_tone INTEGER DEFAULT 0,
                        ruaccent_enabled INTEGER DEFAULT 1,
                        quality_preset TEXT DEFAULT 'standard'
                    )
                """)
                await db.execute("""
                    INSERT OR REPLACE INTO chats_new (
                        chat_id, preset, reply_chance, tts_mode, stable_tone, ruaccent_enabled, quality_preset
                    )
                    SELECT chat_id, preset, reply_chance, tts_mode, stable_tone, ruaccent_enabled, quality_preset
                    FROM chats
                """)
                await db.execute("DROP TABLE chats")
                await db.execute("ALTER TABLE chats_new RENAME TO chats")

            await db.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id INTEGER PRIMARY KEY,
                    preset TEXT DEFAULT 'assistant',
                    reply_chance REAL DEFAULT 0.3,
                    tts_mode INTEGER DEFAULT 0,
                    stable_tone INTEGER DEFAULT 0,
                    ruaccent_enabled INTEGER DEFAULT 1,
                    quality_preset TEXT DEFAULT 'standard'
                )
            """)
            
            await db.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
            await db.execute("CREATE TABLE IF NOT EXISTS user_facts (chat_id INTEGER, user_id INTEGER, username TEXT, fact TEXT, PRIMARY KEY (chat_id, user_id))")
            
            await db.execute("UPDATE chats SET preset = 'assistant'")
            
            await db.commit()

    async def get_chat_settings(self, chat_id):
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT preset, reply_chance, tts_mode, stable_tone, ruaccent_enabled, quality_preset FROM chats WHERE chat_id = ?", (chat_id,)) as cursor:
                row = await cursor.fetchone()
                if row: return {"preset": row[0], "reply_chance": row[1], "tts_mode": row[2], "stable_tone": row[3], "ruaccent_enabled": row[4], "quality_preset": row[5]}
                await db.execute("INSERT INTO chats (chat_id) VALUES (?)", (chat_id,))
                await db.commit()
                return {"preset": "assistant", "reply_chance": 0.3, "tts_mode": 0, "stable_tone": 0, "ruaccent_enabled": 1, "quality_preset": "standard"}

    async def update_chat_setting(self, chat_id, key, value):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(f"UPDATE chats SET {key} = ? WHERE chat_id = ?", (value, chat_id))
            await db.commit()

    async def add_history(self, chat_id, role, content):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("INSERT INTO history (chat_id, role, content) VALUES (?, ?, ?)", (chat_id, role, content))
            await db.execute("DELETE FROM history WHERE chat_id = ? AND id NOT IN (SELECT id FROM history WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 20)", (chat_id, chat_id))
            await db.commit()

    async def get_history(self, chat_id):
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT role, content FROM history WHERE chat_id = ? ORDER BY timestamp ASC", (chat_id,)) as cursor:
                return [{"role": row[0], "content": row[1]} for row in await cursor.fetchall()]

    async def get_user_facts(self, chat_id):
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT user_id, username, fact FROM user_facts WHERE chat_id = ?", (chat_id,)) as cursor:
                return {row[0]: {"username": row[1], "fact": row[2]} for row in await cursor.fetchall()}

    async def update_user_fact(self, chat_id, user_id, username, fact):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("INSERT INTO user_facts (chat_id, user_id, username, fact) VALUES (?, ?, ?, ?) ON CONFLICT(chat_id, user_id) DO UPDATE SET fact = fact || '; ' || excluded.fact, username = excluded.username", (chat_id, user_id, username, fact))
            await db.commit()

    async def clear_facts(self, chat_id):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM user_facts WHERE chat_id = ?", (chat_id,))
            await db.commit()

    async def clear_history(self, chat_id):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM history WHERE chat_id = ?", (chat_id,))
            await db.commit()

db = Database()
LAST_MESSAGE_TIMESTAMPS = {}

def clean_response(text):
    if not text: return ""
    text = re.sub(r'<\|think\|>.*?<\|/think\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)
    text = re.sub(r'\[SAVE:.*?\]', '', text)
    return text.strip()

async def get_system_instruction(chat_id):
    settings = await db.get_chat_settings(chat_id)
    preset_key = settings.get("preset", "assistant")
    preset_prompt = PRESETS.get(preset_key, PRESETS["assistant"])["prompt"]
    
    facts = await db.get_user_facts(chat_id)
    facts_str = "\n".join([f"- {v['username']}: {v['fact']}" for v in facts.values()])
    
    # Добавляем жесткое правило про кириллицу
    rule_cyrillic = "\nВАЖНО: Пиши ВСЕ имена пользователей, технические термины и любые английские слова ТОЛЬКО русскими буквами (кириллицей)."
    
    return f"<|think|>\n{preset_prompt}{rule_cyrillic}\nТвоя память о людях:\n{facts_str if facts_str else 'Пусто.'}\nЕсли узнал важное, пиши [SAVE: факт]. Ты {BOT_NAME}."

# --- Озвучка через F5-TTS (Локальная) ---

async def generate_voice(text, stable_tone=0, ruaccent_enabled=1, quality_preset="standard"):
    """Генерирует голос через локальный F5-TTS."""
    # Шаги генерации: стандарт (32) или высокое качество (45)
    steps = 45 if quality_preset == "high" else 32
    
    if not text or not text.strip():
        return None

    # Очистка текста
    tts_text = re.sub(r'\*.*?\*', '', text)
    
    if stable_tone:
        tts_text = tts_text.replace("!", ".").replace("?", ".")
    
    # Нормализация чисел (F5-TTS не умеет читать цифры сам)
    def replace_numbers(match):
        return num2words(match.group(), lang='ru')
    
    tts_text = re.sub(r'\d+', replace_numbers, tts_text)

    # Расстановка ударений через RUAccent
    if ruaccent_enabled:
        try:
            accentizer = tts_manager.get_accentizer()
            tts_text = accentizer.process_all(tts_text)
        except Exception as e:
            logger.error(f"RUAccent Error: {e}")
    else:
        tts_manager.unload_accentizer()

    tts_text = re.sub(r'[^\w\s\.,!?\-\+\u0400-\u04FF]', '', tts_text)
    tts_text = " ".join(tts_text.split())

    if not tts_text or len(tts_text) < 2:
        return None

    try:
        model = tts_manager.get_model()
        output_path = "output_voice.wav"
        
        def synthesize():
            model.infer(
                ref_file=PRIMARY_VOICE_FILE,
                ref_text="", # Пустая строка заставит модель саму распознать текст референса (ASR)
                gen_text=tts_text,
                file_wave=output_path,
                nfe_step=steps,
                sway_sampling_coef=-1.0
            )        
        await asyncio.to_thread(synthesize)
        
        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
                audio_data = f.read()
            if audio_data:
                return io.BytesIO(audio_data)
    except Exception as e:
        logger.error(f"F5-TTS Error: {e}")
    return None

# --- Обработчики ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await db.get_chat_settings(update.effective_chat.id)
    await update.message.reply_text(f'Здравствуйте! Я ваш {BOT_NAME}. Настроить меня можно в /settings.')

async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    settings = await db.get_chat_settings(chat_id)
    stable_status = "✅ Вкл" if settings.get("stable_tone") else "❌ Выкл"
    ruaccent_status = "✅ Вкл" if settings.get("ruaccent_enabled", 1) else "❌ Выкл"
    quality_name = "Высокое ✨" if settings.get("quality_preset") == "high" else "Стандарт ⚡"
    
    text = (f"⚙️ *Настройки {BOT_NAME}*\n\n🎭 Личность: {PRESETS.get(settings['preset'], PRESETS['assistant'])['name']}\n🎲 Шанс: {int(settings['reply_chance'] * 100)}%\n🎤 Режим: {TTS_MODES[settings['tts_mode']]}\n🗣 Голос: {PRIMARY_VOICE_NAME}\n⚖️ Стабильный тон: {stable_status}\n🅰️ Ударения: {ruaccent_status}\n💎 Качество: {quality_name}")
    
    keyboard = [
        [InlineKeyboardButton("🎭 Сменить личность", callback_data="set_preset_list")],
        [InlineKeyboardButton("🎤 Режим голоса", callback_data="cycle_tts")],
        [InlineKeyboardButton("⚖️ Стабильный тон", callback_data="toggle_stable"),
         InlineKeyboardButton("🅰️ Ударения", callback_data="toggle_ruaccent")],
        [InlineKeyboardButton("💎 Качество", callback_data="cycle_quality")],
        [InlineKeyboardButton("🧹 Сброс истории", callback_data="clear_hist"),
         InlineKeyboardButton("🧠 Сброс памяти", callback_data="clear_mem")]
    ]
    await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    chat_id = query.message.chat_id
    data = query.data
    await query.answer()

    try:
        if data == "set_preset_list":
            keyboard = [[InlineKeyboardButton(p["name"], callback_data=f"save_preset_{k}")] for k, p in PRESETS.items()]
            keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")])
            await query.edit_message_text("Выбери личность:", reply_markup=InlineKeyboardMarkup(keyboard))
        elif data.startswith("save_preset_"):
            await db.update_chat_setting(chat_id, "preset", data.replace("save_preset_", ""))
            await db.clear_history(chat_id)
            await query.edit_message_text("✅ Личность изменена, история очищена!")
        elif data == "cycle_tts":
            settings = await db.get_chat_settings(chat_id)
            new_mode = (settings["tts_mode"] + 1) % 3
            await db.update_chat_setting(chat_id, "tts_mode", new_mode)
            await query.edit_message_text(f"✅ Режим озвучки: {TTS_MODES[new_mode]}")
        elif data == "toggle_stable":
            settings = await db.get_chat_settings(chat_id)
            new_val = 1 if not settings.get("stable_tone") else 0
            await db.update_chat_setting(chat_id, "stable_tone", new_val)
            status = "включен" if new_val else "выключен"
            await query.edit_message_text(f"⚖️ Стабильный тон {status}!")
        elif data == "toggle_ruaccent":
            settings = await db.get_chat_settings(chat_id)
            new_val = 1 if not settings.get("ruaccent_enabled", 1) else 0
            await db.update_chat_setting(chat_id, "ruaccent_enabled", new_val)
            if not new_val:
                tts_manager.unload_accentizer()
            status = "включены" if new_val else "выключены"
            await query.edit_message_text(f"🅰️ Ударения {status}!")
        elif data == "cycle_quality":
            settings = await db.get_chat_settings(chat_id)
            new_val = "high" if settings.get("quality_preset") == "standard" else "standard"
            await db.update_chat_setting(chat_id, "quality_preset", new_val)
            quality_text = "Высокое ✨ (45 шагов)" if new_val == "high" else "Стандарт ⚡ (32 шага)"
            await query.edit_message_text(f"✅ Качество генерации: {quality_text}")
        elif data == "clear_hist":
            await db.clear_history(chat_id)
            await query.edit_message_text("🧹 История стерта.")
        elif data == "clear_mem":
            await db.clear_facts(chat_id)
            await query.edit_message_text("🧠 Я вас забыл...")
        elif data == "back_to_main":
            settings = await db.get_chat_settings(chat_id)
            stable_status = "✅ Вкл" if settings.get("stable_tone") else "❌ Выкл"
            ruaccent_status = "✅ Вкл" if settings.get("ruaccent_enabled", 1) else "❌ Выкл"
            quality_name = "Высокое ✨" if settings.get("quality_preset") == "high" else "Стандарт ⚡"
            
            text = (f"⚙️ *Настройки {BOT_NAME}*\n\n🎭 Личность: {PRESETS.get(settings['preset'], PRESETS['assistant'])['name']}\n🎲 Шанс: {int(settings['reply_chance'] * 100)}%\n🎤 Режим: {TTS_MODES[settings['tts_mode']]}\n🗣 Голос: {PRIMARY_VOICE_NAME}\n⚖️ Стабильный тон: {stable_status}\n🅰️ Ударения: {ruaccent_status}\n💎 Качество: {quality_name}")
            
            keyboard = [
                [InlineKeyboardButton("🎭 Сменить личность", callback_data="set_preset_list")],
                [InlineKeyboardButton("🎤 Режим голоса", callback_data="cycle_tts")],
                [InlineKeyboardButton("⚖️ Стабильный тон", callback_data="toggle_stable"),
                 InlineKeyboardButton("🅰️ Ударения", callback_data="toggle_ruaccent")],
                [InlineKeyboardButton("💎 Качество", callback_data="cycle_quality")],
                [InlineKeyboardButton("🧹 Сброс истории", callback_data="clear_hist"),
                 InlineKeyboardButton("🧠 Сброс памяти", callback_data="clear_mem")]
            ]
            await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
    except BadRequest as e:
        if "Message is not modified" not in str(e):
            logger.error(f"Button error: {e}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text: return
    chat_id = update.effective_chat.id
    
    # Транслитерируем ник пользователя, если он на английском
    username = update.effective_user.first_name or "Аноним"
    
    text = update.message.text
    settings = await db.get_chat_settings(chat_id)
    LAST_MESSAGE_TIMESTAMPS[chat_id] = datetime.now()

    should_reply = update.message.chat.type == 'private' or 'ассистент' in text.lower() or (update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id) or random.random() < settings["reply_chance"]

    if should_reply:
        for attempt in range(10):
            try:
                history_rows = await db.get_history(chat_id)
                history = [types.Content(role=h['role'], parts=[types.Part(text=h['content'])]) for h in history_rows]
                gen_config = types.GenerateContentConfig(temperature=1, system_instruction=await get_system_instruction(chat_id))
                chat = client.chats.create(model=TEXT_MODEL, config=gen_config, history=history)
                response = chat.send_message(f"{username}: {text}")
                bot_text = clean_response(response.text)

                facts = re.findall(r'\[SAVE: (.*?)\]', response.text)
                for f in facts: await db.update_user_fact(chat_id, update.effective_user.id, username, f)
                
                await db.add_history(chat_id, "user", f"{username}: {text}")
                await db.add_history(chat_id, "model", bot_text)

                audio_file = None
                if settings["tts_mode"] > 0:
                    audio_file = await generate_voice(
                        bot_text, 
                        settings.get("stable_tone", 0), 
                        settings.get("ruaccent_enabled", 1),
                        settings.get("quality_preset", "standard")
                    )

                try:
                    if audio_file:
                        audio_file.name = "voice.mp3"
                        if settings["tts_mode"] == 1: 
                            await update.message.reply_voice(voice=audio_file)
                        else: 
                            await update.message.reply_voice(voice=audio_file, caption=bot_text)
                    else:
                        await update.message.reply_text(bot_text)
                    return # Выход из функции при успехе
                except BadRequest as e:
                    if "Message to reply not found" in str(e) or "message to be replied not found" in str(e).lower():
                        logger.info(f"Сообщение в чате {chat_id} было удалено, отмена ответа.")
                        return
                    raise e
            except Exception as e:
                logger.error(f"Попытка {attempt + 1} не удалась: {e}")
                if attempt < 2:
                    await asyncio.sleep(2) # Ждем 2 секунды перед повтором
                else:
                    logger.error(f"Все 10 попыток провалены. Ошибка: {e}")
                    try:
                        await update.message.reply_text("ой, чет голова закружилась...")
                    except:
                        pass

async def say_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Озвучивает текст, переданный пользователем или из реплая."""
    text = ""
    if context.args:
        text = " ".join(context.args)
    elif update.message.reply_to_message and (update.message.reply_to_message.text or update.message.reply_to_message.caption):
        text = update.message.reply_to_message.text or update.message.reply_to_message.caption
    
    if not text:
        await update.message.reply_text("Напиши текст после команды или ответь этой командой на сообщение.")
        return

    chat_id = update.effective_chat.id
    settings = await db.get_chat_settings(chat_id)
    
    await context.bot.send_chat_action(chat_id=chat_id, action="record_voice")
    
    audio_file = await generate_voice(
        text, 
        settings.get("stable_tone", 0), 
        settings.get("ruaccent_enabled", 1),
        settings.get("quality_preset", "standard")
    )

    if audio_file:
        audio_file.name = "say.mp3"
        try:
            await update.message.reply_voice(voice=audio_file)
        except BadRequest:
            await context.bot.send_voice(chat_id=chat_id, voice=audio_file)
    else:
        await update.message.reply_text("не получилось озвучить...")

async def post_init(application: Application):
    """Инициализация после запуска бота."""
    await db.init()
    logger.info("База данных инициализирована.")

def main():
    # Создаем папку VOICES если её нет
    if not os.path.exists("VOICES"):
        os.makedirs("VOICES")

    if not os.path.exists(PRIMARY_VOICE_FILE):
        raise FileNotFoundError(f"Не найден основной голос: {PRIMARY_VOICE_FILE}")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("settings", settings_cmd))
    app.add_handler(CommandHandler("say", say_cmd))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("Бот запущен с drop_pending_updates=True")
    app.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
