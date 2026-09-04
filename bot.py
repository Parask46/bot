import discord
from discord.ext import commands
import requests
import os
from dotenv import load_dotenv

load_dotenv()  # Reads key=value pairs from .env into the environment

# --- Configuration ---
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# --- Load system prompt ---
try:
    with open("prompt.md", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read().strip()
except FileNotFoundError:
    SYSTEM_PROMPT = "You are a normal human. No emojis. Do not say you are an AI."

# --- Bot setup ---
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# --- Per-user state ---
# user_memory: {user_id: [{"role": "...", "content": "..."}, ...]}
# user_effort: {user_id: "max" | "high" | "medium" | "low"}
user_memory: dict[int, list[dict]] = {}
user_effort: dict[int, str] = {}

VALID_EFFORT_LEVELS = ["low", "medium", "high", "max"]
MAX_HISTORY_MESSAGES = 10  # Excludes the system prompt; covers 5 back-and-forth turns


@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f"Logged in as {bot.user} - Ready to chat!")


@bot.tree.command(name="effort", description="Change the bot's reasoning effort level")
async def effort(interaction: discord.Interaction, level: str):
    level = level.lower()
    if level not in VALID_EFFORT_LEVELS:
        await interaction.response.send_message(
            f"Invalid level. Choose one of: {', '.join(VALID_EFFORT_LEVELS)}",
            ephemeral=True,
        )
        return

    user_effort[interaction.user.id] = level
    await interaction.response.send_message(
        f"Reasoning effort set to **{level}**.", ephemeral=True
    )


@bot.event
async def on_message(message: discord.Message):
    # Ignore the bot's own messages
    if message.author == bot.user:
        return

    # Only respond when directly @mentioned
    if not bot.user.mentioned_in(message):
        await bot.process_commands(message)
        return

    user_id = message.author.id

    # Strip the mention ping from the incoming text
    clean_text = message.clean_content.replace(f"@{bot.user.name}", "").strip()
    if not clean_text:
        clean_text = "hey"  # Fallback for a bare mention with no text

    # Initialise this user's history if needed
    if user_id not in user_memory:
        user_memory[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Append the new user turn
    user_memory[user_id].append({"role": "user", "content": clean_text})

    # Keep history bounded: system prompt + last MAX_HISTORY_MESSAGES messages
    if len(user_memory[user_id]) > MAX_HISTORY_MESSAGES + 1:
        user_memory[user_id] = (
            [user_memory[user_id][0]] + user_memory[user_id][-MAX_HISTORY_MESSAGES:]
        )

    current_effort = user_effort.get(user_id, "max")

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
    }

    payload = {
        "model": "moonshotai/kimi-k3",
        "messages": user_memory[user_id],
        "max_tokens": 4096,
        "seed": 0,
        "stream": False,
        "temperature": 0.7,
        "reasoning_effort": current_effort,
    }

    async with message.channel.typing():
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            bot_reply = data["choices"][0]["message"]["content"]

            # Save the assistant turn so context carries forward
            user_memory[user_id].append({"role": "assistant", "content": bot_reply})

            # Discord enforces a 2000-character message limit; chunk if needed
            if len(bot_reply) > 2000:
                for i in range(0, len(bot_reply), 2000):
                    await message.reply(bot_reply[i : i + 2000])
            else:
                await message.reply(bot_reply)

        except requests.exceptions.Timeout:
            await message.reply("took too long to think, try again in a sec")
        except requests.exceptions.HTTPError as e:
            await message.reply(f"api threw an error: {e.response.status_code}")
        except Exception as e:
            await message.reply(f"something went wrong on my end. ({e})")

    await bot.process_commands(message)


bot.run(DISCORD_TOKEN)
