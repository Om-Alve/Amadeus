# Importing required libraries
import asyncio
import json
import os
import random
import re
import urllib.parse
import urllib.request
import discord
import praw
import requests
import youtube_dl
from bs4 import BeautifulSoup
from discord import FFmpegPCMAudio
from discord.ext import commands
from discord import Member, File
from discord.ext.commands import has_permissions, MissingPermissions
from quotes import Quotes
import random_responses
from googletrans import Translator
from discord.ui import Button, View
from AnilistPython import Anilist
from markdown import markdown
from easy_pil import Editor, load_image_async, Font
from env import *


# Discord config

intents = discord.Intents.all()
intents.message_content = True
client = discord.Client(intents=intents)

client = commands.Bot(intents=intents, command_prefix='$')
# Removing existing help command
client.remove_command('help')


titles = []
queues = []


# Reddit config
reddit = praw.Reddit(client_id=id,
                     client_secret=secret,
                     user_agent='USER_AGENT HERE')


# Queue function

def check_queue(ctx):
    if len(queues) > 1:
        source = queues[1]
        channel = client.get_channel(886964805783662613)
        client.loop.create_task(channel.send(
            f'```Now playing... \n{titles[1]}```'))
        queues.pop(0)
        titles.pop(0)
        vc = ctx.voice_client
        vc.play(source, after=lambda e: check_queue(ctx))

    else:
        queues.clear()
        titles.clear()
        channel = client.get_channel(886964805783662613)
        client.loop.create_task(channel.send(
            f'```There are no songs in the queue```'))

# Indicates whether the bot is ready or not


@ client.event
async def on_ready():
    print('I am ready!')

# Send random quote


@ client.command()
async def inspire(ctx):
    quote = Quotes().random()
    await ctx.send(quote[1] + "    -" + quote[0])

# Member join message


@ client.event
async def on_member_join(member):
    channel = client.get_channel(791294669577388073)
    background = Editor("pic1.jpg")
    profile_image = await load_image_async(str(member.avatar.url))
    profile = Editor(profile_image).resize((150, 150)).circle_image()
    poppins = Font.poppins(size=50, variant="bold")
    poppins_small = Font.poppins(size=30, variant="bold")
    background.paste(profile, (325, 90))
    background.ellipse((325, 90), 150, 150, outline="white", stroke_width=5)

    background.text(
        (400, 260), f"WELCOME TO {member.guild.name}", color="white", font=poppins, align="center")
    background.text((400, 325), f"{member.name}#{member.discriminator}",
                    color="white", font=poppins_small, align="center")
    file = File(fp=background.image_bytes, filename="welcome.jpg")
    await channel.send(f"Welcome! {member.mention} Just Joined")
    await channel.send(file=file)

# Member leave message


@ client.event
async def on_member_leave(member):
    channel = client.get_channel(791294669577388073)
    embed = discord.Embed(title="Goodbye!",
                          description=f"{member.mention} Just left the server")
    await channel.send(embed=embed)

# Simulate a coin toss


@ client.command()
async def toss(ctx):
    n = random.randint(-1, 2)
    if n == 0:
        await ctx.send('Heads!')
    else:
        await ctx.send('Tails!')

# Send a random number from given range


@ client.command()
async def roll(ctx, arg):
    try:
        n = int(arg)
        if n == 0:
            await ctx.send("0")
        elif n < 0:
            await ctx.send(random.randint(n, 0))
        else:
            await ctx.send(random.randint(0, n+1))
    except:
        await ctx.send("Please enter a valid integer!")

# Send a random compliment


@ client.command()
async def compliment(ctx):
    compliment = requests.get("https://complimentr.com/api").text
    compliment = json.loads(compliment)
    await ctx.send(compliment['compliment'])

# Send a random activity to do


@ client.command()
async def bored(ctx):
    activity = requests.get("http://www.boredapi.com/api/activity/").text
    activity = json.loads(activity)
    await ctx.send(activity['activity'])

# Send random joke


@ client.command()
async def joke(ctx):
    response = requests.get(
        "https://v2.jokeapi.dev/joke/Any?type=single").text
    response = json.loads(response)
    await ctx.send(response["joke"])

# Send random fact


@ client.command()
async def fact(ctx):
    limit = 1
    api_url = 'https://api.api-ninjas.com/v1/facts?limit={}'.format(limit)
    fact = requests.get(api_url,
                        headers={
                            'X-Api-Key': 'GA6QcTARZVBcWgVqbkSQEw==dpVA2YDfQE8VE6v1'
                        }).text
    fact = json.loads(fact)
    fact = fact[0]['fact']
    await ctx.send(fact)

# Send random meme


@ client.command()
async def meme(ctx):
    memes_submissions = reddit.subreddit('memes').new()
    post_to_pick = random.randint(1, 100)
    for i in range(0, post_to_pick):
        submission = next(x for x in memes_submissions if not x.stickied)

    await ctx.reply(submission.url)

# QR code generator


@ client.command()
async def qrcode(ctx, *, url):
    image = requests.get(
        f"http://api.qrserver.com/v1/create-qr-code/?data={url}")
    await ctx.reply(image.url)

# Translator


@ client.command()
async def translate(ctx, lang, *, text):
    translator = Translator()
    translation = translator.translate(text, dest=lang)
    await ctx.send(translation.text)


@ client.command()
async def anime(ctx, *, name):
    anilist = Anilist()
    data = anilist.get_anime(name)
    em1 = discord.Embed(title=f"{data['name_english']}({data['name_romaji']})",
                        color=discord.Color.random())
    em1.set_thumbnail(url=data['cover_image'])
    em1.add_field(name="Start time :", value=data['starting_time'])
    em1.add_field(name="End time :",
                  value=data['ending_time'], inline=False)
    em1.add_field(name="Episodes",
                  value=data['airing_episodes'], inline=False)
    em1.add_field(name="Rating", value=data['average_score'], inline=False)
    em1.add_field(name="Genres", value=data['genres'])

    page1 = Button(label="<", style=discord.ButtonStyle.primary)

    async def button_callback(interaction):
        page1.disabled = True
        page2.disabled = False
        await interaction.response.edit_message(embed=em1, view=view)
    page1.callback = button_callback

    html = markdown(data['desc'])
    text = ''.join(BeautifulSoup(html, features="lxml").findAll(text=True))
    em2 = discord.Embed(title="Description", description=text,
                        color=discord.Color.random())

    page2 = Button(label=">", style=discord.ButtonStyle.primary)

    async def button_callback(interaction):
        page2.disabled = True
        page1.disabled = False
        await interaction.response.edit_message(embed=em2, view=view)
    page2.callback = button_callback

    view = View()
    view.add_item(page1)
    view.add_item(page2)
    await ctx.send(embed=em1, view=view)

# Kick members


@client.command()
@has_permissions(kick_members=True)
async def kick(ctx, member: discord.Member, *, reason=None):
    await member.kick(reason=reason)
    await ctx.send(f"``` The user {member} has been kicked!```")


@kick.error
async def kick_error(ctx, error):
    if (isinstance(error, commands.MissingPermissions)):
        await ctx.send("```You don't have permissions to kick people!```")

# Ban members


@client.command()
@has_permissions(ban_members=True)
async def ban(ctx, member: discord.Member, *, reason=None):
    await member.ban(reason=reason)
    await ctx.send(f"``` The user {member} has been banned!```")


@ban.error
async def ban_error(ctx, error):
    if (isinstance(error, commands.MissingPermissions)):
        await ctx.send("```You don't have permissions to ban people!```")

# Avatar


@client.command()
async def avatar(ctx, *, avamember: discord.Member = None):
    if avamember == None:
        member = ctx.message.author
        userAvatar = member.avatar.url
        embed = discord.Embed(title=f"Avatar of {member}")
        embed.set_image(url=userAvatar)
        await ctx.send(embed=embed)
        return
    av = avamember.avatar.url
    embed = discord.Embed(title=f"Avatar of {avamember}")
    embed.set_image(url=av)
    await ctx.send(embed=embed)


# Join vc


@client.command(pass_context=True)
async def join(ctx):
    if ctx.voice_client:
        await ctx.send("I'm already in a voice channel")
        return
    if (ctx.author.voice):
        channel = ctx.message.author.voice.channel
        voice = await channel.connect()
    else:
        await ctx.send("You aren't connected to a voice channel!")

# Leave vc


@client.command(pass_context=True)
async def leave(ctx):
    if (ctx.voice_client):
        queues.clear()
        titles.clear()
        await ctx.guild.voice_client.disconnect()
        await ctx.send("I left the voice channel")
    else:
        await ctx.send("I am currently not in a voice channel!")

# Leave vc if alone in vc


@client.event
async def on_voice_state_update(member, before, after):
    voice_state = member.guild.voice_client
    if voice_state is None:
        # Exiting if the bot it's not connected to a voice channel
        return

    if len(voice_state.channel.members) == 1:
        await asyncio.sleep(60)
        while (len(voice_state.channel.members) != 1):
            return
        else:
            queues = []
            titles = []
            await voice_state.disconnect()

# Play song


@client.command(pass_context=True)
async def play(ctx, *, search=None):
    if not ctx.voice_client:
        await ctx.send("I'm not connected to a voice channel!")
        return
    if not search:
        await ctx.send("Please enter the name or url of the song to be played!")
        return
    FFMPEG_OPTIONS = {
        'before_options': '-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5', 'options': '-vn'}
    YDL_OPTIONS = {'format': 'bestaudio/best',
                   'outtmpl': '%(extractor)s-%(id)s-%(title)s.%(ext)s',
                   'restrictfilenames': True,
                   'noplaylist': True,
                   'nocheckcertificate': True,
                   'ignoreerrors': False,
                   'logtostderr': False,
                   'quiet': True,
                   'no_warnings': True,
                   'default_search': 'auto',
                   'source_address': '0.0.0.0'}
    vc = ctx.voice_client

    # searching for url
    query_string = urllib.parse.urlencode(
        {'search_query': search}
    )
    html_content = urllib.request.urlopen(
        'https://www.youtube.com/results?' + query_string
    )

    search_results = re.findall(
        r"watch\?v=(\S{11})", html_content.read().decode())

    url = ('https://www.youtube.com/watch?v='+search_results[0])
    # converting the text
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    guild_id = ctx.message.guild.id

    if not vc.is_playing():
        with youtube_dl.YoutubeDL(YDL_OPTIONS) as ydl:
            info = ydl.extract_info(url, download=False)
            url2 = info['formats'][0]['url']
        titles.append(soup.find("meta", itemprop="name")["content"])
        source = await discord.FFmpegOpusAudio.from_probe(url2, **FFMPEG_OPTIONS)
        queues.append(source)
        await ctx.send(
            f'```Now playing... \n{soup.find("meta", itemprop="name")["content"]}```')
        vc.play(source, after=lambda x=None: check_queue(
            ctx))
    else:
        with youtube_dl.YoutubeDL(YDL_OPTIONS) as ydl:
            info = ydl.extract_info(url, download=False)
            url2 = info['formats'][0]['url']
        source = await discord.FFmpegOpusAudio.from_probe(url2, **FFMPEG_OPTIONS)
        titles.append(soup.find("meta", itemprop="name")["content"])
        queues.append(source)
        await ctx.send(
            f'```Queued... \n{soup.find("meta", itemprop="name")["content"]}```')

# Show current queue


@client.command(pass_context=True)
async def queue(ctx):
    if not ctx.voice_client:
        await ctx.send("You aren't connected to a voice channel!")
    if titles != []:
        ln = len(titles)
        songs = "```Playing: "
        songs += f"{titles[0]}\n\n"
        songs += f"Queued songs:\n"
        for i in range(1, ln):
            songs += f"{i}.{titles[i]}\n"
        songs += "```"
        await ctx.send(songs)

    else:
        await ctx.send("There are no songs in the queue")


# Skip current song


@client.command()
async def skip(ctx):
    ctx.voice_client.pause()
    check_queue(ctx)

# Song pause


@client.command(pass_context=True)
async def pause(ctx):
    if not ctx.voice_client:
        await ctx.send("You aren't connected to a voice channel!")
    voice = discord.utils.get(client.voice_clients, guild=ctx.guild)
    if (ctx.voice_client):
        if voice.is_playing():
            voice.pause()
        else:
            await ctx.send("No audio is playing at the moment!")
    else:
        await ctx.send("I am currently not in a voice channel!")

# Song resume


@client.command(pass_context=True)
async def resume(ctx):
    if not ctx.voice_client:
        await ctx.send("You aren't connected to a voice channel!")
    voice = discord.utils.get(client.voice_clients, guild=ctx.guild)
    if (ctx.voice_client):
        if voice.is_paused():
            voice.resume()
        else:
            await ctx.send("No audio is paused at the moment.")
    else:
        await ctx.send("I am currently not in a voice channel!")

# Queue clear


@client.command()
async def clear(ctx):
    ctx.voice_client.stop()
    queues.clear()
    titles.clear()
    await ctx.send("Queue cleared")

# Chatbot functionality


@client.event
async def on_message(message):
    await client.process_commands(message)
    channel = client.get_channel(791316266111598603)
    if message.author == client.user or message.channel != channel:
        return
    msg = message.content
    response = random_responses.get_response(msg)
    if msg[0] != "$":
        await message.channel.send(response)


# Custom help command

@client.group(invoke_without_command=True)
async def help(ctx):
    # Home
    em = discord.Embed(
        title="Help", description="Use $help <command_name> to get details about a specific command", color=discord.Color.random())
    em.add_field(name="About Me",
                 value="I'm Amadeus, a bot made by Om Alve .", inline=False)
    em.add_field(name="LinkedIn",
                 value="https://www.linkedin.com/in/om-alve-1b8645252/", inline=False)
    em.add_field(
        name="Github", value="https://github.com/Om-Alve", inline=False)
    em.add_field(name="Website",
                 value="https://om-alve.github.io/", inline=False)
    # Music
    music = discord.Embed(
        title="Music", color=discord.Color.random())
    music.add_field(
        name="$join", value="Joins the vc the user is connected to.", inline=False)
    music.add_field(name="$leave", value="Leaves the vc.", inline=False)
    music.add_field(
        name="$play <keywords for searching the song or url>", value="Plays music searched by the user or queues the song if theres a song already playing.", inline=False)
    music.add_field(
        name="$pause", value="Pauses the ongoing song.", inline=False)
    music.add_field(
        name="$resume", value="Resumes the paused song.", inline=False)
    music.add_field(name="$queue", value="Shows current queue", inline=False)
    music.add_field(name="$clear", value="Clears the queue.", inline=False)

    # Fun
    fun = discord.Embed(
        title="Fun", color=discord.Color.random())
    fun.add_field(name="$inspire", value="Tells a random quote", inline=False)
    fun.add_field(
        name="$roll <range(int)>", value="Gives a random number in the given range", inline=False)
    fun.add_field(name="$toss", value="Simulates a coin toss", inline=False)
    fun.add_field(
        name="$meme", value="Gives a random meme from r/memes", inline=False)
    fun.add_field(name="$joke", value="Tells a joke", inline=False)
    fun.add_field(name="$fact", value="Tells a random fact", inline=False)
    fun.add_field(name="$compliment",
                  value="Gives you a compliment to make your day", inline=False)
    fun.add_field(
        name="$bored", value="Gives you a fun activity to do if your bored", inline=False)

    # Utility
    utility = discord.Embed(
        title="Utility", color=discord.Color.random())
    utility.add_field(
        name="$qrcode <text or link>", value="Converts given text or link to a QR code", inline=False)
    utility.add_field(name="$translate <language code> <text to translate>",
                      value="Translates given text to the language required by the user.", inline=False)
    utility.add_field(name="$anime <search keywords>",
                      value="Gives information about the searched anime", inline=False)
    utility.add_field(name="$kick <@member> <reason(optional)>",
                      value="Kicks the mentioned member", inline=False)
    utility.add_field(name="$ban <@member> <reason(optional)>",
                      value="Bans the mentioned member", inline=False)
    utility.add_field(name="$avatar <@member(optional)>",
                      value="Gets the avatar of the mentioned member", inline=False)

    # Adding Buttons

    music_button = Button(label="Music", style=discord.ButtonStyle.primary)

    async def button_callback(interaction):
        await interaction.response.edit_message(embed=music, view=view)
    music_button.callback = button_callback

    fun_button = Button(label="Fun", style=discord.ButtonStyle.primary)

    async def button_callback(interaction):
        await interaction.response.edit_message(embed=fun, view=view)
    fun_button.callback = button_callback

    utility_button = Button(label="Utility", style=discord.ButtonStyle.primary)

    async def button_callback(interaction):
        await interaction.response.edit_message(embed=utility, view=view)
    utility_button.callback = button_callback

    home_button = Button(label="Home", style=discord.ButtonStyle.red)

    async def button_callback(interaction):
        await interaction.response.edit_message(embed=em, view=view)
    home_button.callback = button_callback

    # Adding all the components to a view
    view = View()
    view.add_item(music_button)
    view.add_item(fun_button)
    view.add_item(utility_button)
    view.add_item(home_button)
    await ctx.channel.send(embed=em, view=view)

# Individual help commands

# Music


@help.command()
async def join(ctx):
    em = discord.Embed(
        title="join", description="Joins the vc the user is connected to.", color=discord.Color.random())
    em.add_field(name="**Syntax**", value="$join")
    await ctx.send(embed=em)


@help.command()
async def leave(ctx):
    em = discord.Embed(
        title="join", description="Leaves the vc.", color=discord.Color.random())
    em.add_field(name="**Syntax**", value="$leave")
    await ctx.send(embed=em)


@help.command()
async def play(ctx):
    em = discord.Embed(
        title="play", description="Plays music searched by the user or queues the song if theres a song already playing.", color=discord.Color.random())
    em.add_field(name="**Syntax**",
                 value="$play <keywords for searching the song or url>")
    await ctx.send(embed=em)


@help.command()
async def queue(ctx):
    em = discord.Embed(
        title="queue", description="Shows current queue", color=discord.Color.random())
    em.add_field(name="**Syntax**", value="$queue")
    await ctx.send(embed=em)


@help.command()
async def resume(ctx):
    em = discord.Embed(
        title="resume", description="Resumes the paused song.", color=discord.Color.random())
    em.add_field(name="**Syntax**", value="$resume")
    await ctx.send(embed=em)


@help.command()
async def pause(ctx):
    em = discord.Embed(
        title="pause", description="Pauses the ongoing song.", color=discord.Color.random())
    em.add_field(name="**Syntax**", value="$pause")
    await ctx.send(embed=em)


@help.command()
async def skip(ctx):
    em = discord.Embed(
        title="skip", description="Skips the current song.", color=discord.Color.random())
    em.add_field(name="**Syntax**", value="$skip")
    await ctx.send(embed=em)


@help.command()
async def clear(ctx):
    em = discord.Embed(
        title="clear", description="Clears the queue.", color=discord.Color.random())
    em.add_field(name="**Syntax**", value="$clear")
    await ctx.send(embed=em)

# Fun


@help.command()
async def roll(ctx):
    em = discord.Embed(
        title="roll", description="Gives a random number in the given range", color=discord.Color.random())
    em.add_field(name="**Syntax**", value="$roll <range(int)>")
    await ctx.send(embed=em)


@help.command()
async def toss(ctx):
    em = discord.Embed(
        title="toss", description="Simulates a coin toss", color=discord.Color.random())
    em.add_field(name="**Syntax**", value="$toss")
    await ctx.send(embed=em)


@help.command()
async def meme(ctx):
    em = discord.Embed(
        title="meme", description="Gives a random meme from r/memes", color=discord.Color.random())
    em.add_field(name="**Syntax**", value="$meme")
    await ctx.send(embed=em)


@help.command()
async def joke(ctx):
    em = discord.Embed(title="joke", description="Tells a joke",
                       color=discord.Color.random())
    em.add_field(name="**Syntax**", value="$joke")
    await ctx.send(embed=em)


@help.command()
async def fact(ctx):
    em = discord.Embed(
        title="fact", description="Tells a random fact", color=discord.Color.random())
    em.add_field(name="**Syntax**", value="$fact")
    await ctx.send(embed=em)


@help.command()
async def inspire(ctx):
    em = discord.Embed(
        title="inspire", description="Tells a random quote", color=discord.Color.random())
    em.add_field(name="**Syntax**", value="$inspire")
    await ctx.send(embed=em)


@help.command()
async def compliment(ctx):
    em = discord.Embed(title="compliment",
                       description="Gives you a compliment to make your day", color=discord.Color.random())
    em.add_field(name="**Syntax**", value="$compliment")
    await ctx.send(embed=em)


@help.command()
async def bored(ctx):
    em = discord.Embed(
        title="bored", description="Gives you a fun activity to do if your bored", color=discord.Color.random())
    em.add_field(name="**Syntax**", value="$bored")
    await ctx.send(embed=em)

# Utility


@help.command()
async def qrcode(ctx):
    em = discord.Embed(
        title="qrcode", description="Converts given text or link to a QR code", color=discord.Color.random())
    em.add_field(name="**Syntax**", value="$qrcode <text or link>")
    await ctx.send(embed=em)


@help.command()
async def translate(ctx):
    em = discord.Embed(
        title="translate", description="Translates given text to the language required by the user.", color=discord.Color.random())
    em.add_field(name="**Syntax**",
                 value="$translate <language code> <text to translate>", inline=False)
    em.add_field(name="For language codes visit the site below: ",
                 value="https://developers.google.com/admin-sdk/directory/v1/languages", inline=False)

    await ctx.send(embed=em)


@help.command()
async def anime(ctx):
    em = discord.Embed(
        title="anime", description="Gives information about the searched anime", color=discord.Color.random())
    em.add_field(name="**Syntax**",
                 value="$anime <search keywords>", inline=False)
    await ctx.send(embed=em)


@help.command()
async def kick(ctx):
    em = discord.Embed(
        title="kick", description="Kicks the mentioned member", color=discord.Color.random())
    em.add_field(name="**Syntax**", value="$kick <@member> <reason(optional)>")
    await ctx.send(embed=em)


@help.command()
async def ban(ctx):
    em = discord.Embed(
        title="Ban", description="Bans the mentioned member", color=discord.Color.random())
    em.add_field(name="**Syntax**", value="$ban <@member> <reason(optional)>")
    await ctx.send(embed=em)


@help.command()
async def avatar(ctx):
    em = discord.Embed(
        title="Avatar", description="Gets the avatar of the mentioned member", color=discord.Color.random())
    em.add_field(name="**Syntax**", value="$avatar <@member(optional)>")
    await ctx.send(embed=em)


# Running the bot
client.run(TOKEN)
