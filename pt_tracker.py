import datetime
import pytz
import os
import re
import math
import traceback
import discord
from discord.ext import commands
from discord.ext import tasks
from discord.ext.commands import CommandNotFound
import asyncio
import urllib.request
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from PIL import Image
import numpy as np
import dateutil.parser as dp
import sqlite3
import requests
from dotenv import load_dotenv
from get_past_events import past_event
from get_acc import get_acc


load_dotenv()
TOKEN = os.getenv("TOKEN")
intents = discord.Intents.all()
bot = commands.Bot(command_prefix='!', intents=intents)

hktz = pytz.timezone("Etc/GMT-8")
con = sqlite3.connect("pt_tracker.db")
cur = con.cursor()

bot.event_no = 0
bot.start_day = datetime.datetime.now(hktz)
bot.end_day = datetime.datetime.now(hktz)
bot.next_event = 1
bot.next_start = datetime.datetime.now(hktz)
bot.next_end = datetime.datetime.now(hktz)
bot.predict500 = 0
bot.predict1000 = 0


def check_event_no(now_dt):
    response = urllib.request.urlopen("https://sekai-world.github.io/sekai-master-db-tc-diff/events.json")
    events = json.loads(response.read())
    # [[eid, start time, end time], ...]
    events_times = [[e["id"], datetime.datetime.fromtimestamp(e["startAt"]/1000, hktz),
                     datetime.datetime.fromtimestamp((e["aggregateAt"])/1000+1, hktz)]
                    for e in events]
    low = 0
    high = len(events_times) - 1

    if now_dt < events_times[low][1]:
        return False
    if now_dt > events_times[high][1]:
        return events_times[high], False

    while low <= high:
        mid = (low + high) // 2
        if events_times[mid][1] <= now_dt < events_times[mid + 1][1]:
            return events_times[mid], events_times[mid + 1]
        elif now_dt > events_times[mid][1]:
            low = mid + 1
        else:
            high = mid - 1

    return False  # current datetime not found in events


@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')
    channel = bot.get_channel(1007203228515057687)
    await channel.send("Saki on ready <:ln_saki_cute:1015609268055060530>")
    now = datetime.datetime.now(hktz)
    curr_event, next_event = check_event_no(now)
    if curr_event:
        bot.event_no = curr_event[0]
        bot.start_day = curr_event[1]
        bot.end_day = curr_event[2]
        bot.next_event = bot.event_no + 1
        await channel.send(f"Current event: {bot.event_no}\n" +
                           "- Event start: " + bot.start_day.strftime("%m/%d %H:%M") + "\n" +
                           "- Event end: " + bot.end_day.strftime("%m/%d %H:%M"))
    if next_event:
        bot.next_start = next_event[1]
        bot.next_end = next_event[2]
        await channel.send(f"Next event: {bot.next_event}\n" +
                           "- Event start: " + bot.next_start.strftime("%m/%d %H:%M") + "\n" +
                           "- Event end: " + bot.next_end.strftime("%m/%d %H:%M"))
    else:
        await channel.send("<@598066719659130900> Failed to find current event <:ln_saki_weapon:1006929901745614859>")

    check_event_change.start()
    await channel.send("check_event_change started")
    predict.start()
    await channel.send("predict loop started")
    insert_live_rank.start()
    await channel.send("insert_live_rank started")
    restart_stopped_loops.start()


@bot.event
async def on_raw_reaction_add(payload):
    channel = bot.get_channel(1006460760081321985)
    if payload.channel_id != channel.id:
        return
    if payload.user_id != 1034565760846135356:
        guild_id = payload.guild_id
    else:
        return
    if payload.message_id == 1177396331115978762:
        emoji = bot.get_emoji(1006929901745614859)
        if payload.emoji == emoji:
            await payload.member.add_roles(discord.utils.get(bot.get_guild(guild_id).roles, id=1177391192602849390))
        else:
            message = await bot.get_partial_messageable(payload.channel_id).fetch_message(payload.message_id)
            await message.remove_reaction(payload.emoji, payload.member)


@bot.event
async def on_raw_reaction_remove(payload):
    channel = bot.get_channel(1006460760081321985)
    if payload.channel_id != channel.id:
        return
    if payload.user_id != 1034565760846135356:
        guild_id = payload.guild_id
        guild = bot.get_guild(guild_id)
        member = guild.get_member(payload.user_id)
    else:
        return
    if payload.message_id == 1177396331115978762:
        emoji = bot.get_emoji(1006929901745614859)
        if payload.emoji == emoji:
            await member.remove_roles(discord.utils.get(bot.get_guild(guild_id).roles, id=1177391192602849390))


@bot.event
async def on_message(message):
    tick = bot.get_emoji(1008440207873409116)
    cross = bot.get_emoji(1008633491807817789)

    async def error(err_message: discord.Message, error_desc):
        await err_message.add_reaction(cross)
        await err_message.reply(error_desc)

    if message.author.id == 1034565760846135356:
        return
    if message.channel.id in [1296159421704966296, 1296159613921525793]:
        if not re.match("[0-9]+", message.content):
            await error(message, "分數上報格式不正確 (`分數`) <:ln_saki_weapon:1006929901745614859>")
            return
        if message.channel.id == 1296159421704966296:
            rank = 500
        else:
            rank = 1000
        pt = int(message.content)
        now = datetime.datetime.now(hktz)

        if bot.start_day <= now <= bot.end_day:
            start_hrs = round((now - bot.start_day).total_seconds() / (60 * 60), 3)
            if rank == 500:
                cur.execute("INSERT INTO timept500 VALUES (?, ?, ?)", (bot.event_no, start_hrs, pt))
                con.commit()
            else:
                cur.execute("INSERT INTO timept1000 VALUES (?, ?, ?)", (bot.event_no, start_hrs, pt))
                con.commit()
            await message.add_reaction(tick)

        elif bot.end_day < now <= bot.next_start - datetime.timedelta(hours=15):  # 0000 before next event
            start_hrs = math.floor((bot.end_day - bot.start_day).total_seconds() / (60 * 60))
            if rank == 500:
                cur.execute("INSERT INTO timept500 VALUES (?, ?, ?)", (bot.event_no, start_hrs, pt))
                con.commit()
            else:
                cur.execute("INSERT INTO timept1000 VALUES (?, ?, ?)", (bot.event_no, start_hrs, pt))
                con.commit()
            await message.add_reaction(tick)

        else:
            await message.add_reaction(cross)
            await error(message, "現在沒有正在進行的活動 <:ln_saki_cry:1008601057380814849>")
            return

    await bot.process_commands(message)


@bot.event
async def on_raw_message_delete(payload):
    if payload.channel_id in [1296159421704966296, 1296159613921525793]:
        if payload.channel_id == 1296159421704966296:
            channel = bot.get_channel(1296159421704966296)
            rank = 500
        else:
            channel = bot.get_channel(1296159613921525793)
            rank = 1000
        message = payload.cached_message

        if message is not None:
            if message.author.bot:
                return
            else:
                pt = int(message.content)
                if rank == 500:
                    check_record = cur.execute("SELECT * FROM timept500 WHERE eid = ? and pt = ?",
                                               (bot.event_no, pt)).fetchone()
                    if not check_record:
                        await channel.send("分數紀錄刪除失敗 <:ln_saki_weapon:1006929901745614859>")
                        return
                    cur.execute("DELETE FROM timept500 WHERE eid = ? and pt = ?", (bot.event_no, pt))
                    con.commit()
                else:
                    check_record = cur.execute("SELECT * FROM timept1000 WHERE eid = ? and pt = ?",
                                               (bot.event_no, pt)).fetchone()
                    if not check_record:
                        await channel.send("分數紀錄刪除失敗 <:ln_saki_weapon:1006929901745614859>")
                        return
                    cur.execute("DELETE FROM timept1000 WHERE eid = ? and pt = ?", (bot.event_no, pt))
                    con.commit()
                await channel.send("成功刪除分數紀錄 <:ln_saki_cute:1015609268055060530>")
        else:
            await channel.send("分數紀錄刪除失敗 <:ln_saki_weapon:1006929901745614859>")


# ----- loops -----
@tasks.loop(time=datetime.time(4, 0, 0, tzinfo=hktz), reconnect=True)
async def check_event_change():  # check every day at 04:00
    print(f"check event change")
    now = datetime.datetime.now(hktz)
    if bot.next_start - datetime.timedelta(hours=15) <= now < bot.next_start:  # 00:00 <= now < 15:00
        channel = bot.get_channel(1007203228515057687)
        await channel.send(f"Event change activated [{now.strftime('%d/%m %H:%M:%S')}]")
        # ----- calculate accuracy ----
        api500 = f"https://api.sekai.best/event/{bot.event_no}/rankings/graph?region=tw&rank=500"
        res500 = json.loads(requests.get(api500).text)["data"]["eventRankings"][-1]
        actual500 = res500["score"]
        actual500_timestamp = res500["timestamp"]
        await channel.send(f"T500: {actual500}pt at {actual500_timestamp}")

        api1000 = f"https://api.sekai.best/event/{bot.event_no}/rankings/graph?region=tw&rank=1000"
        res1000 = json.loads(requests.get(api1000).text)["data"]["eventRankings"][-1]
        actual1000 = res1000["score"]
        actual1000_timestamp = res1000["timestamp"]
        await channel.send(f"T1000: {actual1000}pt at {actual1000_timestamp}")

        try:
            cur.execute("INSERT INTO accuracy VALUES (?, ?, ?, ?, ?, ?, ?)", (bot.event_no,
                                                                              bot.predict500, actual500,
                                                                              bot.predict500 - actual500,
                                                                              bot.predict1000, actual1000,
                                                                              bot.predict1000 - actual1000))
            con.commit()
        except sqlite3.IntegrityError:
            cur.execute(
                "UPDATE accuracy SET predict500 = ?,  predict1000 = ? , diff500 = ?, diff1000 = ? WHERE eid = ?",
                (bot.predict500, bot.predict1000, bot.predict500 - actual500, bot.predict1000 - actual1000,
                 bot.event_no))
            con.commit()
        await channel.send("Insertion into accuracy db completed")

        """
        total_hrs = (bot.end_day - bot.start_day).total_seconds() / (60 * 60)
        end_pt_500 = cur.execute("SELECT * FROM timept500 WHERE eid = ? and time = ?",
                                 (bot.event_no, total_hrs)).fetchone()
        end_pt_1000 = cur.execute("SELECT * FROM timept1000 WHERE eid = ? and time = ?",
                                  (bot.event_no, total_hrs)).fetchone()
        if end_pt_500 and end_pt_1000:
            end_pt_500 = end_pt_500[0]
            end_pt_1000 = end_pt_1000[0]
            try:
                cur.execute("INSERT INTO accuracy VALUES (?, ?, ?, ?, ?, ?, ?)", (bot.event_no,
                                                                                  bot.predict500, actual500,
                                                                                  bot.predict500 - actual500,
                                                                                  bot.predict1000, end_pt_1000,
                                                                                  bot.predict1000 - end_pt_1000))
                con.commit()
            except sqlite3.IntegrityError:
                cur.execute(
                    "UPDATE accuracy SET predict500 = ?,  predict1000 = ? , diff500 = ?, diff1000 = ? WHERE eid = ?",
                    (bot.predict500, bot.predict1000, bot.predict500 - end_pt_500, bot.predict1000 - end_pt_1000,
                     bot.event_no))
                con.commit()
            await channel.send("Insertion into accuracy db completed")

        elif not end_pt_500:
            await channel.send("<@598066719659130900>\n"
                               "沒有結活T500分數記錄，請儘快在<#1296159421704966296>上報 <:ln_saki_otsu:1006480191431909457>")

        if not end_pt_1000:
            await channel.send("<@598066719659130900>\n"
                               "沒有結活T1000分數記錄，請儘快在<#1296159613921525793>上報 <:ln_saki_otsu:1006480191431909457>")

        upload_channel = bot.get_channel(1296159421704966296)
        await upload_channel.send(
            "----- " + str(bot.event_no) + "期分數上報結束 <:ln_saki_otsu:1006480191431909457> -----")
        upload_channel = bot.get_channel(1296159613921525793)
        await upload_channel.send(
            "----- " + str(bot.event_no) + "期分數上報結束 <:ln_saki_otsu:1006480191431909457> -----")
        """

        # ----- insert current event graph to db -----
        data500 = past_event(bot.event_no, 500)
        data1000 = past_event(bot.event_no, 1000)
        for i in data500:
            try:
                cur.execute("INSERT INTO timept500 VALUES (?, ?, ?)", (bot.event_no, i[0], i[1]))
                con.commit()
            except sqlite3.IntegrityError:
                continue
        for j in data1000:
            try:
                cur.execute("INSERT INTO timept1000 VALUES (?, ?, ?)", (bot.event_no, j[0], j[1]))
                con.commit()
            except sqlite3.IntegrityError:
                continue

        # ----- change to next event -----
        bot.event_no = bot.next_event
        bot.start_day = bot.next_start
        bot.end_day = bot.next_end
        bot.next_event += 1
        response = urllib.request.urlopen("https://sekai-world.github.io/sekai-master-db-tc-diff/events.json")
        events = json.loads(response.read())
        next_event = [e for e in events if e["id"] == bot.next_event][0]
        bot.next_start = datetime.datetime.fromtimestamp(next_event["startAt"]/1000, hktz)
        bot.next_end = datetime.datetime.fromtimestamp(next_event["aggregateAt"]/1000+1, hktz)
        cur.execute("INSERT INTO timept500 VALUES (?, ?, ?)", (bot.event_no, 0, 0))
        cur.execute("INSERT INTO timept1000 VALUES (?, ?, ?)", (bot.event_no, 0, 0))
        con.commit()
        await channel.send("Event change <:ln_saki_excited:1011509870081626162>\n"
                           f"Current event: {bot.event_no}\n"
                           f"- Event start: {bot.start_day.strftime('%m/%d %H:%M')}\n"
                           f"- Event end: {bot.end_day.strftime('%m/%d %H:%M')}\n"
                           f"Next event: {bot.next_event}\n" +
                           "- Event start: " + bot.next_start.strftime("%m/%d %H:%M") + "\n" +
                           "- Event end: " + bot.next_end.strftime("%m/%d %H:%M"))
        predict.restart()
        await channel.send("Predict loop restarted")


@bot.command(hidden=True)
async def change_event(ctx):
    print(f"Event change manually called")
    now = datetime.datetime.now(hktz)
    channel = bot.get_channel(1007203228515057687)
    await channel.send(f"Event change activated [{now.strftime('%d/%m %H:%M:%S')}]")
    # calculate accuracy
    total_hrs = (bot.end_day - bot.start_day).total_seconds() / (60 * 60)
    end_pt_500 = cur.execute("SELECT * FROM timept500 WHERE eid = ? and time = ?",
                             (bot.event_no, total_hrs)).fetchone()
    end_pt_1000 = cur.execute("SELECT * FROM timept1000 WHERE eid = ? and time = ?",
                              (bot.event_no, total_hrs)).fetchone()
    if end_pt_500 and end_pt_1000:
        end_pt_500 = end_pt_500[2]
        end_pt_1000 = end_pt_1000[2]
        try:
            cur.execute("INSERT INTO accuracy VALUES (?, ?, ?, ?, ?, ?, ?)", (bot.event_no,
                                                                              bot.predict500, end_pt_500,
                                                                              bot.predict500 - end_pt_500,
                                                                              bot.predict1000, end_pt_1000,
                                                                              bot.predict1000 - end_pt_1000))
            con.commit()
        except sqlite3.IntegrityError:
            pass
        await channel.send("Insertion into accuracy db completed")

    elif not end_pt_500:
        await channel.send("<@598066719659130900>\n"
                           "沒有結活T500分數記錄，請儘快在<#1296159421704966296>上報 <:ln_saki_otsu:1006480191431909457>")

    if not end_pt_1000:
        await channel.send("<@598066719659130900>\n"
                           "沒有結活T1000分數記錄，請儘快在<#1296159613921525793>上報 <:ln_saki_otsu:1006480191431909457>")

    upload_channel = bot.get_channel(1296159421704966296)
    await upload_channel.send(
        "----- " + str(bot.event_no) + "期分數上報結束 <:ln_saki_otsu:1006480191431909457> -----")
    upload_channel = bot.get_channel(1296159613921525793)
    await upload_channel.send(
        "----- " + str(bot.event_no) + "期分數上報結束 <:ln_saki_otsu:1006480191431909457> -----")

    # change to next event
    bot.event_no = bot.next_event
    bot.start_day = bot.next_start
    bot.end_day = bot.next_end
    bot.next_event += 1
    response = urllib.request.urlopen("https://sekai-world.github.io/sekai-master-db-tc-diff/events.json")
    events = json.loads(response.read())
    next_event = [e for e in events if e["id"] == bot.next_event][0]
    bot.next_start = datetime.datetime.fromtimestamp(next_event["startAt"]/1000, hktz)
    bot.next_end = datetime.datetime.fromtimestamp(next_event["aggregateAt"]/1000+1, hktz)
    try:
        cur.execute("INSERT INTO timept500 VALUES (?, ?, ?)", (bot.event_no, 0, 0))
        cur.execute("INSERT INTO timept1000 VALUES (?, ?, ?)", (bot.event_no, 0, 0))
        con.commit()
    except sqlite3.IntegrityError:
        pass
    await channel.send("Event change <:ln_saki_excited:1011509870081626162>\n"
                       f"Current event: {bot.event_no}\n"
                       f"- Event start: {bot.start_day.strftime('%m/%d %H:%M')}\n"
                       f"- Event end: {bot.end_day.strftime('%m/%d %H:%M')}\n"
                       f"Next event: {bot.next_event}\n" +
                       "- Event start: " + bot.next_start.strftime("%m/%d %H:%M") + "\n" +
                       "- Event end: " + bot.next_end.strftime("%m/%d %H:%M"))


def calc_mape(prev, curr):
    return np.mean(np.abs((prev - curr) / prev))


@tasks.loop(time=[datetime.time(0, 0, 45, tzinfo=hktz), datetime.time(16, 0, 45, tzinfo=hktz),
                  datetime.time(19, 0, 45, tzinfo=hktz), datetime.time(20, 0, 45, tzinfo=hktz)], reconnect=True)
async def predict():
    now = datetime.datetime.now(hktz)
    print("predict task run")
    if bot.end_day.hour == 21:  # normal event (X world link)
        if now.hour == 19:
            return
        if now < bot.start_day + datetime.timedelta(days=1, hours=8, minutes=59):  # before D3 0000 (< D2 2359)
            return
        if now.date() != bot.end_day.date():
            if now.hour == 20:
                return
        if now.date() > bot.end_day.date():
            return
    else:  # world link (ends 20:00)
        if now.hour == 20:
            return
        if now < bot.start_day + datetime.timedelta(days=1, hours=8, minutes=59):  # before D3 0000 (< D2 2359)
            return
        if now.date() != bot.end_day.date():
            if now.hour == 19:
                return
        if now.date() > bot.end_day.date():
            return

    start_hrs = math.floor((now - bot.start_day).total_seconds() / (60 * 60))
    total_hrs = (bot.end_day - bot.start_day).total_seconds() / (60 * 60)
    prev_eids = cur.execute("SELECT distinct eid FROM timept500").fetchall()
    prev_eids = [int(i[0]) for i in prev_eids]
    # prev_eids = [36, 37, 38, 39, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 57, 59, 60, 61, 62, 63, 64, 65, 66]

    # ----- T500 prediction -----
    curr_500 = list(past_event(bot.event_no, 500))
    # (old) curr_500 = cur.execute("SELECT time, pt FROM timept500 WHERE eid = ?", (bot.event_no,)).fetchall()
    curr_500_x = [i[0] for i in curr_500]
    curr_500_y = [i[1] for i in curr_500]
    last_500_x = curr_500_x[-1]
    ts = np.arange(1, math.floor(last_500_x)+1, 4)  # hrs 1, 5, 9... 61 (eg.last_x=64.5), arange: [1,floor.last_x+1)
    ts_subset = ts[2:]  # from hr = 9
    curr_500_interp = np.interp(ts, curr_500_x, curr_500_y)  # get values at each timestep in ts
    mapes = []
    for e in prev_eids:
        prev_500 = cur.execute("SELECT time, pt FROM timept500 WHERE eid = ?", (e,)).fetchall()
        prev_x = [i[0] for i in prev_500]
        prev_y = [i[1] for i in prev_500]
        prev_500_interp = np.interp(ts, prev_x, prev_y)
        mape = calc_mape(prev_500_interp, curr_500_interp)
        mapes.append(mape)
    event_mapes = zip(prev_eids, mapes)
    event_mapes = [[i[0], i[1]] for i in event_mapes]
    event_mapes = sorted(event_mapes, key=lambda x: x[1])
    ms_event = event_mapes[0][0]  # most similar event
    s_events = [event_mapes[0][0], event_mapes[1][0], event_mapes[2][0]]  # most similar 3 events
    acc = get_acc(ms_event, 500)  # accumulation change time and magnitude [[acc_time], [acc_score]] during final rush
    if acc == KeyError:
        print("most similar error - trying 2nd most similar")
        print(s_events[1])
        acc = get_acc(s_events[1], 500)
        if acc == KeyError:
            print("2nd similar error - trying 3rd most similar")
            print(s_events[2])
            acc = get_acc(s_events[2], 500)
    acc = [[total_hrs - i[0], i[1]] for i in acc]  # [[hr from start, pt increase to next acc], ... [hr, pt increase to final]]
    # ----- extrapolate -----
    acc_hr = acc[0][0]  # first acc time, hours from start
    curr_500_subset = curr_500_interp[2:]  # pt at hr = 9 to now
    extrap_500 = np.poly1d(np.polyfit(ts_subset, curr_500_subset, 1))  # extrapolate from hr 9 (D1 speed often faster)
    acc_pt = extrap_500(acc_hr).item()
    # ----- add acc data -----
    extrap_500_x = [acc_hr]
    extrap_500_y = [acc_pt]
    last_pt = acc_pt
    final_pt = 0
    for i in range(len(acc)):
        if i != len(acc) - 1:
            extrap_500_x.append(acc[i+1][0])
            extrap_500_y.append(last_pt + acc[i][1])
            last_pt += acc[i][1]
        else:
            extrap_500_x.append(total_hrs)
            final_pt = last_pt + acc[i][1]
            extrap_500_y.append(final_pt)
    if start_hrs <= total_hrs - 8:
        predict_500_x = [curr_500_x[-1]] + extrap_500_x
        predict_500_y = [curr_500_y[-1]] + extrap_500_y
    else:
        predict_500_x = extrap_500_x
        predict_500_y = extrap_500_y
    # ----- plot (current + prediction + most similar + 2 others) -----
    colours = ["darkblue", "royalblue", "lightskyblue"]
    fig, ax = plt.subplots()
    ax.grid(which='minor', linestyle=':', linewidth=0.4, color='#dbdbdb', axis="y")
    ax.grid(which='major', linestyle='-', linewidth=0.45, color='#dbdbdb', axis="y")
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    for x in [9, 33, 57, 81, 105, 129, 153, 177, 201, 225, 249]:
        ax.axvline(x, color='#b8b8b8', linewidth=0.75, linestyle='-')
    ci = 0
    for i in s_events:
        timept = cur.execute("SELECT time, pt FROM timept500 WHERE eid = ? ORDER BY time", (i, )).fetchall()
        if timept:
            x = [r[0] for r in timept]
            y = [r[1] for r in timept]
            ax.plot(x, y, label=str(i), linewidth=1, color=colours[ci])
            ci += 1
        else:
            continue
    ax.plot(curr_500_x, curr_500_y, label="Current", linewidth=1.2, color="r")
    ax.plot(predict_500_x, predict_500_y, label="Prediction", linewidth=1.25, color="r", ls="--")
    ax.legend(title="Event", alignment="left", labelspacing=0.2, fontsize="x-small", title_fontsize="small", loc=2)
    ax.set_title("T500 分數線預測")
    ax.set_xlabel("Time (hrs since start, grey=00:00)")
    ax.set_ylabel("Points")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.savefig("predict500.png", dpi=600, bbox_inches='tight', pad_inches=0.25)
    channel = bot.get_channel(1177954438326013993)
    await channel.send("**__" + now.strftime("%m/%d %H:%M") + " 結活分數預測__**")
    await channel.send("T500: " + str(int(round(final_pt))))
    await channel.send(file=discord.File('predict500.png'))
    if bot.end_day.hour == 21:
        if now.date() == bot.end_day.date() and now.hour == 20:
            bot.predict500 = int(round(final_pt))
    else:  # world link: ends at 20:00
        if now.date() == bot.end_day.date() and now.hour == 20:
            bot.predict500 = int(round(final_pt))
    plt.clf()
    plt.close("all")

    # ----- T1000 prediction -----
    curr_1000 = list(past_event(bot.event_no, 1000))
    # curr_1000 = cur.execute("SELECT time, pt FROM timept1000 WHERE eid = ?", (bot.event_no,)).fetchall()
    curr_1000_x = [i[0] for i in curr_1000]
    curr_1000_y = [i[1] for i in curr_1000]
    last_1000_x = curr_1000_x[-1]
    ts = np.arange(1, math.floor(last_1000_x)+1, 4)
    ts_subset = ts[2:]
    curr_1000_interp = np.interp(ts, curr_1000_x, curr_1000_y)
    mapes = []
    for e in prev_eids:
        prev_1000 = cur.execute("SELECT time, pt FROM timept1000 WHERE eid = ?", (e,)).fetchall()
        prev_x = [i[0] for i in prev_1000]
        prev_y = [i[1] for i in prev_1000]
        prev_1000_interp = np.interp(ts, prev_x, prev_y)
        mape = calc_mape(prev_1000_interp, curr_1000_interp)
        mapes.append(mape)
    event_mapes = zip(prev_eids, mapes)
    event_mapes = [[i[0], i[1]] for i in event_mapes]
    event_mapes = sorted(event_mapes, key=lambda x: x[1])
    ms_event = event_mapes[0][0]
    s_events = [event_mapes[0][0], event_mapes[1][0], event_mapes[2][0]]
    acc = get_acc(ms_event, 1000)
    if acc == KeyError:
        print("most similar error - trying 2nd most similar")
        print(s_events[1])
        acc = get_acc(s_events[1], 1000)
        if acc == KeyError:
            print("2nd similar error - trying 3rd most similar")
            print(s_events[2])
            acc = get_acc(s_events[2], 1000)
    acc = [[total_hrs - i[0], i[1]] for i in acc]
    # ----- extrapolate -----
    acc_hr = acc[0][0]  # first acc time, hours from start
    curr_1000_subset = curr_1000_interp[2:]  # pt at hr = 9 to now
    extrap_1000 = np.poly1d(np.polyfit(ts_subset, curr_1000_subset, 1))
    acc_pt = extrap_1000(acc_hr).item()
    # ----- add acc data -----
    extrap_1000_x = [acc_hr]
    extrap_1000_y = [acc_pt]
    last_pt = acc_pt
    final_pt = 0
    for i in range(len(acc)):
        if i != len(acc) - 1:
            extrap_1000_x.append(acc[i + 1][0])
            extrap_1000_y.append(last_pt + acc[i][1])
            last_pt += acc[i][1]
        else:
            extrap_1000_x.append(total_hrs)
            final_pt = last_pt + acc[i][1]
            extrap_1000_y.append(final_pt)
    predict_1000_x = [curr_1000_x[-1]] + extrap_1000_x
    predict_1000_y = [curr_1000_y[-1]] + extrap_1000_y
    # ----- plot (current + prediction + most similar + 2 others) -----
    colours = ["darkblue", "royalblue", "lightskyblue"]
    fig, ax = plt.subplots()
    ax.grid(which='minor', linestyle=':', linewidth=0.4, color='#dbdbdb', axis="y")
    ax.grid(which='major', linestyle='-', linewidth=0.45, color='#dbdbdb', axis="y")
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    for x in [9, 33, 57, 81, 105, 129, 153, 177, 201, 225, 249]:
        ax.axvline(x, color='#b8b8b8', linewidth=0.75, linestyle='-')
    ci = 0
    for i in s_events:
        timept = cur.execute("SELECT time, pt FROM timept1000 WHERE eid = ? ORDER BY time", (i,)).fetchall()
        if timept:
            x = [r[0] for r in timept]
            y = [r[1] for r in timept]
            ax.plot(x, y, label=str(i), linewidth=1, color=colours[ci])
            ci += 1
        else:
            continue
    ax.plot(curr_1000_x, curr_1000_y, label="Current", linewidth=1.2, color="r")
    ax.plot(predict_1000_x, predict_1000_y, label="Prediction", linewidth=1.25, color="r", ls="--")
    ax.legend(title="Event", alignment="left", labelspacing=0.2, fontsize="x-small", title_fontsize="small", loc=2)
    ax.set_title("T1000 分數線預測")
    ax.set_xlabel("Time (hrs since start, grey=00:00)p")
    ax.set_ylabel("Points")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.savefig("predict1000.png", dpi=600, bbox_inches='tight', pad_inches=0.25)
    # {D{(last_1000_x-9)//23.9999999 + 1}) (bot.start_day + datetime.timedelta(hours=last_1000_x)).strftime('%H:%M')}")
    await channel.send(f"T1000: {int(round(final_pt))}")
    await channel.send(file=discord.File('predict1000.png'))
    if now.date() == bot.end_day.date() and now.hour == 20:
        bot.predict1000 = int(round(final_pt))
    plt.clf()
    plt.close("all")


@predict.error
async def predict_exited(self):
    print(self)
    channel = bot.get_channel(1007203228515057687)
    await channel.send("predict loop error <@598066719659130900>, error:")
    await channel.send(self)
    traceback.print_exc()


@tasks.loop(minutes=3.05, reconnect=True)  # loop every 3min 3s (api updates ~ every 3min)
async def insert_live_rank():
    api = "https://api.sekai.best/event/live?region=tw"
    res = json.loads(requests.get(api).text)["data"]["eventRankings"]
    res = [(i['eventId'], int(dp.parse(i['timestamp']).timestamp()), i['rank'], i['score'],
            i['userId'], i['userName']) for i in res]
    try:
        cur.executemany("INSERT INTO liverank VALUES (?, ?, ?, ?, ?, ?)", res)
        con.commit()
    except sqlite3.IntegrityError:
        pass


@tasks.loop(minutes=5, reconnect=True)
async def restart_stopped_loops():
    if not predict.is_running():
        predict.start()
    if not insert_live_rank.is_running():
        insert_live_rank.start()
    

@bot.command(name="role", hidden=True)
async def role(ctx):
    reminder_message = await ctx.send(
        "按反應領取分數上報提醒身分組:\n"
        "<:ln_saki_weapon:1006929901745614859> 活動期間在指定時段提醒你上報分數")
    bot.reminder_message_id = reminder_message.id
    await reminder_message.add_reaction("<:ln_saki_weapon:1006929901745614859>")


# ----- loop actions -----
@bot.command(name="start_predict", hidden=True)  # hours=1, every 00
async def start_predict(ctx):
    if not predict.is_running():
        predict.start()
        now = datetime.datetime.now(hktz)
        await ctx.send("Predict loop started at " + now.strftime("%H:%M:%S"))


@bot.command(name="end_predict")
async def end_predict(ctx):
    if predict.is_running():
        predict.cancel()
        await ctx.send("Predict loop stopped")


@bot.command(name="start_live_rank", hidden=True)
async def start_live_rank(ctx):
    if not insert_live_rank.is_running():
        insert_live_rank.start()
        now = datetime.datetime.now(hktz)
        await ctx.send("insert_live_rank started at " + now.strftime("%H:%M:%S"))


@bot.command(name="check_loops", hidden=True)
async def check_loops(ctx):
    if predict.is_running():
        await ctx.send("predict running")
    else:
        await ctx.send("predict stopped")
    if check_event_change.is_running():
        await ctx.send("check_event_change running")
    else:
        await ctx.send("check_event_change stopped")
    if insert_live_rank.is_running():
        await ctx.send("insert_live_rank running")
    else:
        await ctx.send("insert_live_rank stopped")


# ----- commands -----
@bot.command(name='saki')
async def saki(ctx, mod=None):
    if mod == "mod":
        embed_all = discord.Embed(title="__Saki指令表(mod)__",
                                  description="天馬咲希bot由SK所寫\n以下為saki的指令介紹:"
                                              "\n\n**!start_predict, !end_predict, !start_live_rank**"
                                              "\nStart/end loops manually"
                                              "\n\n**!check_loops**"
                                              "\nCheck loops status"
                                              "\n\n**!change_event**"
                                              "\nStart change event manually"
                                              "\n\n**!test_predict**"
                                              "\nManually call predict task then send T500 and T1000 predictions"
                                              "\n\n**!q <query>**"
                                              "\nExecute query"
                                              "\n\n----- fix databases -----"
                                              "\n**!create [db=None]**"
                                              "\nCreate db (None=all):"
                                              "\n- timept500/1000(eid, time, pt)"
                                              "\n- accuracy(eid, predict500, actual500, diff500, "
                                              "predict1000, actual1000, diff1000)"
                                              "\n- liverank(eid, time, rank, pt, pid, pname)"
                                              "\n\n**!insert_past_events <start> <end>**"
                                              "\nInsert past event pts into dbs timept500/1000"
                                              "\n\n**!insert_0**"
                                              "\nAdd (time,pt)=(0,0) to all events in timept500/1000"
                                              "\n\n**!fix_data**"
                                              "\nAdd synthetic pt data for missing values in timept500/1000"
                                              "\n** **",
                                  colour=0xFFDD45)
        embed_all.set_footer(text="最後更新: 17/02/2025")
    else:
        embed_all = discord.Embed(title="__Saki指令表__",
                                  description="天馬咲希bot由SK所寫\n以下為saki的指令介紹:"
                                              "\n\n**!plot <rank> <*events>**\n查看當期和指定過往活動的分數線，每次最多可查20個活動"
                                              "\n(40, 42, 54, 55, 56, 58, 67-116期沒有紀錄)"
                                              "\nrank: 500 | 1000"
                                              "\n例: !plot 500 36 37 59"
                                              "\n\n**!accuracy**\n查看過往活動結活當天20:00分數預測的誤差"
                                              "\n\n**!speed <rank>**\n查看指定玩家或排名的時速"
                                              "\nT1-50: 玩家時速"
                                              "\nT51或以下: 排名分數線時速"
                                              "\n(可查排名: T1-100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, "
                                              "4000, 5000, 10000, 20000, 30000, 40000, 50000)"
                                              "\n\n**!saki**\n顯示此指令表"
                                              "\n** **",
                                  colour=0xFFDD45)
        embed_all.set_footer(text="最後更新: 17/02/2025")
    await ctx.send(embed=embed_all)


@bot.command(name="plot")
async def plot(ctx, rank: int, *events):
    if rank not in [500, 1000]:
        await ctx.send("查詢排名必須 = 500 | 1000 <:ln_saki_cry:1008601057380814849>")
        return
    eids = [int(i) for i in events]
    if len(eids) > 20:
        await ctx.send("每次最多只能查20個活動 <:ln_saki_weapon:1006929901745614859>")
        return
    past_eids = cur.execute("SELECT DISTINCT eid from timept500").fetchall()
    eids = [i for i in eids if i in [j[0] for j in past_eids]]

    async with ctx.typing():
        fig, ax = plt.subplots()
        ax.grid(which='minor', linestyle=':', linewidth=0.4, color='#dbdbdb', axis="y")
        ax.grid(which='major', linestyle='-', linewidth=0.45, color='#dbdbdb', axis="y")
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        for x in [9, 33, 57, 81, 105, 129, 153, 177, 201, 225, 249]:
            ax.axvline(x, color='#b8b8b8', linewidth=0.75, linestyle='-')

        colours = plt.get_cmap("tab20")
        colours = colours(np.linspace(0, 1, len(eids)))
        ci = 0
        for eid in eids:
            if rank == 500:
                ts = cur.execute("SELECT time, pt FROM timept500 WHERE eid = ? ORDER BY time", (eid,)).fetchall()
            else:
                ts = cur.execute("SELECT time, pt FROM timept1000 WHERE eid = ? ORDER BY time", (eid,)).fetchall()
            if ts:
                x = [r[0] for r in ts]
                y = [r[1] for r in ts]
                ax.plot(x, y, label=str(eid), linewidth=1, color=colours[ci])
                ci += 1
            else:
                continue

        curr = list(past_event(bot.event_no, rank))
        curr_x = [i[0] for i in curr]
        curr_y = [i[1] for i in curr]
        ax.plot(curr_x, curr_y, label="Current", linewidth=1, color="r")

        ax.set_title("T" + str(rank) + " 分數線")
        ax.legend(title="Event", alignment="left", labelspacing=0.2, fontsize="x-small", title_fontsize="small", loc=2)
        ax.set_xlabel("Time (hrs since start, grey=00:00)")
        ax.set_ylabel("Points")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        fig.savefig("plot.png", dpi=600, bbox_inches='tight', pad_inches=0.25)
    await ctx.send(file=discord.File('plot.png'))
    fig.clf()
    plt.close("all")


@bot.command(name="accuracy")
async def accuracy(ctx):
    async with ctx.typing():
        data = cur.execute("SELECT * FROM accuracy").fetchall()
        cell_text = []
        for i in data:
            if i[3] <= 0 and i[6] <= 0:
                cell_text.append([i[0], int(i[1]), i[2], int(i[3]), int(i[4]), i[5], int(i[6])])
            elif i[3] > 0 and i[6] <= 0:
                cell_text.append([i[0], int(i[1]), i[2], "+" + str(int(i[3])), int(i[4]), i[5], int(i[6])])
            elif i[3] <= 0 and i[6] > 0:
                cell_text.append([i[0], int(i[1]), i[2], int(i[3]), int(i[4]), i[5], "+" + str(int(i[6]))])
            elif i[3] > 0 and i[6] > 0:
                cell_text.append([i[0], int(i[1]), i[2], "+" + str(int(i[3])), int(i[4]), i[5], "+" + str(int(i[6]))])
        col_labels = ["活動", "T500 預測", "結活分數", "誤差", "T1000 預測", "結活分數", "誤差"]
        rows = len(data)
        row_colours = ["white", "white", "white", "#b1daf0", "white", "white", "#b1daf0"]
        cell_colours = []
        for i in range(rows):
            cell_colours.append(row_colours)
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        table = ax.table(cellText=cell_text, colLabels=col_labels,
                         colColours=['#9ad0ed', '#9ad0ed', '#9ad0ed', '#68b8e3',
                                     '#9ad0ed', '#9ad0ed', '#68b8e3'],
                         cellColours=cell_colours,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8.8)
        plt.tight_layout()
        plt.savefig('acc_table.png', dpi=300, bbox_inches='tight', edgecolor='blue')
        with Image.open('acc_table.png') as im:
            cropped = im.crop(im.getbbox())
            cropped.save('acc_table.png')
    await ctx.send(file=discord.File('acc_table.png'))
    plt.clf()
    plt.close("all")


@bot.command(name="speed")
async def speed(ctx, rank: int):
    ranks = cur.execute("SELECT DISTINCT rank FROM liverank").fetchall()
    ranks = [i[0] for i in ranks]
    if rank not in ranks:
        await ctx.send(f"不支持查詢T{rank}的周回 <:ln_saki_cry:1008601057380814849>\n"
                       f"(可查 T1-100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, "
                       f"10000, 20000, 30000, 40000, 50000)")
        return
    async with ctx.typing():
        rank_data = cur.execute("SELECT * FROM liverank WHERE eid = ? AND rank = ? ORDER BY time DESC",
                                (bot.event_no, rank,)).fetchone()

        curr_time = rank_data[1]
        curr_p = rank_data[3]

        if 1 <= rank <= 50:
            pname = rank_data[5]
            pid = rank_data[4]
            p_15 = cur.execute("SELECT pt FROM liverank WHERE eid = ? AND pid = ? AND (time BETWEEN ? AND ?)",
                               (bot.event_no, pid, curr_time - 950, curr_time - 850)).fetchone()
            p_30 = cur.execute("SELECT pt FROM liverank WHERE eid = ? AND pid = ? AND (time BETWEEN ? AND ?)",
                               (bot.event_no, pid, curr_time - 1850, curr_time - 1750)).fetchone()
            p_60 = cur.execute("SELECT pt FROM liverank WHERE eid = ? AND pid = ? AND (time BETWEEN ? AND ?)",
                               (bot.event_no, pid, curr_time - 3650, curr_time - 3550)).fetchone()
        else:
            pname = ""
            p_15 = cur.execute("SELECT pt FROM liverank WHERE eid = ? AND rank = ? AND (time BETWEEN ? AND ?)",
                               (bot.event_no, rank, curr_time - 950, curr_time - 850)).fetchone()
            p_30 = cur.execute("SELECT pt FROM liverank WHERE eid = ? AND rank = ? AND (time BETWEEN ? AND ?)",
                               (bot.event_no, rank, curr_time - 1850, curr_time - 1750)).fetchone()
            p_60 = cur.execute("SELECT pt FROM liverank WHERE eid = ? AND rank = ? AND (time BETWEEN ? AND ?)",
                               (bot.event_no, rank, curr_time - 3650, curr_time - 3550)).fetchone()

        try:
            speed_15 = (curr_p - p_15[0])*4
        except TypeError:
            speed_15 = "沒有記錄"
        try:
            speed_30 = (curr_p - p_30[0])*2
        except TypeError:
            speed_30 = "沒有記錄"
        try:
            speed_60 = curr_p - p_60[0]
        except TypeError:
            speed_60 = "沒有記錄"

        embed = discord.Embed(title=f"**T{rank} {pname}**",
                              description=f"分數: {curr_p}"
                                          f"\n- 時速: {speed_60}"
                                          f"\n- 30min*2時速: {speed_30}"
                                          f"\n- 15min*4時速: {speed_15}",
                              colour=0xFFDD45)
        embed.set_footer(text=f"最後更新: {datetime.datetime.fromtimestamp(curr_time, tz=hktz).strftime('%m/%d %H:%M:%S')}")
    await ctx.send(embed=embed)


# ----- databases -----

@bot.command(name="create", hidden=True)
async def create(ctx, db):
    if not db:
        cur.execute("CREATE TABLE timept500(eid, time, pt, "
                    "PRIMARY KEY(eid, time)"
                    ")")
        cur.execute("CREATE TABLE timept1000(eid, time, pt, "
                    "PRIMARY KEY(eid, time)"
                    ")")
        cur.execute("CREATE TABLE accuracy(eid, predict500, actual500, diff500, predict1000, actual1000, diff1000, "
                    "PRIMARY KEY(eid)"
                    ")")
        cur.execute("CREATE TABLE liverank(eid, time, rank, pt, pid, pname, "
                    "PRIMARY KEY(eid, time, rank, pt)"
                    ")")
    elif db == "timept500":
        cur.execute("CREATE TABLE timept500(eid, time, pt, "
                    "PRIMARY KEY(eid, time)"
                    ")")
    elif db == "timept1000":
        cur.execute("CREATE TABLE timept1000(eid, time, pt, "
                    "PRIMARY KEY(eid, time)"
                    ")")
    elif db == "accuracy":
        cur.execute("CREATE TABLE accuracy(eid, predict500, actual500, diff500, predict1000, actual1000, diff1000, "
                    "PRIMARY KEY(eid)"
                    ")")
    elif db == "liverank":
        cur.execute("CREATE TABLE liverank(eid, time, rank, pt, pid, pname, "
                    "PRIMARY KEY(eid, time, rank, pt)"
                    ")")
    else:
        await ctx.send("Incorrect db name (timept500/1000, accuracy, liverank)")
        return
    await ctx.send("Db created successfully")

    """
    cur.execute("CREATE TABLE tracker(time, rank, pt, "
                "PRIMARY KEY(time, rank)"
                ")")
    """


@bot.command(name="q", hidden=True)
async def q(ctx, *, sql):
    if ctx.message.author.id not in [598066719659130900, 998784261249310740]:
        return
    res = cur.execute(sql)
    res = res.fetchall()
    con.commit()
    try:
        print(res)
        await ctx.send(res)
    except Exception as e:
        print(e)


@bot.command(name="insert_past_events", hidden=True)
async def insert_past_events(ctx, start: int, end: int):
    async with ctx.typing():
        for rank in [500, 1000]:
            print(f"----- T{rank} -----")
            # event = 36
            event = start
            while event <= end:  # while event < 71
                if event in [40, 42, 54, 55, 56, 58] or 67 <= event <= 116:
                    event += 1
                    continue
                else:
                    print(f"inserting event {event}")
                    data = past_event(event, rank)
                    for i in data:
                        try:
                            if rank == 500:
                                cur.execute("INSERT INTO timept500 VALUES (?, ?, ?)", (event, i[0], i[1]))
                            else:
                                cur.execute("INSERT INTO timept1000 VALUES (?, ?, ?)", (event, i[0], i[1]))
                            con.commit()
                        except sqlite3.IntegrityError:
                            continue
                    event += 1
    await ctx.send("Insert complete")


@bot.command(name="insert_0", hidden=True)
async def insert_0(ctx):
    eids = cur.execute("SELECT DISTINCT eid FROM timept500").fetchall()
    eids = [e[0] for e in eids]
    for i in eids:
        try:
            cur.execute("INSERT INTO timept500 VALUES (?, ?, ?)", (i, 0, 0))
            con.commit()
            cur.execute("INSERT INTO timept1000 VALUES (?, ?, ?)", (i, 0, 0))
            con.commit()
        except sqlite3.IntegrityError:
            continue
    await ctx.send("insert (0, 0) complete")


@bot.command(name="fix_data", hidden=True)
async def fix_data(ctx):
    try:
        cur.execute("INSERT INTO timept500 VALUES (41, 13.5, 250000), (41, 45, 500000), (41, 74.25, 750000)")
        cur.execute("INSERT INTO timept1000 VALUES (41, 23.5, 250000), (41, 54.75, 500000), (41, 90, 750000)")
    except sqlite3.IntegrityError:
        pass
    try:
        cur.execute("INSERT INTO timept500 VALUES (43, 12, 250000)")
    except sqlite3.IntegrityError:
        pass
    try:
        cur.execute("INSERT INTO timept500 VALUES (45, 12, 470000)")
        cur.execute("INSERT INTO timept1000 VALUES (45, 12, 257140)")
    except sqlite3.IntegrityError:
        pass
    try:
        cur.execute("INSERT INTO timept500 VALUES (57, 12, 500000)")
        cur.execute("INSERT INTO timept1000 VALUES (57, 12, 357000)")
    except sqlite3.IntegrityError:
        pass
    try:
        cur.execute("INSERT INTO timept500 VALUES (77, 3.7, 220000), (77, 7.7, 345000)")
        cur.execute("INSERT INTO timept1000 VALUES (77, 7.5, 241000), (77, 8.83, 271000)")
    except sqlite3.IntegrityError:
        pass
    con.commit()
    await ctx.send("Missing data filled")


# ----- test loops -----
@bot.command(name="test_predict", hidden=True)
async def test_predict(ctx):
    def calc_mape(prev, curr):
        return np.mean(np.abs((prev - curr) / prev))

    async with ctx.typing():
        now = datetime.datetime.now(hktz)
        start_hrs = math.floor((now - bot.start_day).total_seconds() / (60 * 60))
        total_hrs = (bot.end_day - bot.start_day).total_seconds() / (60 * 60)
        prev_eids = cur.execute("SELECT distinct eid FROM timept500").fetchall()
        prev_eids = [int(i[0]) for i in prev_eids]
        # prev_eids = [36, 37, 38, 39, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 57, 59, 60, 61, 62, 63, 64, 65, 66]

        # ----- T500 prediction -----
        curr_500 = list(past_event(bot.event_no, 500))
        # curr_500 = cur.execute("SELECT time, pt FROM timept500 WHERE eid = ?", (bot.event_no,)).fetchall()
        curr_500_x = [i[0] for i in curr_500]
        curr_500_y = [i[1] for i in curr_500]
        last_500_x = curr_500_x[-1]
        ts = np.arange(1, math.floor(last_500_x) + 1, 4)  # hrs 1, 5, 9... to last value before last_x
        if last_500_x != ts[-1]:
            ts = np.append(ts, last_500_x)
        ts_subset = ts[2:]  # from hr = 9
        curr_500_interp = np.interp(ts, curr_500_x, curr_500_y)  # get values at each timestep in ts
        mapes = []
        for e in prev_eids:
            prev_500 = cur.execute("SELECT time, pt FROM timept500 WHERE eid = ?", (e,)).fetchall()
            prev_x = [i[0] for i in prev_500]
            prev_y = [i[1] for i in prev_500]
            prev_500_interp = np.interp(ts, prev_x, prev_y)
            mape = calc_mape(prev_500_interp, curr_500_interp)
            mapes.append(mape)
        event_mapes = zip(prev_eids, mapes)
        event_mapes = [[i[0], i[1]] for i in event_mapes]
        event_mapes = sorted(event_mapes, key=lambda x: x[1])
        ms_event = event_mapes[0][0]  # most similar event
        s_events = [event_mapes[0][0], event_mapes[1][0], event_mapes[2][0]]  # most similar 3 events
        acc = get_acc(ms_event,
                      500)  # accumulation change time and magnitude [[acc_time], [acc_score]] during final rush
        if acc == KeyError:
            print("most similar error - trying 2nd most similar")
            print(s_events[1])
            acc = get_acc(s_events[1], 500)
            if acc == KeyError:
                print("2nd similar error - trying 3rd most similar")
                print(s_events[2])
                acc = get_acc(s_events[2], 500)
        acc = [[total_hrs - i[0], i[1]] for i in
               acc]  # [[hr from start, pt increase to next acc], ... [hr, pt increase to final]]
        # ----- extrapolate -----
        acc_hr = acc[0][0]  # first acc time, hours from start
        curr_500_subset = curr_500_interp[2:]  # pt at hr = 9 to now
        extrap_500 = np.poly1d(
            np.polyfit(ts_subset, curr_500_subset, 1))  # extrapolate from hr 9 (D1 speed often faster)
        acc_pt = extrap_500(acc_hr).item()
        # ----- add acc data -----
        extrap_500_x = [acc_hr]
        extrap_500_y = [acc_pt]
        last_pt = acc_pt
        final_pt = 0
        for i in range(len(acc)):
            if i != len(acc) - 1:
                extrap_500_x.append(acc[i + 1][0])
                extrap_500_y.append(last_pt + acc[i][1])
                last_pt += acc[i][1]
            else:
                extrap_500_x.append(total_hrs)
                final_pt = last_pt + acc[i][1]
                extrap_500_y.append(final_pt)
        if start_hrs <= total_hrs - 8:
            predict_500_x = [curr_500_x[-1]] + extrap_500_x
            predict_500_y = [curr_500_y[-1]] + extrap_500_y
        else:
            predict_500_x = extrap_500_x
            predict_500_y = extrap_500_y

        # ----- plot (current + prediction + most similar + 2 others) -----
        colours = ["darkblue", "royalblue", "lightskyblue"]
        fig, ax = plt.subplots()
        ax.grid(which='minor', linestyle=':', linewidth=0.4, color='#dbdbdb', axis="y")
        ax.grid(which='major', linestyle='-', linewidth=0.45, color='#dbdbdb', axis="y")
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        for x in [9, 33, 57, 81, 105, 129, 153, 177, 201, 225, 249]:
            ax.axvline(x, color='#b8b8b8', linewidth=0.75, linestyle='-')
        ci = 0
        for i in s_events:
            timept = cur.execute("SELECT time, pt FROM timept500 WHERE eid = ? ORDER BY time", (i,)).fetchall()
            if timept:
                x = [r[0] for r in timept]
                y = [r[1] for r in timept]
                ax.plot(x, y, label=str(i), linewidth=1, color=colours[ci])
                ci += 1
            else:
                continue
        ax.plot(curr_500_x, curr_500_y, label="Current", linewidth=1.2, color="r")
        ax.plot(predict_500_x, predict_500_y, label="Prediction", linewidth=1.25, color="r", ls="--")
        ax.legend(title="Event", alignment="left", labelspacing=0.2, fontsize="x-small", title_fontsize="small", loc=2)
        ax.set_title("T500 分數線預測")
        ax.set_xlabel("Time (hrs since start, Dday at 00)")
        ax.set_ylabel("Points")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        fig.savefig("predict500.png", dpi=600, bbox_inches='tight', pad_inches=0.25)
        await ctx.send("**__" + now.strftime("%m/%d %H:%M") + " 結活分數預測__**")
        await ctx.send("T500: " + str(int(round(final_pt))))
        await ctx.send(file=discord.File('predict500.png'))
        if now.date() == bot.end_day.date() and now.hour == 20:
            bot.predict500 = int(round(final_pt))
        plt.clf()
        plt.close("all")

        curr_1000 = list(past_event(bot.event_no, 1000))
        # curr_1000 = cur.execute("SELECT time, pt FROM timept1000 WHERE eid = ?", (bot.event_no,)).fetchall()
        curr_1000_x = [i[0] for i in curr_1000]
        curr_1000_y = [i[1] for i in curr_1000]
        last_1000_x = curr_1000_x[-1]
        ts = np.arange(1, math.floor(last_1000_x) + 1, 4)
        if last_1000_x != ts[-1]:
            ts = np.append(ts, last_1000_x)
        print(ts)
        ts_subset = ts[2:]
        curr_1000_interp = np.interp(ts, curr_1000_x, curr_1000_y)
        mapes = []
        for e in prev_eids:
            prev_1000 = cur.execute("SELECT time, pt FROM timept1000 WHERE eid = ?", (e,)).fetchall()
            prev_x = [i[0] for i in prev_1000]
            prev_y = [i[1] for i in prev_1000]
            prev_1000_interp = np.interp(ts, prev_x, prev_y)
            mape = calc_mape(prev_1000_interp, curr_1000_interp)
            mapes.append(mape)
        event_mapes = zip(prev_eids, mapes)
        event_mapes = [[i[0], i[1]] for i in event_mapes]
        event_mapes = sorted(event_mapes, key=lambda x: x[1])
        ms_event = event_mapes[0][0]
        s_events = [event_mapes[0][0], event_mapes[1][0], event_mapes[2][0]]
        acc = get_acc(ms_event, 1000)
        if acc == KeyError:
            print("most similar error - trying 2nd most similar")
            print(s_events[1])
            acc = get_acc(s_events[1], 1000)
            if acc == KeyError:
                print("2nd similar error - trying 3rd most similar")
                print(s_events[2])
                acc = get_acc(s_events[2], 1000)
        acc = [[total_hrs - i[0], i[1]] for i in acc]
        # ----- extrapolate -----
        acc_hr = acc[0][0]  # first acc time, hours from start
        curr_1000_subset = curr_1000_interp[2:]  # pt at hr = 9 to now
        extrap_1000 = np.poly1d(np.polyfit(ts_subset, curr_1000_subset, 1))
        acc_pt = extrap_1000(acc_hr).item()
        # ----- add acc data -----
        extrap_1000_x = [acc_hr]
        extrap_1000_y = [acc_pt]
        last_pt = acc_pt
        final_pt = 0
        for i in range(len(acc)):
            if i != len(acc) - 1:
                extrap_1000_x.append(acc[i + 1][0])
                extrap_1000_y.append(last_pt + acc[i][1])
                last_pt += acc[i][1]
            else:
                extrap_1000_x.append(total_hrs)
                final_pt = last_pt + acc[i][1]
                extrap_1000_y.append(final_pt)
        predict_1000_x = [curr_1000_x[-1]] + extrap_1000_x
        predict_1000_y = [curr_1000_y[-1]] + extrap_1000_y
        # ----- plot (current + prediction + most similar + 2 others) -----
        colours = ["darkblue", "royalblue", "lightskyblue"]
        fig, ax = plt.subplots()
        ax.grid(which='minor', linestyle=':', linewidth=0.4, color='#dbdbdb', axis="y")
        ax.grid(which='major', linestyle='-', linewidth=0.45, color='#dbdbdb', axis="y")
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        for x in [9, 33, 57, 81, 105, 129, 153, 177, 201, 225, 249]:
            ax.axvline(x, color='#b8b8b8', linewidth=0.75, linestyle='-')
        ci = 0
        for i in s_events:
            timept = cur.execute("SELECT time, pt FROM timept1000 WHERE eid = ? ORDER BY time", (i,)).fetchall()
            if timept:
                x = [r[0] for r in timept]
                y = [r[1] for r in timept]
                ax.plot(x, y, label=str(i), linewidth=1, color=colours[ci])
                ci += 1
            else:
                continue
        ax.plot(curr_1000_x, curr_1000_y, label="Current", linewidth=1.2, color="r")
        ax.plot(predict_1000_x, predict_1000_y, label="Prediction", linewidth=1.25, color="r", ls="--")
        ax.legend(title="Event", alignment="left", labelspacing=0.2, fontsize="x-small", title_fontsize="small", loc=2)
        ax.set_title("T1000 分數線預測")
        ax.set_xlabel("Time (hrs since start, Dday at 00)")
        ax.set_ylabel("Points")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        fig.savefig("predict1000.png", dpi=600, bbox_inches='tight', pad_inches=0.25)
        await ctx.send("T1000: " + str(int(round(final_pt))))
        await ctx.send(file=discord.File('predict1000.png'))
        if now.date() == bot.end_day.date() and now.hour == 20:
            bot.predict1000 = int(round(final_pt))
        plt.clf()
        plt.close("all")


@bot.command(hidden=True)
async def date(ctx):
    now = datetime.datetime.now(hktz)
    await ctx.send(f"now: {now.strftime('%d/%m %H:%M:%S')}\n"
                   f"event start: {bot.end_day.strftime('%d/%m %H:%M:%S')}\n"
                   f"event end: {bot.end_day.strftime('%d/%m %H:%M:%S')}")


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, CommandNotFound):
        return
    raise error


"""
@tasks.loop(minutes=15, reconnect=True)  # 00, 15, 30, 45
async def reminder():
    channel = bot.get_channel(1153702450365210634)
    now = datetime.datetime.now() + datetime.timedelta(hours=8)
    start_hrs = math.floor((now - bot.start_day).total_seconds() / (60 * 60))
    # ----- start loops -----
    if start_hrs == 33:  # D3 00
        if now.minute == 15:  # D3 0015
            await asyncio.sleep(60)  # D3 0016
            predict.start()
            await bot.get_channel(1007203228515057687).send("predict loop started")
    if now.date() == bot.start_day.date():
        if now.hour == 15 and now.minute == 15:  # D1 1515
            track.start()
            await bot.get_channel(1007203228515057687).send("track loop started")
    # ----- reminders -----
    elif now.date() == bot.end_day.date():  # last day
        if now.hour in [0, 20] and now.minute == 0:
            await channel.send("<@&1177391192602849390> 請在10分鐘内上報分數 <:ln_saki_excited:1011509870081626162>")
    elif bot.start_day.date() < now.date() < bot.end_day.date():  # middle days
        if now.minute == 0 and now.hour == 0:
            await channel.send("<@&1177391192602849390> 請在10分鐘内上報分數 <:ln_saki_excited:1011509870081626162>")
    else:
        return
        
@reminder.error
async def reminder_exited(self):
    print(self)
    channel = bot.get_channel(1007203228515057687)
    await channel.send("reminder loop error <@598066719659130900>, error:")
    await channel.send(self)
    traceback.print_exc()
    
@bot.command(name="end_reminder")
async def end_reminder(ctx):
    if reminder.is_running():
        reminder.cancel()
        await ctx.send("Reminder loop stopped")

@tasks.loop(minutes=30, reconnect=True)  # every 15, 45
async def track():
    channel = bot.get_channel(1177475155929342023)
    now = datetime.datetime.now() + datetime.timedelta(hours=8)
    # D1: 22 / D2-DL-1: 00, 12, 22 / DL-1: 00, 12, 20
    if now.hour not in [0, 20] or now.minute != 15:
        return
    if now.date() > bot.end_day.date():
        return
    if now.date() < bot.end_day.date() and now.hour == 20:
        return
    if now.date() == bot.start_day.date() and now.hour == 0:
        return
    start_hrs = math.floor((now - bot.start_day).total_seconds() / (60 * 60))
    data = cur.execute("SELECT rank, pt FROM tracker WHERE time = ? ORDER BY rank", (start_hrs,)).fetchall()
    if len(data) < 2:
        await channel.send("**__" + now.strftime("%m/%d %H") + ":00 分數線估算__**")
        await channel.send("數據不足 無法估算分數")
        return
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    spl = pchip(x, y)

    await channel.send("**__" + now.strftime("%m/%d %H") + ":00 分數線估算__**")
    min_rank = min(x)
    max_rank = max(x)

    x2 = np.linspace(min_rank, max_rank, num=1000)
    plt.plot(spl(x2), x2, "k-", linewidth=1)
    plt.plot(y, x, "o", markersize=2.5, mec="darkorchid", mfc="darkorchid")

    if min_rank <= 500:
        pt_500 = round(float(spl(500)))
        cur.execute("INSERT INTO timept500 VALUES (?, ?, ?)", (bot.event_no, start_hrs, pt_500))
        con.commit()
        await channel.send("T500: " + str(pt_500))
        plt.axvline(x=pt_500, lw=0.8, linestyle="--", c="red")
        plt.axhline(y=500, lw=0.8, linestyle="--", c="red")
    else:
        await channel.send("數據不足 無法估算T500分數")
    if max_rank >= 1000:
        pt_1000 = round(float(spl(1000)))
        cur.execute("INSERT INTO timept1000 VALUES (?, ?, ?)", (bot.event_no, start_hrs, pt_1000))
        con.commit()
        await channel.send("T1000: " + str(pt_1000))
        plt.axvline(x=pt_1000, lw=0.8, linestyle="--", c="mediumblue")
        plt.axhline(y=1000, lw=0.8, linestyle="--", c="mediumblue")
    else:
        await channel.send("數據不足 無法估算T1000分數")

    plt.title(now.strftime("%m/%d %H") + ":00 分數線估算")
    plt.xlabel("Points")
    plt.ylabel("Rank")
    ax = plt.gca()
    ax.invert_yaxis()
    plt.savefig("tracker.png", dpi=600, bbox_inches='tight', pad_inches=0.25)
    await channel.send(file=discord.File('tracker.png'))
    plt.clf()
    plt.close("all")

@track.error
async def track_exited(self):
    print(self)
    channel = bot.get_channel(1007203228515057687)
    await channel.send("track loop error <@598066719659130900>, error:")
    await channel.send(self)
    traceback.print_exc()

@bot.command(name="start_track", hidden=True)  # minutes=30, every 15, 45
async def start_track(ctx):
    if not track.is_running():
        now = datetime.datetime.now() + datetime.timedelta(hours=8)
        if now.minute < 15:
            seconds = (60 - now.second) + (15 - now.minute - 1) * 60
        elif now.minute >= 45:
            seconds = (60 - now.second) + (60 + 15 - now.minute - 1) * 60
        else:
            seconds = (60 - now.second) + (45 - now.minute - 1) * 60
        await ctx.send("Track loop starting in " + str(seconds) + " seconds")
        await asyncio.sleep(seconds)
        track.start()
        now = datetime.datetime.now() + datetime.timedelta(hours=8)
        await ctx.send("Track loop started at " + now.strftime("%H:%M:%S"))

@bot.command(name="end_track")
async def end_track(ctx):
    if track.is_running():
        track.cancel()
        await ctx.send("Track loop stopped")

@bot.command(name="clear_tracker", hidden=True)
async def clear_track(ctx):
    cur.execute("DELETE FROM tracker")
    con.commit()
    await ctx.send("tracker db cleared")

@bot.command(name="start")
async def start(ctx, event_no: int):
    bot.event_no = event_no
    response = urllib.request.urlopen("https://sekai-world.github.io/sekai-master-db-tc-diff/events.json")
    events_json = json.loads(response.read())
    for i in events_json:
        if i["id"] == event_no:
            start_dt = (i["startAt"]) / 1000
            end_dt = (i["aggregateAt"]) / 1000 + 1  # 21:00 HKT
            break
    bot.start_day = datetime.datetime.fromtimestamp(start_dt, hktz)
    bot.end_day = datetime.datetime.fromtimestamp(end_dt, hktz)

    try:
        cur.execute("INSERT INTO timept500 VALUES (?, 0, 0)", (bot.event_no, ))
        cur.execute("INSERT INTO timept1000 VALUES (?, 0, 0)", (bot.event_no,))
    except sqlite3.IntegrityError:
        pass

    if not reminder.is_running():
        await ctx.send("成功更新活動資料\n"
                       "活動: " + str(event_no) +
                       "\n開活時間: " + bot.start_day.strftime("%Y/%m/%d %H:%M") +
                       "\n結活時間: " + bot.end_day.strftime("%Y/%m/%d %H:%M"))
        now = datetime.datetime.now()
        if now.minute < 15:
            seconds = (60 - now.second) + (15 - now.minute - 1) * 60
        elif 15 <= now.minute < 30:
            seconds = (60 - now.second) + (30 - now.minute - 1) * 60
        elif 30 <= now.minute < 45:
            seconds = (60 - now.second) + (45 - now.minute - 1) * 60
        else:
            seconds = (60 - now.second) + (60 - now.minute - 1) * 60
        await ctx.send("Loop starting in " + str(seconds) + " seconds")
        await asyncio.sleep(seconds)
        if not reminder.is_running():
            reminder.start()
            await ctx.send("Reminder loop successfully started at " + (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%H:%M:%S"))
    else:
        await ctx.send("Loop already started\n成功更新活動資料\n"
                       "活動: " + str(event_no) +
                       "\n開活時間: " + bot.start_day.strftime("%Y/%m/%d %H:%M") +
                       "\n結活時間: " + bot.end_day.strftime("%Y/%m/%d %H:%M"))

@bot.command("test_track", hidden=True)
async def test_track(ctx, time: float):
    if ctx.author.id != 598066719659130900:
        return
    async with ctx.typing():
        # now = datetime.datetime.now() + datetime.timedelta(hours=8)
        # start_hrs = math.floor((bot.end_day - bot.start_day).total_seconds() / (60 * 60))
        pt_500 = 0
        pt_1000 = 0
        data = cur.execute("SELECT rank, pt FROM tracker WHERE time = ? ORDER BY rank", (time,)).fetchall()
        x = [i[0] for i in data]
        y = [i[1] for i in data]
        spl = pchip(x, y, extrapolate=True)
        min_rank = min(x)
        max_rank = max(x)
        if min_rank <= 500 and max_rank >= 1000:
            pt_500 = round(float(spl(500)))
            pt_1000 = round(float(spl(1000)))
            try:
                cur.execute("INSERT INTO timept500 VALUES (?, ?, ?)", (bot.event_no, time, pt_500))
                con.commit()
                cur.execute("INSERT INTO timept1000 VALUES (?, ?, ?)", (bot.event_no, time, pt_1000))
                con.commit()
            except sqlite3.IntegrityError:
                cur.execute("UPDATE timept500 SET pt = ? WHERE eid = ? AND time = ?", (pt_500, bot.event_no, time))
                con.commit()
                cur.execute("UPDATE timept1000 SET pt = ? WHERE eid = ? AND time = ?", (pt_1000, bot.event_no, time))
                con.commit()
        elif min_rank > 500 and max_rank < 1000:
            await ctx.send("分數上報紀錄不足以做估算 <:ln_saki_weapon:1006929901745614859>")
            return
        elif min_rank > 500:
            pt_1000 = round(float(spl(1000)))
            try:
                cur.execute("INSERT INTO timept1000 VALUES (?, ?, ?)", (bot.event_no, time, pt_1000))
                con.commit()
            except sqlite3.IntegrityError:
                cur.execute("UPDATE timept1000 SET pt = ? WHERE eid = ? AND time = ?", (pt_1000, bot.event_no, time))
                con.commit()
        elif max_rank < 1000:
            pt_500 = round(float(spl(500)))
            try:
                cur.execute("INSERT INTO timept500 VALUES (?, ?, ?)", (bot.event_no, time, pt_500))
                con.commit()
            except sqlite3.IntegrityError:
                cur.execute("UPDATE timept500 SET pt = ? WHERE eid = ? AND time = ?", (pt_500, bot.event_no, time))
                con.commit()

        title = bot.start_day + datetime.timedelta(hours=time)
        if str(time)[-2:] == ".5":
            await ctx.send("**__" + title.strftime("%m/%d %H") + ":30 分數線估算__**")
        else:
            await ctx.send("**__" + title.strftime("%m/%d %H") + ":00 分數線估算__**")
        if pt_500 != 0 and pt_1000 !=0:
            await ctx.send("T500: " + str(pt_500))
            await ctx.send("T1000: " + str(pt_1000))
        elif pt_500 == 0:
            await ctx.send("數據不足 無法估算T500分數")
            await ctx.send("T1000: " + str(pt_1000))
        else:
            await ctx.send("T500: " + str(pt_500))
            await ctx.send("數據不足 無法估算T1000分數")
        x2 = np.linspace(min_rank, max_rank, num=1000)

        plt.plot(spl(x2), x2, "k-", linewidth=1)
        plt.plot(y, x, "o", markersize=2.5, mec="darkorchid", mfc="darkorchid")
        if pt_1000 != 0:
            plt.axvline(x=pt_1000, lw=0.8, linestyle="--", c="mediumblue")
            plt.axhline(y=1000, lw=0.8, linestyle="--", c="mediumblue")
        if pt_500 != 0:
            plt.axvline(x=pt_500, lw=0.8, linestyle="--", c="red")
            plt.axhline(y=500, lw=0.8, linestyle="--", c="red")
        if str(time)[-2:] == ".5":
            plt.title(title.strftime("%m/%d %H") + ":30 分數線估算")
        else:
            plt.title(title.strftime("%m/%d %H") + ":00 分數線估算")
        plt.xlabel("Points")
        plt.ylabel("Rank")
        ax = plt.gca()
        ax.invert_yaxis()
        plt.savefig("tracker.png", dpi=600, bbox_inches='tight', pad_inches=0.25)
        await ctx.send(file=discord.File('tracker.png'))
        plt.clf()
        plt.close("all")

@bot.command(name="end", hidden=True)
async def end(ctx):
    if ctx.author.id != 598066719659130900:
        return
    total_hrs = (bot.end_day - bot.start_day).total_seconds() / (60 * 60)
    data = cur.execute("SELECT rank, pt FROM tracker WHERE time = ? ORDER BY rank", (total_hrs,)).fetchall()
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    spl = pchip(x, y)
    pt_500 = round(float(spl(500)))
    pt_1000 = round(float(spl(1000)))
    try:
        cur.execute("INSERT INTO timept500 VALUES (?, ?, ?)", (bot.event_no, total_hrs, pt_500))
        con.commit()
        cur.execute("INSERT INTO timept1000 VALUES (?, ?, ?)", (bot.event_no, total_hrs, pt_1000))
        con.commit()
    except sqlite3.IntegrityError:
        cur.execute("UPDATE timept500 SET pt = ? WHERE eid = ? AND time = ?", (pt_500, bot.event_no, total_hrs))
        con.commit()
        cur.execute("UPDATE timept1000 SET pt = ? WHERE eid = ? AND time = ?", (pt_1000, bot.event_no, total_hrs))
        con.commit()

    await ctx.send("**__結活分數線估算__**")
    await ctx.send("T500: " + str(pt_500))
    await ctx.send("T1000: " + str(pt_1000))
    min_rank = min(x)
    max_rank = max(x)
    if min_rank > 500:
        min_rank = 500
    if max_rank < 1000:
        max_rank = 1000
    x2 = np.linspace(min_rank, max_rank, num=1000)

    plt.plot(spl(x2), x2, "k-", linewidth=1)
    plt.plot(y, x, "o", markersize=2.5, mec="darkorchid", mfc="darkorchid")
    plt.axvline(x=pt_1000, lw=0.8, linestyle="--", c="mediumblue")
    plt.axhline(y=1000, lw=0.8, linestyle="--", c="mediumblue")
    plt.axvline(x=pt_500, lw=0.8, linestyle="--", c="red")
    plt.axhline(y=500, lw=0.8, linestyle="--", c="red")
    plt.title("結活分數線估算")
    plt.xlabel("Points")
    plt.ylabel("Rank")
    ax = plt.gca()
    ax.invert_yaxis()
    plt.savefig("tracker.png", dpi=600, bbox_inches='tight', pad_inches=0.25)
    await ctx.send(file=discord.File('tracker.png'))
    plt.clf()
    plt.close("all")

    try:
        cur.execute("INSERT INTO accuracy VALUES (?, ?, ?, ?, ?, ?, ?)", (bot.event_no,
                                                                          bot.predict500, pt_500, bot.predict500 - pt_500,
                                                                          bot.predict1000, pt_1000, bot.predict1000 - pt_1000))
        con.commit()
    except sqlite3.IntegrityError:
        cur.execute("UPDATE accuracy SET predict500 = ?,  predict1000 = ? , diff500 = ?, diff1000 = ? WHERE eid = ?",
                    (bot.predict500, bot.predict1000, bot.predict500 - pt_500, bot.predict1000 - pt_1000, bot.event_no))
        con.commit()

    bot_channel = bot.get_channel(1007203228515057687)
    await bot_channel.send("Insertion into accuracy db completed")
    upload_channel = bot.get_channel(1153702450365210634)
    await upload_channel.send("----- " + str(bot.event_no) + "期分數上報結束 <:ln_saki_otsu:1006480191431909457> -----")
    if reminder.is_running():
        reminder.cancel()
    if track.is_running():
        track.cancel()
    if predict.is_running():
        predict.cancel()
    await bot_channel.send("All loops stopped")
"""


bot.run(TOKEN)


