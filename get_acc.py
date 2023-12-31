import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.signal import argrelextrema
import sqlite3

api = "https://api.sekai.best/event/"
query = "/rankings/graph?region=tw&rank="

con = sqlite3.connect("pt_tracker.db")
cur = con.cursor()


def get_acc(event, rank):
    res = json.loads(requests.get(api + str(event) + query + str(rank)).text)["data"]["eventRankings"]
    data = pd.DataFrame(res)
    try:
        timestamps = pd.to_datetime(data["timestamp"]).values.astype(np.int64) // 10 ** 6
    except KeyError:
        return KeyError
    scores = data["score"].values
    end_pt = scores[-1]
    end_time = timestamps[-1]
    timestamps_subset = timestamps[-30:]  # 0.5hr per record, 30 = 15hrs (?)
    scores_subset = scores[-30:]
    """except KeyError:
        if rank == 500:
            db = cur.execute("SELECT time, pt FROM timept500 WHERE eid = ? ORDER BY time", (event, ))
        else:
            db = cur.execute("SELECT time, pt FROM timept1000 WHERE eid = ? ORDER BY time", (event, ))
        scores = [i[1] for i in db]
        timestamps = [i[0] for i in db]
        end_pt = scores[-1]
        end_time = timestamps[-1]
"""
    d = np.gradient(scores_subset, timestamps_subset)  # d
    d2 = np.gradient(d, timestamps_subset)  # second d
    d2_peak = argrelextrema(d2, np.greater_equal, order=3)[0]  # find peaks in d2 (changes in acc during final rush)

    acc_time = []
    acc_score = []  # delta pts between peaks
    i = 0
    for p in d2_peak:
        if i != 0:
            acc_time.append(timestamps_subset[p])
            acc_score.append(scores_subset[p] - scores_subset[d2_peak[i - 1]])
        else:
            acc_time.append(timestamps_subset[p])
        i += 1

    acc_time = [datetime.fromtimestamp(end_time/1000) - datetime.fromtimestamp(ts/1000) for ts in acc_time]  # /1000 to change from milliseconds to seconds
    acc_time = [ts.seconds / (60*60) for ts in acc_time]  # acc times in hours before end
    final_inc = end_pt - scores_subset[d2_peak[len(d2_peak) - 1]]  # pt increase from last peak to end
    acc_score.append(final_inc)
    data = zip(acc_time, acc_score)
    return data
