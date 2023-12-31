import json
import requests
import urllib.request
import numpy as np
import pandas as pd


def past_event(event, rank):
    api = "https://api.sekai.best/event/"
    query = "/rankings/graph?region=tw&rank="
    events_url = "https://sekai-world.github.io/sekai-master-db-tc-diff/events.json"
    response = urllib.request.urlopen(events_url)
    events_json = json.loads(response.read())

    res = json.loads(requests.get(api + str(event) + query + str(rank)).text)["data"]["eventRankings"]
    data = pd.DataFrame(res)
    timestamps = pd.to_datetime(data["timestamp"]).values.astype(np.int64) // 10 ** 6
    scores = data["score"].values

    start_time = 0
    for i in events_json:
        if i["id"] == event:
            start_time = i["startAt"]
            break
    timestamps = [(t - start_time)/1000/(60*60) for t in timestamps]  # no of seconds from start of event
    scores = scores.tolist()
    data = zip(timestamps, scores)
    return data






