import requests
import os

def retrieve_tweets():
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAOqPkAEAAAAA%2FjppBJBHiGf3cyds4poEBU956lk%3DypGk85lgeQUOM9ifgWweAI1ul6RF0az1hRkQX40NYxKB7c7HW8"

    query = "review"  # Replace with your own query
    query = "review lang:en"

    url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results=10&tweet.fields=lang"

    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "User-Agent": "v2FilteredStreamPython"
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(
            f"Request returned an error: {response.status_code} {response.text}"
        )

    json_response = response.json()

    tweet_message = []
    for tweet in json_response['data']:
        tweet_message.append(tweet['text'])
        break

    return tweet_message

#retrieve_tweets()