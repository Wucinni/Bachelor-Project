#############################################
#                                           #
#   This script manages the Twitter API     #
#                                           #
#      It handles request for posts         #
#                                           #
#############################################


import requests


# Private token used for authentication
bearer_token = "AAAAAAAAAAAAAAAAAAAAAOqPkAEAAAAAcMGpCYHaF4wM%2BCQWB1mxM%2FqnmXQ%3DksBEK38YLjesHrHQDg2qKRU4sEj2WHZkCPW9KRJFaK2ENuNdqo"


def retrieve_tweets(query_input):
    """
        Function will retrieve tweet posts using a query
        input - query; Type STR
        output - tweet text; Type STR
    """

    # Set query to a default query or to input
    query = "review lang:en -is:retweet" if (query_input is None or len(str(query_input)) == 0) else (str(
        query_input) + " lang:en -is:retweet")

    # Create URL endpoint for request using the input query
    url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results=10&tweet.fields=lang"

    # Create proper authentication
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "User-Agent": "v2TweetLookupPython"
    }

    # Send request using Twitter API
    response = requests.get(url, headers=headers)

    # If request fails raise error
    if response.status_code != 200:
        raise Exception(
            f"Request returned an error: {response.status_code} {response.text}"
        )

    # De-jsonify data in response
    json_response = response.json()

    tweet_message = []

    # Extract data from response
    for tweet in json_response['data']:
        tweet_message.append(tweet['text'])

    return tweet_message
