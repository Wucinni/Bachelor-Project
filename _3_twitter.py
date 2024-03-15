import requests

bearer_token = "AAAAAAAAAAAAAAAAAAAAAOqPkAEAAAAAcMGpCYHaF4wM%2BCQWB1mxM%2FqnmXQ%3DksBEK38YLjesHrHQDg2qKRU4sEj2WHZkCPW9KRJFaK2ENuNdqo"


def retrieve_tweets(query_input):
    if query_input is None or len(str(query_input)) == 0:
        query = "review lang:en -is:retweet"
    else:
        query = str(query_input) + " lang:en -is:retweet"

    print(query)

    url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results=10&tweet.fields=lang"

    headers = {
        "Authorization": f"Bearer {bearer_token}",
        #"User-Agent": "v2FilteredStreamPython"
        "User-Agent": "v2TweetLookupPython"
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(
            f"Request returned an error: {response.status_code} {response.text}"
        )

    json_response = response.json()

    tweet_message = []
    for tweet in json_response['data']:
        print(tweet)
        tweet_message.append(tweet['text'])

    return tweet_message


def reply():
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    import time
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    tweet_id = '1661794226236006400'  # Replace with the actual tweet ID
    reply_text = 'api test'

    # Set up the Selenium driver (make sure to download the appropriate driver for your browser)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # Open Twitter in the browser
    driver.get('https://twitter.com')

    # Wait for the page to load
    time.sleep(2)

    # Find the login button and click it
    login_button = driver.find_element_by_xpath('//a[@data-testid="loginButton"]')
    login_button.click()

    # Wait for the login page to load
    time.sleep(2)

    # Find the username and password fields, and enter your credentials
    username_field = driver.find_element_by_name('session[username_or_email]')
    password_field = driver.find_element_by_name('session[password]')

    username_field.send_keys('YourTwitterUsername')
    password_field.send_keys('YourTwitterPassword')

    # Submit the login form
    password_field.send_keys(Keys.RETURN)

    # Wait for the home page to load after login
    time.sleep(2)

    # Search for the tweet by its ID
    search_input = driver.find_element_by_css_selector('[data-testid="SearchBox_Search_Input"]')
    search_input.send_keys('https://twitter.com/anyuser/status/' + tweet_id)
    search_input.send_keys(Keys.RETURN)

    # Wait for the search results to load
    time.sleep(2)

    # Click on the tweet to open it
    tweet_link = driver.find_element_by_css_selector('[href="https://twitter.com/anyuser/status/' + tweet_id + '"]')
    tweet_link.click()

    # Wait for the tweet page to load
    time.sleep(2)

    # Find the reply input field and enter the reply text
    reply_input = driver.find_element_by_css_selector('[data-testid="tweetTextarea_0"]')
    reply_input.send_keys(reply_text)

    # Find the reply button and click it
    reply_button = driver.find_element_by_css_selector('[data-testid="tweetButtonInline"]')
    reply_button.click()

    # Wait for the reply to be posted
    time.sleep(2)

    # Close the browser
    driver.quit()
