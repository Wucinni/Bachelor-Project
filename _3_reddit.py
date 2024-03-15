#############################################
#                                           #
#   This script manages the Reddit API      #
#                                           #
#  It handles request for post and comments #
#       retrieval and comments response     #
#                                           #
#############################################


import praw

# Praw object for Reddit Credentials
reddit_full_credentials = praw.Reddit(
        client_id='Bq406aMbmTGkdb53cuvi8w',
        client_secret='cRlmOxHwBW1yMo4WWUGPQ5hY-RewHw',
        user_agent='RedditCommentResponse',
        username='thesis_tester_99',
        password='DacaEValoare49'
    )

# Praw object for Reddit Credentials
reddit_no_credentials = praw.Reddit(
        client_id="Bq406aMbmTGkdb53cuvi8w",
        client_secret="cRlmOxHwBW1yMo4WWUGPQ5hY-RewHw",
        user_agent="RedditPostRetrieval"
    )


def retrieve_posts(query=None):
    """
        Function uses the Reddit API to retrieve a post using the query
        input - query; Type STR
        output - post id; Type INT
    """

    # Set query to a default query or to input
    query = "python" if (query is None or len(str(query)) == 0) else str(query)

    # Search key query in all subsections of Reddit
    subreddit = reddit_no_credentials.subreddit("all")

    # Search for the first post in the subreddit that includes the query in title
    posts = subreddit.search(query, limit=1)

    posts_titles = []
    posts_ids = []

    # Save all posts and their ids in lists
    for post in posts:
        posts_titles.append(post.title)
        posts_ids.append(post.id)

    return posts_ids


def retrieve_comments(query):
    """
        Function will retrieve comments from a post based on its id
        input - query to be passed to post id function; Type STR
        output - all comments texts and ids; Type Tuple of Lists
    """

    # Get post id from post function
    post_id = retrieve_posts(query)

    # Extract submission object from the post
    submission = reddit_no_credentials.submission(id=post_id[0])

    # Extract comments from submission
    submission.comments.replace_more(limit=0)

    comments_bodies = []
    comments_ids = []

    counter = 0 # Counter to check how many comments were retrieved
    # Extract all comments and their ids from the submission
    for comment in submission.comments.list()[0:1000]:
        # Limit comment size and don't retrieve delete notification
        if len(comment.body) <= 100 and comment.body != "[deleted]":
            comments_ids.append(comment.id)
            comments_bodies.append(comment.body)

            # Counter to limit the number of comments extracted
            counter = counter + 1
            if counter == 10:
                break

    return comments_bodies, comments_ids


def respond_to_comment(comment_id, text):
    """
        Function will respond to a comment using Reddit API
        input - comment_id; Type STR
              - text; Type STR
        output - None
    """
    # Create comment object
    comment = reddit_full_credentials.comment(id=comment_id)

    # Send reply
    comment.reply(text)
