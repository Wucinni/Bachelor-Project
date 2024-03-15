import praw

reddit_full_credentials = praw.Reddit(
        client_id='Bq406aMbmTGkdb53cuvi8w',
        client_secret='cRlmOxHwBW1yMo4WWUGPQ5hY-RewHw',
        user_agent='RedditCommentResponse',
        username='thesis_tester_99',
        password='DacaEValoare49'
    )

reddit_no_credentials = praw.Reddit(
        client_id="Bq406aMbmTGkdb53cuvi8w",
        client_secret="cRlmOxHwBW1yMo4WWUGPQ5hY-RewHw",
        user_agent="RedditPostRetrieval"
    )


def retrieve_posts(query_input):
    print("In Retrieve Post")
    if query_input is None or len(str(query_input)) == 0:
        query = "python"
    else:
        query = str(query_input)

    print("Getting Subreddit")
    subreddit = reddit_no_credentials.subreddit("all")

    print("Getting Post")
    posts = subreddit.search(query, limit=1)

    posts_titles = []
    posts_ids = []

    #for post in posts:
    #    print("\n\n", post.title, "\n\n")

    print("Searching in Posts")
    print(posts)
    for post in posts:
        posts_titles.append(post.title)
        print("Title:", post.title)
        posts_ids.append(post.id)
        print("ID:", post.id)

    return posts_ids


def retrieve_comments(query):
    print("In retrieve comments")
    post_id = retrieve_posts(query)
    print("POST ID:", post_id)

    submission = reddit_no_credentials.submission(id=post_id[0])

    submission.comments.replace_more(limit=0)

    print("Here")
    comments_bodies = []
    comments_ids = []

    counter = 0
    for comment in submission.comments.list()[0:1000]:
        if len(comment.body) <= 100 and comment.body != "[deleted]":
            comments_ids.append(comment.id)
            comments_bodies.append(comment.body)
            counter = counter + 1
            if counter == 10:
                break

    return comments_bodies, comments_ids


def respond_to_comment(comment_id, text):
    print("Replying")
    comment = reddit_full_credentials.comment(id=comment_id)
    comment.reply(text)

#value = retrieve_comments("Samsung")[0]
#print(type(value))
#i= 0
#import _4_preprocessing as p
#for j in value:
#    #print("\n", j)
#    print(type(p.text_cleaning(j)))
#    i = i+1
#print(i)
#a = [1, 2, 3, 4]
#print(a)