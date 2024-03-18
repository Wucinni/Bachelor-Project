#############################################
#                                           #
#            Main application               #
#                                           #
#     Run this to deploy flask server       #
#            and web interface              #
#                                           #
#############################################


import _3_reddit as reddit
import _4_tester as tester
from flask import Flask, render_template, request


# Data provided through Reddit API
comments_ids = []
# Model selected in the web interface
model_selected = None

STATIC_FOLDER = 'templates/assets'
app = Flask(__name__, static_folder=STATIC_FOLDER)


@app.route("/home")
def home():
    return render_template('home.html')


@app.route("/prediction")
def prediction(tweets_text, reviews):
    return render_template('prediction.html', tweets_text=tweets_text, reviews=reviews)


@app.route('/', methods=['POST', 'GET'])
def index():
    global comments_ids

    if request.method == "POST":

        # Send selected model and query input to back-end
        if request.form.get('run_model') == 'Make Predictions':
            # Get model from request
            model_selected = request.form['selected_value']
            # Call script to run model and analyze sentiments
            output = tester.run_model(model_selected, request.form['query'])
            # Extract comments ids for responses
            comments_ids = output[2]

        # Send responses to comments
        if request.form.get('sender') == 'Send Responses':
            comment_position = 0
            # Run while input is not empty
            while request.form.get('response' + str(comment_position)) is not None:
                # If comment is not empty send response with Reddit API
                if len(request.form.get('response' + str(comment_position))) > 0:
                    reddit.respond_to_comment(comments_ids[comment_position],
                                              request.form.get('response' + str(comment_position)))
                # Increment comment position to get next id
                comment_position = comment_position + 1
            return home()

        return prediction(tweets_text=output[1], reviews=output[0])

    return home()


app.run(debug=False, host='0.0.0.0')
