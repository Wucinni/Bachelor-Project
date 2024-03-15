#############################################
#                                           #
#           Main application                #
#                                           #
#   Run this to deploy flask server         #
#           and web interface               #
#                                           #
#############################################


from flask import Flask, render_template, request
import _3_reddit as reddit
import _4_tester as tester

# Data provided through Reddit API
reviews = []
tweets_text = []
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
    return render_template('prediction.html', tweets_text = tweets_text, reviews = reviews)


@app.route('/', methods=['POST', 'GET'])
def index():
    global reviews, tweets_text, comments_ids, model_selected

    if request.method == "POST":
        # Send selected model and query input to back-end
        if request.form.get('tester_loader') == 'Make Predictions':
            print("Started Prediction")
            model_selected = request.form['selected_value']
            output = tester.loader(model_selected, request.form['query'])
            print("Finished Prediction")
            comments_ids = output[2]

        # Send responses to comments where user input is not None
        if request.form.get('sender') == 'Send Responses':
            i = 0
            while request.form.get('response' + str(i)) is not None:
                if len(request.form.get('response' + str(i))) > 0:
                    reddit.respond_to_comment(comments_ids[i], request.form.get('response' + str(i)))
                i = i + 1

            return home()

        return prediction(tweets_text = output[1], reviews = output[0])

    return home()


app.run(debug=False, host='0.0.0.0')
