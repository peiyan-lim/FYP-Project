from flask import Flask, render_template, url_for, request, abort
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField
from wtforms.validators import DataRequired
import modules

app = Flask(__name__)

# Flask-WTF requires an enryption key - the string can be anything
app.config['SECRET_KEY'] = 'C2HWGVoMGfNTBsrYQg8EcMrdTimkZfAb'

# Flask-Bootstrap requires this line
Bootstrap(app)

class NameForm(FlaskForm):
    msg = TextAreaField("", render_kw={"placeholder": "Enter text"}, validators=[DataRequired()])
    submit = SubmitField('Check Sentiment')

# Define the root route
@app.route('/')
def index():
    try:
        return render_template('index.html')
    except IndexError:
        abort(404)

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.route('/500')
def error500():
    abort(500)

@app.route('/callback/<endpoint>')
def cb(endpoint):   
    data_response = modules.get_data(request.args.get('data'))
    df = modules.json_to_df(data_response)
    #modules.get_wordcloud(df,"Positive", request.args.get('data'))
    #modules.get_wordcloud(df,"Negative", request.args.get('data'))
    if endpoint == "getData": 
        scatter = modules.get_scatter(df)
        return scatter
    elif endpoint == "getPie":
        pie = modules.get_pie(df)
        return pie
    elif endpoint == "getInfo":
        data = request.args.get('data')
        tablePos = modules.top_3(df, "Positive")
        tableNeg = modules.top_3(df, "Negative")
        startDate = df['Post Created Date'].min()
        endDate = df['Post Created Date'].max()
        jsonDate = {"data": data, "startDate": startDate.strftime('%Y-%m-%d'), "endDate":endDate.strftime('%Y-%m-%d'), "htmltablePos": tablePos, "htmltableNeg": tableNeg}
        return jsonDate
    elif endpoint == "getScatter":
        return modules.get_postbydate_scatter(request.args.get('data'))
    else:
        return "Bad endpoint", 400


@app.route('/sentiment-checker', methods=['GET', 'POST'])
def sentiment_checker():
    form = NameForm()
    message = ""
    if form.validate_on_submit():
        msg = form.msg.data
        sentiment = modules.chk_sentiment(msg)
        if sentiment == "error":
            message = "Server down, please try again later"
        else:
            message = sentiment
    return render_template("sentiment_checker.html", form=form, message=message)