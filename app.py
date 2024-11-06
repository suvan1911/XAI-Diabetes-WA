from flask import Flask
import xai_view

app = Flask(__name__)

app.add_url_rule('/xai', view_func=xai_view.index, methods=['GET', 'POST'])
app.add_url_rule('/', view_func=xai_view.index, methods=['GET', 'POST'])

if __name__ == '__main__':
    app.run()
