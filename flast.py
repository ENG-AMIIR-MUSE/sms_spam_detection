from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "App working!"

@app.route('/home')
def dashboard():
    return "Home Test Flask First api "

if __name__ == '__main__':
    app.run(debug=True)
