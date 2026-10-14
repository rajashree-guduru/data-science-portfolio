"""University data imputation web app (Flask).

Log in, type a university name, and the app fills its missing values from the nearest
K-means cluster centre (same model as the notebook: standardised features, k = 4).
Run:  python app.py   then open http://127.0.0.1:5000
"""
import os
import secrets
import sqlite3
from functools import wraps
from flask import Flask, g, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
import clustering as C

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or secrets.token_hex(32)   # never hard-code secrets
DATABASE = os.environ.get('USERS_DB', os.path.join(C.HERE, 'users.db'))

df, features = C.load()
scaler, km = C.fit(df, features)
C.name_clusters(C.profiles(df, features, km))
UNIVERSITIES = sorted(df['College Name'].dropna().unique())


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)')
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return view(*args, **kwargs)
    return wrapped


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or len(password) < 8:
            error = 'Enter a username and a password of at least 8 characters.'
        else:
            try:
                db = get_db()
                db.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, generate_password_hash(password)))
                db.commit()
                session['user'] = username
                return redirect(url_for('home'))
            except sqlite3.IntegrityError:
                error = 'That username is already taken.'
    return render_template('signup.html', error=error)


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        row = get_db().execute('SELECT password FROM users WHERE username = ?', (username,)).fetchone()
        if row and check_password_hash(row[0], password):      # the original version skipped this check
            session['user'] = username
            return redirect(url_for('home'))
        error = 'Invalid username or password.'
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/')
@login_required
def home():
    return render_template('index.html', user=session['user'], universities=UNIVERSITIES)


@app.route('/impute', methods=['POST'])
@login_required
def impute():
    name = (request.get_json(silent=True) or {}).get('university', '').strip()
    if not name:
        return jsonify(error='Enter a university name.'), 400
    try:
        result = C.impute(df, features, scaler, km, name)
    except KeyError:
        return jsonify(error=f'"{name}" was not found.'), 404
    if not result['missing']:
        result['message'] = 'This university has no missing values.'
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=False)
