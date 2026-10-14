"""Run with:  python -m pytest tests   (from the university-clustering-app folder)"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['USERS_DB'] = os.path.join(tempfile.mkdtemp(), 'test_users.db')
import app as A


def client():
    A.app.config['TESTING'] = True
    return A.app.test_client()


def test_requires_login():
    assert client().get('/').status_code == 302


def test_signup_login_and_wrong_password():
    c = client()
    assert c.post('/signup', data={'username': 'ana', 'password': 'correct-horse'}).status_code == 302
    c.get('/logout')
    bad = c.post('/login', data={'username': 'ana', 'password': 'wrong-password'})
    assert b'Invalid username or password' in bad.data          # the original app let this through
    good = c.post('/login', data={'username': 'ana', 'password': 'correct-horse'})
    assert good.status_code == 302


def test_impute_tufts():
    c = client()
    c.post('/signup', data={'username': 'ben', 'password': 'another-pass'})
    r = c.post('/impute', json={'university': 'Tufts University'}).get_json()
    assert r['missing'] == ['# PT undergrad'] and r['cluster_label'] == 'Selective private universities'


def test_unknown_university():
    c = client()
    c.post('/signup', data={'username': 'cy', 'password': 'third-pass1'})
    assert c.post('/impute', json={'university': 'Hogwarts'}).status_code == 404
