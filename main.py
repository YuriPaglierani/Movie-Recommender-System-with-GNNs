import threading
from api.app import app as flask_app
from frontend.dashboard import app as dash_app

def run_flask():
    flask_app.run(port=8050)

def run_dash():
    dash_app.run_server(port=8050)

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask)
    dash_thread = threading.Thread(target=run_dash)

    flask_thread.start()
    dash_thread.start()

    flask_thread.join()
    dash_thread.join()