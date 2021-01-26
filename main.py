"""
Run the application
"""
from src import app2 as app
import sys

if __name__ == "__main__":
    app.run()
    sys.exit(app.app.exec_())
