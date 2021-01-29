"""
Run the application
"""
import sys
import subprocess

generate_files = False

#generate_files = True

if __name__ == "__main__":
    if generate_files:
        import time
        subprocess.Popen(
            r"powershell.exe B:\boska\Documents\Dilan\Dropbox\Github\ultimatevocalremovergui\src\UIFileConverter.ps1", shell=True)
        time.sleep(4)
    from src import app
    app.run()
