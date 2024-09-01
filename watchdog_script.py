import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(".py"):
            print(f"Detected change in {event.src_path}. Restarting Flask server...")
            # Terminate the current Flask process if running
            if hasattr(self, 'flask_process') and self.flask_process.poll() is None:
                self.flask_process.terminate()
            # Start a new Flask process
            self.flask_process = subprocess.Popen(["flask", "run"])

if __name__ == "__main__":
    path = "."  # Monitor the current directory
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    # Start the Flask server for the first time
    event_handler.flask_process = subprocess.Popen(["flask", "run"])

    try:
        while True:
            pass  # Keep the script running
    except KeyboardInterrupt:
        observer.stop()
        if event_handler.flask_process.poll() is None:
            event_handler.flask_process.terminate()

    observer.join()
