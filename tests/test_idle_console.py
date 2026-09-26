"""Headless regression tests: python -m unittest discover -s tests -v."""

import ast
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
import unittest


class RecordingText:
    """Records widget effects without starting Tk or the application."""

    def __init__(self, master=None, **options):
        self.calls = []
        self.text = ""
        self.state = options.get("state", "normal")
        self.scheduled = []

    def configure(self, **options):
        self.calls.append(("configure", options))
        self.state = options["state"]

    def insert(self, index, text):
        self.calls.append(("insert", index, text))
        self.text += text

    def delete(self, start, end):
        self.calls.append(("delete", start, end))
        self.text = ""

    def see(self, index):
        self.calls.append(("see", index))

    def update_idletasks(self):
        self.calls.append(("update_idletasks",))

    def after(self, delay, callback):
        self.scheduled.append((delay, callback))


def load_console():
    # Execute the production class without UVR.py's imports and GUI startup.
    path = Path(__file__).resolve().parents[1] / "UVR.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    console = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "ThreadSafeConsole")
    namespace = {
        "queue": queue,
        "tk": SimpleNamespace(Text=RecordingText, NORMAL="normal",
                              DISABLED="disabled", END="end"),
    }
    exec(compile(ast.Module(body=[console], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["ThreadSafeConsole"]


class IdleConsoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.console_class = load_console()

    def setUp(self):
        self.console = self.console_class(None)
        self.console.calls.clear()
        self.console.scheduled.clear()

    def test_console_starts_read_only(self):
        self.assertEqual(self.console.state, "disabled")

    def test_idle_poll_does_not_mutate_widgets(self):
        for _ in range(20):
            self.console.update_me()
        self.assertEqual(self.console.calls, [])
        self.assertEqual(self.console.state, "disabled")
        self.assertEqual(len(self.console.scheduled), 20)
        for delay, callback in self.console.scheduled:
            self.assertEqual(delay, 100)
            self.assertEqual(callback, self.console.update_me)

    def test_batch_preserves_messages_and_updates_display_once(self):
        for line in ("first", 7, "\nlast"):
            self.console.write(line)
        self.console.update_me()
        self.assertEqual(self.console.text, "first7\nlast")
        self.assertEqual(self.console.calls, [
            ("configure", {"state": "normal"}),
            ("insert", "end", "first"),
            ("insert", "end", "7"),
            ("insert", "end", "\nlast"),
            ("see", "end"),
            ("configure", {"state": "disabled"}),
        ])
        self.assertEqual(len(self.console.scheduled), 1)

    def test_clear_preserves_queue_order(self):
        self.console.write("obsolete")
        self.console.clear()
        self.console.write("replacement")
        self.console.update_me()
        self.assertEqual(self.console.text, "replacement")
        self.assertEqual(self.console.state, "disabled")
        self.assertTrue(self.console.queue.empty())

    def test_clear_without_text_is_handled(self):
        self.console.clear()
        self.console.update_me()
        self.assertEqual(self.console.text, "")
        self.assertEqual(self.console.state, "disabled")
        self.assertTrue(self.console.queue.empty())

    def test_worker_only_enqueues_messages(self):
        def write_from_worker():
            self.console.write("old")
            self.console.clear()
            self.console.write("new")

        worker = threading.Thread(target=write_from_worker)
        worker.start()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.console.calls, [])
        self.assertEqual(self.console.scheduled, [])
        self.console.update_me()
        self.assertEqual(self.console.text, "new")

    def test_messages_enqueued_during_flush_are_not_lost(self):
        insert = self.console.insert

        def insert_and_enqueue(index, text):
            insert(index, text)
            if text == "first":
                self.console.write("second")

        self.console.insert = insert_and_enqueue
        self.console.write("first")
        self.console.update_me()
        self.assertEqual(self.console.text, "firstsecond")
        self.assertTrue(self.console.queue.empty())
        self.assertEqual(self.console.state, "disabled")


if __name__ == "__main__":
    unittest.main()
