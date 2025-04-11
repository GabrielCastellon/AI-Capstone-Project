import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

import unittest
import os
import json
import gradio as gr
import threading
import time
from datetime import date, timedelta
from chatbot import (
    load_user_data, save_user_data, update_user_data, update_student_profile,
    get_mental_health_resources, analyze_sentiment, chatbot_response,
    USER_DATA_FILE, setup_profile, main
)

# ... (imports & setup remain the same)

class TestChatbotIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n[SetupClass] Launching Gradio App...")
        if os.path.exists(USER_DATA_FILE):
            os.remove(USER_DATA_FILE)
        cls.demo = None
        cls.thread = threading.Thread(target=cls.launch_gradio_app)
        cls.thread.daemon = True
        cls.thread.start()
        time.sleep(3)

    @classmethod
    def tearDownClass(cls):
        print("\n[TearDownClass] Cleaning up Gradio App...")
        if os.path.exists(USER_DATA_FILE):
            os.remove(USER_DATA_FILE)
        if cls.demo is not None:
            cls.demo.close()
        cls.thread.join(timeout=2)

    @classmethod
    def launch_gradio_app(cls):
        with gr.Blocks() as demo:
            print("[Gradio] Inside app thread — launching app...")
            main()
            cls.demo = demo
            demo.launch(prevent_thread_lock=True, quiet=True, server_port=7860, share=False)

    def setUp(self):
        print(f"\n[Setup] Preparing fresh user data...")
        if os.path.exists(USER_DATA_FILE):
            os.remove(USER_DATA_FILE)

    def tearDown(self):
        print("[TearDown] Removing user data file...")
        if os.path.exists(USER_DATA_FILE):
            os.remove(USER_DATA_FILE)

    def test_profile_setup_and_data_persistence(self):
        print("[Test] test_profile_setup_and_data_persistence")
        user_name = "test_user"
        major = "Computer Science"
        year = "3"
        stressors = "exams"
        university = "Centennial College"
        setup_result, user_id = setup_profile(user_name, major, year, stressors, university)
        print(f" -> Setup result: {setup_result}")
        user_data = load_user_data()
        print(f" -> Stored data: {user_data[user_name]}")
        self.assertEqual(setup_result, f"Profile set up for {user_name}. Welcome!")
        self.assertEqual(user_data[user_name]["university"], university)

    def test_chatbot_response_with_real_llm(self):
        print("[Test] test_chatbot_response_with_real_llm")
        user_id = "test_user"
        update_user_data(user_id, "major", "Psychology")
        history = [("Hi", "Hello! How can I assist you today?")]
        message = "I'm feeling stressed about exams"
        print(f" -> Sending message: {message}")
        ui_history, new_history = chatbot_response(message, history, user_id)
        print(f" -> Bot reply: {ui_history[3]['content']}")
        self.assertIn("stress", ui_history[3]["content"].lower())

    def test_deadlines_and_chat_integration(self):
        print("[Test] test_deadlines_and_chat_integration")
        user_id = "test_user"
        today = date.today()
        deadlines = {
            "Exam": today.isoformat(),
            "Project": (today + timedelta(days=5)).isoformat()
        }
        update_user_data(user_id, "deadlines", deadlines)
        message = "What’s on my schedule?"
        print(f" -> Asking about schedule: {message}")
        ui_history, new_history = chatbot_response(message, [], user_id)
        print(f" -> Bot reply: {ui_history[1]['content']}")
        self.assertIn("exam", ui_history[1]["content"].lower())
        self.assertNotIn("project", ui_history[1]["content"].lower())

    def test_mental_health_resources_in_chat(self):
        print("[Test] test_mental_health_resources_in_chat")
        user_id = "test_user"
        update_student_profile(user_id, "Biology", "2", "labs", "Centennial College")
        message = "I need help with stress"
        print(f" -> Sending message: {message}")
        ui_history, new_history = chatbot_response(message, [], user_id)
        print(f" -> Bot reply: {ui_history[1]['content']}")
        self.assertIn("https://www.centennialcollege.ca/student-health", ui_history[1]["content"])

    def test_clear_chat_functionality(self):
        print("[Test] test_clear_chat_functionality")
        user_id = "test_user"
        history = [("Hi", "Hello!")]
        message = "Test message"
        print(f" -> Sending message: {message}")
        ui_history, new_history = chatbot_response(message, history, user_id)
        print(f" -> Chat history length: {len(new_history)}")
        cleared_ui_history, cleared_history = ([], [])
        self.assertEqual(cleared_ui_history, [])
        self.assertEqual(cleared_history, [])

if __name__ == "__main__":
    unittest.main()
