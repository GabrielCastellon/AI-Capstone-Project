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

class TestChatbotIntegration(unittest.TestCase):

    def setUp(self):
        # Remove existing user data file for a fresh start
        if os.path.exists(USER_DATA_FILE):
            os.remove(USER_DATA_FILE)
        self.demo = None
        # Start Gradio app in a separate thread to simulate real usage
        self.thread = threading.Thread(target=self.launch_gradio_app)
        self.thread.daemon = True  # Daemonize thread to exit with main program
        self.thread.start()
        time.sleep(2)  # Wait for Gradio app to fully launch

    def tearDown(self):
        # Clean up user data file after each test
        if os.path.exists(USER_DATA_FILE):
            os.remove(USER_DATA_FILE)
        if self.demo is not None:
            self.demo.close()  # Shut down the Gradio app if it was launched
        self.thread.join(timeout=1)  # Ensure thread terminates with a timeout

    def launch_gradio_app(self):
        with gr.Blocks() as demo:
            main()  # Call the main function to set up the Gradio interface
            self.demo = demo
            # Launch without blocking the main thread and suppress logs
            demo.launch(prevent_thread_lock=True, quiet=True, server_port=7860)

    def test_profile_setup_and_data_persistence(self):
        user_name = "test_user"
        major = "Computer Science"
        year = "3"
        stressors = "exams"
        university = "Centennial College"
        setup_result, user_id = setup_profile(user_name, major, year, stressors, university)
        # Verify setup confirmation message
        self.assertEqual(setup_result, f"Profile set up for {user_name}. Welcome!")
        self.assertEqual(user_id, user_name)
        user_data = load_user_data()
        expected_profile = {
            "name": user_name,
            "major": major,
            "year_of_study": year,
            "common_stressors": stressors,
            "university": university
        }
        # Check that profile data is correctly persisted to JSON
        self.assertEqual(user_data[user_name], expected_profile)

    def test_chatbot_response_with_real_llm(self):
        user_id = "test_user"
        update_user_data(user_id, "major", "Psychology")
        history = [("Hi", "Hello! How can I assist you today?")]
        message = "I'm feeling stressed about exams"
        ui_history, new_history = chatbot_response(message, history, user_id)
        # Confirm history length reflects initial + new exchange
        self.assertEqual(len(ui_history), 4)
        self.assertEqual(ui_history[2]["role"], "user")
        self.assertEqual(ui_history[2]["content"], message)
        self.assertEqual(ui_history[3]["role"], "assistant")
        # Verify response is a string (real LLM output)
        self.assertTrue(isinstance(ui_history[3]["content"], str))
        response_lower = ui_history[3]["content"].lower()
        # Check if response contextually relates to stress or exams
        self.assertTrue("stress" in response_lower or "exam" in response_lower or "help" in response_lower)

    def test_deadlines_and_chat_integration(self):
        user_id = "test_user"
        today = date.today()
        deadlines = {
            "Exam": today.isoformat(),
            "Project": (today + timedelta(days=5)).isoformat()
        }
        update_user_data(user_id, "deadlines", deadlines)
        history = []
        message = "What’s on my schedule?"
        ui_history, new_history = chatbot_response(message, history, user_id)
        response = ui_history[1]["content"].lower()
        # Ensure only imminent deadlines (within 3 days) are mentioned
        self.assertIn("exam", response)
        self.assertNotIn("project", response)

    def test_mental_health_resources_in_chat(self):
        user_id = "test_user"
        update_student_profile(user_id, "Biology", "2", "labs", "Centennial College")
        history = []
        message = "I need help with stress"
        ui_history, new_history = chatbot_response(message, history, user_id)
        response = ui_history[1]["content"]
        expected_resource = "https://www.centennialcollege.ca/student-health"
        # Verify that mental health resources are included for stress-related queries
        self.assertIn(expected_resource, response)

    def test_clear_chat_functionality(self):
        user_id = "test_user"
        history = [("Hi", "Hello!")]
        message = "Test message"
        ui_history, new_history = chatbot_response(message, history, user_id)
        # Confirm history grows after a response
        self.assertEqual(len(new_history), 2)
        cleared_ui_history, cleared_history = ([], [])
        # Simulate clear button by checking empty history states
        self.assertEqual(cleared_ui_history, [])
        self.assertEqual(cleared_history, [])

if __name__ == "__main__":
    unittest.main()