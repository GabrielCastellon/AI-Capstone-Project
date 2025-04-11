import os
import sys
import unittest
import json
import random
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

# Add the project's root directory to sys.path so we can import the main module.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from chatbot import (
    load_user_data,
    save_user_data,
    update_user_data,
    update_student_profile,
    get_mental_health_resources,
    analyze_sentiment,
    load_llm,
    generate_summary,
    check_deadlines,
    send_daily_motivation,
    chatbot_response,
    setup_profile,
    USER_DATA_FILE,
    MOTIVATIONAL_QUOTES,
)

class TestChatbotUnit(unittest.TestCase):
    def setUp(self):
        # Remove user_data.json before each test to ensure tests remain isolated.
        if os.path.exists(USER_DATA_FILE):
            os.remove(USER_DATA_FILE)
        random.seed(42)  # Seed for consistent outputs in tests

    def tearDown(self):
        # Clean up user_data.json after each test.
        if os.path.exists(USER_DATA_FILE):
            os.remove(USER_DATA_FILE)

    # --- User Data and Profile Functions ---
    def test_load_user_data_empty(self):
        self.assertEqual(load_user_data(), {})

    def test_save_and_load_user_data(self):
        data = {"user1": {"major": "CS"}}
        save_user_data(data)
        loaded_data = load_user_data()
        self.assertEqual(loaded_data, data)

    def test_update_user_data_new_and_existing(self):
        update_user_data("user1", "major", "CS")
        data = load_user_data()
        self.assertEqual(data["user1"]["major"], "CS")
        
        update_user_data("user1", "year", "2")
        data = load_user_data()
        self.assertEqual(data["user1"], {"major": "CS", "year": "2"})

    def test_update_student_profile(self):
        update_student_profile("user1", "CS", "2", "exams", "Centennial College")
        data = load_user_data()
        expected = {
            "major": "CS",
            "year_of_study": "2",
            "common_stressors": "exams",
            "university": "Centennial College"
        }
        self.assertEqual(data["user1"], expected)

    # --- Resource & Sentiment Functions ---
    def test_get_mental_health_resources_known_university(self):
        update_user_data("user1", "university", "Centennial College")
        result = get_mental_health_resources("user1")
        self.assertIn("https://www.centennialcollege.ca/student-health", result)

    def test_get_mental_health_resources_unknown_university(self):
        update_user_data("user1", "university", "Unknown Uni")
        result = get_mental_health_resources("user1")
        self.assertIn("I recommend checking your university", result)

    def test_get_mental_health_resources_no_university(self):
        result = get_mental_health_resources("user1")
        self.assertIn("I recommend checking your university", result)

    def test_analyze_sentiment(self):
        self.assertEqual(analyze_sentiment("I hate everything"), "sad")
        self.assertEqual(analyze_sentiment("This is annoying"), "frustrated")
        self.assertEqual(analyze_sentiment("Hello world"), "neutral")
        self.assertEqual(analyze_sentiment("I love this"), "excited")
        self.assertEqual(analyze_sentiment("Amazing news!"), "excited")

    # --- LLM and Summary Functions ---
    @patch("chatbot.ChatGroq")
    def test_load_llm(self, mock_chatgroq):
        mock_instance = MagicMock()
        mock_chatgroq.return_value = mock_instance
        llm = load_llm()
        mock_chatgroq.assert_called_once_with(
            temperature=0.6,
            groq_api_key="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",
            model_name="llama-3.3-70b-versatile"
        )
        self.assertEqual(llm, mock_instance)

    @patch("chatbot.ChatGroq")
    def test_generate_summary(self, mock_chatgroq):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "User asked about stress; bot suggested breaks."
        mock_llm.invoke.return_value = mock_response
        mock_chatgroq.return_value = mock_llm
        history = ["I'm stressed", "Take a break!"]
        llm = load_llm()
        summary = generate_summary(history, llm)
        self.assertEqual(summary, "User asked about stress; bot suggested breaks.")

    # --- Deadline and Motivation Functions ---
    def test_check_deadlines_upcoming(self):
        today = date.today()
        deadlines = {
            "Assignment": today.isoformat(),
            "Exam": (today + timedelta(days=2)).isoformat(),
            "Project": (today + timedelta(days=5)).isoformat()
        }
        update_user_data("user1", "deadlines", deadlines)
        message = check_deadlines("user1")
        self.assertIn("Assignment", message)
        self.assertIn("Exam", message)
        self.assertNotIn("Project", message)

    def test_check_deadlines_none(self):
        today = date.today()
        deadlines = {"Project": (today + timedelta(days=5)).isoformat()}
        update_user_data("user1", "deadlines", deadlines)
        message = check_deadlines("user1")
        self.assertEqual(message, "No major deadlines soon. Keep up the good work!")

    def test_send_daily_motivation(self):
        quote = send_daily_motivation()
        self.assertIn(quote, MOTIVATIONAL_QUOTES)

    # --- Chatbot Response and Profile Setup ---
    @patch("chatbot.load_llm")
    def test_chatbot_response(self, mock_load_llm):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Sorry to hear that. How can I help?"
        mock_llm.invoke.return_value = mock_response
        mock_load_llm.return_value = mock_llm

        update_user_data("user1", "major", "CS")
        history = [("Hi", "Hello!")]
        message = "I'm stressed"
        ui_history, updated_history = chatbot_response(message, history, "user1")
        expected_ui_history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "I'm stressed"},
            {"role": "assistant", "content": "Sorry to hear that. How can I help?"}
        ]
        self.assertEqual(ui_history, expected_ui_history)
        expected_history = [("Hi", "Hello!"), ("I'm stressed", "Sorry to hear that. How can I help?")]
        self.assertEqual(updated_history, expected_history)

    def test_setup_profile(self):
        status, user_id = setup_profile("test_user", "Computer Science", "3", "exams", "Centennial College")
        data = load_user_data()
        self.assertIn("test_user", data)
        self.assertEqual(data["test_user"]["university"], "Centennial College")
        self.assertIn("Profile set up for test_user", status)

if __name__ == "__main__":
    unittest.main()
