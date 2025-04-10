import unittest
import os
import json
from unittest.mock import patch, Mock, MagicMock
from datetime import date, timedelta
import random
import sys

# Ensure the project directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot import (
    load_user_data, save_user_data, update_user_data, update_student_profile,
    get_mental_health_resources, analyze_sentiment, load_llm, generate_summary,
    check_deadlines, send_daily_motivation, chatbot_response, USER_DATA_FILE,
    UNIVERSITY_RESOURCES, MOTIVATIONAL_QUOTES
)

class TestChatbot(unittest.TestCase):

    def setUp(self):
        # Remove user data file before each test to start with a clean slate
        if os.path.exists(USER_DATA_FILE):
            os.remove(USER_DATA_FILE)
        random.seed(42)  # Seed random for consistent motivational quote selection

    # --- Helper Functions Tests ---
    def test_load_user_data_empty(self):
        result = load_user_data()
        # Verify that an empty dict is returned when no file exists
        self.assertEqual(result, {})

    def test_save_and_load_user_data(self):
        data = {"user1": {"major": "CS"}}
        save_user_data(data)
        loaded_data = load_user_data()
        # Confirm that saved data matches loaded data
        self.assertEqual(loaded_data, data)

    def test_update_user_data_new_user(self):
        update_user_data("user1", "major", "CS")
        user_data = load_user_data()
        # Check if new user entry is created correctly
        self.assertEqual(user_data["user1"]["major"], "CS")

    def test_update_user_data_existing_user(self):
        save_user_data({"user1": {"major": "CS"}})
        update_user_data("user1", "year", "2")
        user_data = load_user_data()
        # Verify existing user data is updated without overwriting unrelated keys
        self.assertEqual(user_data["user1"], {"major": "CS", "year": "2"})

    def test_update_student_profile(self):
        update_student_profile("user1", "CS", "2", "exams", "Centennial College")
        user_data = load_user_data()
        expected = {
            "major": "CS",
            "year_of_study": "2",
            "common_stressors": "exams",
            "university": "Centennial College"
        }
        # Ensure all profile fields are stored correctly
        self.assertEqual(user_data["user1"], expected)

    def test_get_mental_health_resources_known_university(self):
        update_user_data("user1", "university", "Centennial College")
        result = get_mental_health_resources("user1")
        expected = "If you need support, check out Visit the Student Wellness Centre: https://www.centennialcollege.ca/student-health"
        # Test resource retrieval for a known university
        self.assertEqual(result, expected)

    def test_get_mental_health_resources_unknown_university(self):
        update_user_data("user1", "university", "Unknown Uni")
        result = get_mental_health_resources("user1")
        expected = "I recommend checking your university's website for student wellness resources."
        # Verify fallback message for unknown university
        self.assertEqual(result, expected)

    def test_get_mental_health_resources_no_university(self):
        result = get_mental_health_resources("user1")
        expected = "I recommend checking your university's website for student wellness resources."
        # Check default response when no university is set
        self.assertEqual(result, expected)

    def test_analyze_sentiment(self):
        # Test various sentiment categories based on input text
        self.assertEqual(analyze_sentiment("I hate everything"), "sad")
        self.assertEqual(analyze_sentiment("This is annoying"), "frustrated")
        self.assertEqual(analyze_sentiment("Hello world"), "neutral")
        self.assertEqual(analyze_sentiment("I love this"), "excited")
        self.assertEqual(analyze_sentiment("Amazing news!"), "excited")

    @patch("chatbot.ChatGroq")
    def test_load_llm(self, mock_chatgroq):
        mock_instance = MagicMock()
        mock_chatgroq.return_value = mock_instance
        llm = load_llm()
        # Verify ChatGroq is called with correct parameters
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
        # Confirm summary matches mocked LLM response
        self.assertEqual(summary, "User asked about stress; bot suggested breaks.")

    def test_check_deadlines_upcoming(self):
        today = date.today()
        deadlines = {
            "Assignment": today.isoformat(),
            "Exam": (today + timedelta(days=2)).isoformat(),
            "Project": (today + timedelta(days=5)).isoformat()
        }
        update_user_data("user1", "deadlines", deadlines)
        result = check_deadlines("user1")
        # Verify only deadlines within 3 days are included
        self.assertIn("Assignment", result)
        self.assertIn("Exam", result)
        self.assertNotIn("Project", result)

    def test_check_deadlines_none(self):
        today = date.today()
        deadlines = {"Project": (today + timedelta(days=5)).isoformat()}
        update_user_data("user1", "deadlines", deadlines)
        result = check_deadlines("user1")
        # Test response when no deadlines are within 3 days
        self.assertEqual(result, "No major deadlines soon. Keep up the good work!")

    def test_send_daily_motivation(self):
        result = send_daily_motivation()
        # Check that returned quote is from the predefined list (seeded for consistency)
        self.assertIn(result, MOTIVATIONAL_QUOTES)

    @patch("chatbot.load_llm")
    def test_chatbot_response(self, mock_load_llm):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Sorry to hear that. How can I help?"
        mock_llm.invoke.return_value = mock_response
        mock_load_llm.return_value = mock_llm
        user_id = "user1"
        update_user_data(user_id, "major", "CS")
        history = [("Hi", "Hello!")]
        message = "I'm stressed"
        ui_history, new_history = chatbot_response(message, history, user_id)
        expected_ui_history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "I'm stressed"},
            {"role": "assistant", "content": "Sorry to hear that. How can I help?"}
        ]
        # Verify UI history format and content
        self.assertEqual(ui_history, expected_ui_history)
        expected_history = [("Hi", "Hello!"), ("I'm stressed", "Sorry to hear that. How can I help?")]
        # Check internal history is updated correctly
        self.assertEqual(new_history, expected_history)

    def tearDown(self):
        # Clean up by removing user data file after each test
        if os.path.exists(USER_DATA_FILE):
            os.remove(USER_DATA_FILE)

if __name__ == "__main__":
    unittest.main()
