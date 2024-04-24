import unittest
from unittest.mock import patch
from restful_ollama import send_post_request


class TestOllamaAPIRequests(unittest.TestCase):
    @patch('requests.post')
    def test_send_post_request(self, mock_post):
        # Set up the mock to return a successful response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'status': 'success'}

        # Call the function with a test model name
        response = send_post_request('llama2')

        # Assertions to check if the request was made correctly
        mock_post.assert_called_once_with(
            'http://ollama:11434/api/pull',
            json={'name': 'llama2', 'stream': False}
        )

        # Check the response handling
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'status': 'success'})


if __name__ == '__main__':
    unittest.main()
