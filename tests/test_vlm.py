"""
Tests for mobile_use/utils/vlm.py

Tests the VLMWrapper class for interacting with Vision Language Models.
Includes both unit tests with mocks and integration tests that require API credentials.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time

from mobile_use.utils.vlm import VLMWrapper


class TestVLMWrapperInit:
    """Tests for VLMWrapper initialization."""

    def test_init_with_required_params(self):
        """Test VLMWrapper initialization with required parameters."""
        with patch('mobile_use.utils.vlm.OpenAI') as mock_openai:
            vlm = VLMWrapper(
                model_name="test-model",
                api_key="test-key",
                base_url="https://api.example.com/v1"
            )
            assert vlm.model_name == "test-model"
            assert vlm.max_tokens == 1024
            assert vlm.temperature == 0.0
            mock_openai.assert_called_once()

    def test_init_with_custom_params(self):
        """Test VLMWrapper initialization with custom parameters."""
        with patch('mobile_use.utils.vlm.OpenAI') as mock_openai:
            vlm = VLMWrapper(
                model_name="custom-model",
                api_key="key",
                base_url="url",
                max_retry=5,
                retry_waiting_seconds=3,
                max_tokens=2048,
                temperature=0.5
            )
            assert vlm.max_retry == 5
            assert vlm.retry_waiting_seconds == 3
            assert vlm.max_tokens == 2048
            assert vlm.temperature == 0.5

    def test_init_with_extra_kwargs(self):
        """Test VLMWrapper initialization with extra VLM kwargs."""
        with patch('mobile_use.utils.vlm.OpenAI'):
            vlm = VLMWrapper(
                model_name="model",
                api_key="key",
                base_url="url",
                top_p=0.9,
                presence_penalty=0.5
            )
            assert vlm.vlm_kwargs == {'top_p': 0.9, 'presence_penalty': 0.5}


class TestVLMWrapperPredict:
    """Tests for VLMWrapper predict method."""

    def test_predict_success(self):
        """Test successful prediction."""
        with patch('mobile_use.utils.vlm.OpenAI') as mock_openai:
            # Setup mock
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            vlm = VLMWrapper(
                model_name="test-model",
                api_key="key",
                base_url="url"
            )

            messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
            response = vlm.predict(messages)

            assert response.choices[0].message.content == "Test response"
            mock_client.chat.completions.create.assert_called_once()

    def test_predict_with_stream(self):
        """Test prediction with streaming enabled."""
        with patch('mobile_use.utils.vlm.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            vlm = VLMWrapper(
                model_name="test-model",
                api_key="key",
                base_url="url"
            )

            messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
            vlm.predict(messages, stream=True)

            # Verify stream parameter was passed
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs['stream'] is True

    def test_predict_retry_on_failure(self):
        """Test that predict retries on failure."""
        with patch('mobile_use.utils.vlm.OpenAI') as mock_openai:
            with patch('mobile_use.utils.vlm.time.sleep'):  # Skip sleep in tests
                mock_client = MagicMock()
                # Fail twice, succeed on third try
                mock_client.chat.completions.create.side_effect = [
                    Exception("API Error"),
                    Exception("API Error"),
                    MagicMock()
                ]
                mock_openai.return_value = mock_client

                vlm = VLMWrapper(
                    model_name="test-model",
                    api_key="key",
                    base_url="url",
                    max_retry=3,
                    retry_waiting_seconds=0
                )

                messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
                response = vlm.predict(messages)

                assert response is not None
                assert mock_client.chat.completions.create.call_count == 3

    def test_predict_max_retry_exceeded(self):
        """Test that predict raises error after max retries."""
        with patch('mobile_use.utils.vlm.OpenAI') as mock_openai:
            with patch('mobile_use.utils.vlm.time.sleep'):
                mock_client = MagicMock()
                mock_client.chat.completions.create.side_effect = Exception("API Error")
                mock_openai.return_value = mock_client

                vlm = VLMWrapper(
                    model_name="test-model",
                    api_key="key",
                    base_url="url",
                    max_retry=2,
                    retry_waiting_seconds=0
                )

                messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
                
                with pytest.raises(ValueError, match="Max tries"):
                    vlm.predict(messages)

    def test_predict_passes_vlm_kwargs(self):
        """Test that extra vlm_kwargs are passed to API call."""
        with patch('mobile_use.utils.vlm.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            vlm = VLMWrapper(
                model_name="test-model",
                api_key="key",
                base_url="url",
                top_p=0.9,
                logprobs=True
            )

            messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
            vlm.predict(messages)

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs['top_p'] == 0.9
            assert call_kwargs['logprobs'] is True

    def test_predict_with_additional_kwargs(self):
        """Test predict with additional runtime kwargs."""
        with patch('mobile_use.utils.vlm.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            vlm = VLMWrapper(
                model_name="test-model",
                api_key="key",
                base_url="url"
            )

            messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
            vlm.predict(messages, stop=["END"], logprobs=True)

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs['stop'] == ["END"]
            assert call_kwargs['logprobs'] is True


class TestVLMWrapperIntegration:
    """Integration tests for VLMWrapper that require actual API credentials.
    
    These tests are skipped if VLM_API_KEY and VLM_BASE_URL are not set.
    """

    @pytest.fixture
    def vlm_wrapper(self):
        """Create a VLMWrapper with actual credentials."""
        import os
        api_key = os.getenv('VLM_API_KEY')
        base_url = os.getenv('VLM_BASE_URL')
        
        if not api_key or not base_url:
            pytest.skip("VLM credentials not available")
        
        return VLMWrapper(
            model_name='qwen2.5-vl-72b-instruct',
            api_key=api_key,
            base_url=base_url,
            max_tokens=128,
            max_retry=2,
            temperature=0.0
        )

    @pytest.mark.integration
    def test_predict_text_only(self, vlm_wrapper):
        """Test prediction with text-only message."""
        messages = [
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': 'What is 2 + 2? Answer with just the number.'}]
            }
        ]
        response = vlm_wrapper.predict(messages, stream=False)
        content = response.choices[0].message.content
        assert '4' in content

    @pytest.mark.integration
    def test_predict_with_image(self, vlm_wrapper, mock_image):
        """Test prediction with image."""
        from mobile_use.utils.utils import encode_image_url
        
        image_url = encode_image_url(mock_image)
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'What color is this image?'},
                    {'type': 'image_url', 'image_url': {'url': image_url}}
                ]
            }
        ]
        response = vlm_wrapper.predict(messages, stream=False)
        content = response.choices[0].message.content
        assert len(content) > 0

    @pytest.mark.integration
    def test_predict_stream(self, vlm_wrapper):
        """Test streaming prediction."""
        messages = [
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': 'Say hello in one word.'}]
            }
        ]
        response = vlm_wrapper.predict(messages, stream=True)
        
        chunks = []
        for chunk in response:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        
        full_response = ''.join(chunks)
        assert len(full_response) > 0

