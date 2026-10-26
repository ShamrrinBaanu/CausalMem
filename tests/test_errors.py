from llm_service import user_facing_error

class ProviderError(Exception):
    def __init__(self,status_code,body): self.status_code=status_code;self.body=body

def test_quota_message_is_actionable_and_sanitized():
    exc=ProviderError(429,{"error":{"code":"credit_balance_exhausted","message":"You have no credits remaining."}})
    message=user_facing_error(exc)
    assert "no credits remaining" in message
    assert "credit_balance_exhausted" not in message

def test_invalid_key_is_sanitized():
    exc=ProviderError(401,{"error":{"code":"invalid_api_key","message":"bad secret"}})
    message=user_facing_error(exc)
    assert "rejected the API key" in message
    assert "bad secret" not in message
