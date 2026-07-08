# Payments API Reference

## Rate limiting and retries

The Payments API enforces a rate limit of 100 requests per second per API key.
When a client exceeds this limit, the API responds with HTTP status 429 (Too
Many Requests) and includes a `Retry-After` header indicating the number of
seconds to wait before retrying.

Clients should implement exponential backoff with jitter when they receive a
429 response. The official SDKs retry idempotent requests automatically up to
three times, doubling the delay after each attempt. Non-idempotent requests
(such as creating a charge without an idempotency key) are never retried
automatically, because a retry could result in a duplicate charge.

## Idempotency

Every write endpoint accepts an `Idempotency-Key` header. Supplying a unique
key ensures that retrying a request has no additional effect: the server stores
the result of the first successful request for 24 hours and returns that same
result for any subsequent request carrying the same key.

## Webhooks

The API delivers event notifications via signed webhooks. Each webhook payload
is signed with HMAC-SHA256 using your webhook signing secret. Verify the
`X-Signature` header before trusting the payload. Webhook deliveries are retried
for up to 72 hours with exponential backoff if your endpoint does not return a
2xx status.
