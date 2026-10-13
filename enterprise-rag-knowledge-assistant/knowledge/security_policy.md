# Security Policy

## Authentication
All workforce users must use single sign-on with multi-factor authentication. Service accounts use short-lived credentials and must not share interactive user passwords.

## Audit logging
Administrative actions, authentication events, and data-export events are written to the audit log. Audit records are retained for 365 days for enterprise customers and can be exported by security administrators.

## Data handling
Customer content is encrypted in transit using TLS and encrypted at rest. Production access is restricted by least-privilege role assignments and reviewed quarterly.
