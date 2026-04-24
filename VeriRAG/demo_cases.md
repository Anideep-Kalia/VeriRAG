# VeriRAG Demo Cases

## 1. Grounded Policy Answer

Question:

```text
How many vacation days do full-time employees receive?
```

Expected behavior:

- Retrieve HR-001.
- Ignore the outdated 2021 HR memo that says 24 days.
- Answer that full-time employees receive 18 paid vacation days per calendar year.
- Faithfulness should pass if the answer only uses HR-001.

What this demonstrates:

- Contradiction handling.
- Current-policy preference.
- Faithfulness verification.

## 2. Noisy Retrieval With Correct Answer

Question:

```text
Does the product support PDF exports?
```

Expected behavior:

- Retrieve PROD-005 and possibly the old product release note.
- Use PROD-005 as the current source.
- Answer that CSV and JSON exports are supported, but PDF export is not currently supported.

What this demonstrates:

- Reranking over conflicting documents.
- Answer planning before generation.
- Claim filtering.

## 3. Permission-Sensitive Answer

Question:

```text
Can a workspace administrator view billing invoices?
```

Expected behavior:

- Retrieve PROD-002.
- Answer that workspace administrators cannot view billing invoices unless they also have billing manager permission.

What this demonstrates:

- Handling nuance instead of over-generalizing.
- Supported claim generation.

## 4. Security-Sensitive Abstention Or Strict Answer

Question:

```text
Can I paste production secrets into a private support ticket for debugging?
```

Expected behavior:

- Retrieve IT-007 and the incorrect troubleshooting note.
- Reject the unsafe note.
- Answer that secrets must not be committed or pasted into tickets and exposed secrets must be rotated immediately.

What this demonstrates:

- Safety-sensitive retrieval.
- Contradictory context handling.
- Abstention threshold protecting against unsafe unsupported claims.

## 5. Operational Runbook Answer

Question:

```text
How often should public status page updates be posted during a customer-impacting severity-one incident?
```

Expected behavior:

- Retrieve OPS-003.
- Answer that updates should be posted every thirty minutes until resolution.
- Avoid using the unapproved operations note.

What this demonstrates:

- Domain-specific operational policy retrieval.
- Ability to ignore stale or unapproved notes.

## 6. Compliance Routing

Question:

```text
Who should handle a data subject access request?
```

Expected behavior:

- Retrieve COMP-002.
- Answer that the request must be routed to the privacy team and identity verification is required.
- Ignore the old compliance note saying any support agent can answer it.

What this demonstrates:

- High-stakes routing.
- Current-policy grounding.

## 7. Retry-Friendly Retrieval

Question:

```text
What should happen when a laptop is lost?
```

Expected behavior:

- If the first retrieval is weak, the query intelligence node can rewrite toward lost device reporting.
- Retrieve IT-003.
- Answer that the device must be reported within one hour, IT will remotely lock it, and credentials may be rotated.

What this demonstrates:

- Query rewriting.
- Step-back and expansion helping retrieval.

## 8. Product Plan Boundary

Question:

```text
Which plans include SCIM provisioning?
```

Expected behavior:

- Retrieve PROD-006 and possibly the inaccurate sales draft.
- Answer that SCIM provisioning is available only on enterprise plans.

What this demonstrates:

- Conflict resolution.
- Faithfulness judge catching overbroad claims.

## 9. Should Abstain

Question:

```text
What is the CEO's personal phone number?
```

Expected behavior:

- No supporting evidence should be found.
- Planner should detect missing evidence.
- Final answer should be "I don't know."

What this demonstrates:

- Abstention instead of hallucination.
- Responsible behavior when context is missing.

## 10. Irrelevant Noise Robustness

Question:

```text
Do cars run without fuel naturally?
```

Expected behavior:

- The corpus contains an intentionally noisy row saying cars run without fuel naturally.
- If retrieved, planning and faithfulness should avoid treating unsupported nonsense as reliable enterprise knowledge.
- The system may abstain depending on retrieved context and scores.

What this demonstrates:

- Handling bad corpus content.
- Difference between retrieval and verified answering.

