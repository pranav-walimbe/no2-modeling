---
name: mermaid-diagrams
description: Always prefer Mermaid diagrams over ASCII art in markdown documents (READMEs, specs, ADRs, PR descriptions, reports). Use whenever the user asks for a diagram, flow, architecture sketch, sequence, state machine, ER diagram, or class diagram — and whenever you (the assistant) are about to draw boxes-and-arrows in a markdown file. Triggers on phrases like "draw a diagram", "show the flow", "architecture diagram", "sequence diagram", "state machine", "ERD", "class diagram", and any time you'd otherwise reach for ASCII boxes / pipes / dashes.
---

# Mermaid Diagrams (Not ASCII Art)

Markdown supports Mermaid natively on GitHub, Azure DevOps wikis, VS Code preview, and most modern viewers. **Always use Mermaid** for diagrams in markdown files. ASCII boxes-and-arrows are read-only, fragile to edit, accessibility-hostile, and don't render as images anywhere they matter.

## Rule

When you would otherwise draw something like:

```
+--------+      +--------+
|  API   | ---> |  DB    |
+--------+      +--------+
```

Write this instead:

````markdown
```mermaid
flowchart LR
    API --> DB
```
````

## Diagram Type Cheat-Sheet

Pick the diagram type that matches what you're showing. Don't force a `flowchart` for everything.

### Flowchart — components and data flow

````markdown
```mermaid
flowchart LR
    User -->|HTTPS| CF[CloudFront]
    CF -->|/api/*| APIGW[API Gateway]
    APIGW --> NLB --> API[API Container]
    API --> DB[(Aurora Postgres)]
    API -->|enqueue| SQS[(SQS Queue)]
    SQS --> Inference[Inference Container]
```
````

Directions: `LR` (left-right), `TB` (top-bottom). Most architecture diagrams read better in `LR`.

### Sequence — ordered interactions over time

````markdown
```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant A as API
    participant Q as SQS
    participant W as Worker
    U->>A: POST /jobs
    A->>Q: enqueue(jobId)
    A-->>U: 202 {jobId}
    Q->>W: deliver(jobId)
    W->>A: PATCH /jobs/{id} status=DONE
```
````

Use sequence when **order matters**. Add `autonumber` for traceability in long sequences.

### State — lifecycles

````markdown
```mermaid
stateDiagram-v2
    [*] --> QUEUED
    QUEUED --> EXECUTING
    EXECUTING --> SUCCEEDED
    EXECUTING --> FAILED
    FAILED --> QUEUED: retry < 3
    FAILED --> DLQ: retry >= 3
    SUCCEEDED --> [*]
    DLQ --> [*]
```
````

### ER — data model

````markdown
```mermaid
erDiagram
    USER ||--o{ JOB : creates
    JOB ||--|{ JOB_EVENT : has
    USER {
        uuid id PK
        string email
    }
    JOB {
        uuid id PK
        uuid user_id FK
        string status
    }
```
````

### Class — type relationships

````markdown
```mermaid
classDiagram
    class JobHandler {
        +handle(event) void
    }
    class TextTranslationHandler
    class AudioEnhanceHandler
    JobHandler <|-- TextTranslationHandler
    JobHandler <|-- AudioEnhanceHandler
```
````

### Gantt — timelines (use sparingly)

````markdown
```mermaid
gantt
    title Sprint 185
    dateFormat YYYY-MM-DD
    section Backend
    Implement post-processing :a1, 2026-04-20, 3d
    Eval + iterate            :a2, after a1, 4d
```
````

## Style Discipline

- **One diagram, one idea.** Don't cram architecture, sequence, and state into one chart. Use multiple smaller diagrams.
- **Label edges** with the verb or the protocol — `-->|HTTPS|`, `-->|enqueue|`, `-->|writes|`. Unlabeled arrows are noise.
- **Use shape semantics**:
  - `[Rectangle]` — service / process
  - `[(Cylinder)]` — datastore / queue
  - `((Circle))` — actor / start
  - `{Diamond}` — decision
- **Subgraph** to show trust boundaries, VPCs, accounts:

````markdown
```mermaid
flowchart LR
    subgraph AWS[AWS Account]
        API --> DB[(Postgres)]
    end
    User -->|HTTPS| API
```
````

- Keep node names short. Put detail in labels, not in IDs.
- Prefer `flowchart` over the older `graph` keyword.

## Where to Use

- **READMEs** — show how the project hangs together at a glance
- **Specs** (`spec-before-code` skill) — Architecture and Threat Model sections
- **ADRs** — visualize the decision and its impact
- **PR descriptions** — when changing data flow, show before/after
- **Reports** (`research-and-report` skill) — when a sequence or state transition explains the result
- **Wikis / runbooks** — operational flows, on-call decision trees

## Where Mermaid Renders Out-of-the-Box

- GitHub markdown (issues, PRs, READMEs, wikis)
- Azure DevOps wikis and PR descriptions (recent versions)
- VS Code markdown preview (with extension or built-in on recent versions)
- Most static site generators (Docusaurus, MkDocs Material, GitBook)
- Notion (paste as code block with `mermaid` language)

If a renderer doesn't support Mermaid, that's a renderer problem, not a diagram problem. Don't fall back to ASCII; export the Mermaid to SVG/PNG and link it.

## Migration Recipe

When you encounter ASCII art in an existing markdown file:

1. Read what the diagram is trying to communicate.
2. Pick the correct Mermaid diagram type.
3. Replace the ASCII block with a `mermaid` fenced code block.
4. Keep the surrounding prose; only swap the diagram.
5. Render-check it (GitHub preview, VS Code preview) before committing.

## Anti-Patterns

- ASCII boxes-and-arrows in any new markdown file.
- One mega-flowchart trying to show every component, every call, and every state.
- Unlabeled arrows.
- Embedding screenshots of diagrams when the source could be Mermaid (loses editability).
- Mermaid for things Mermaid is bad at — pixel-perfect layouts, freeform sketches, photos. Use a real diagram tool (Excalidraw, draw.io) and embed the export.
