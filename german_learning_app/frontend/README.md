# Frontend (placeholder)

Reserved for the React + Vite client. Scaffold it here with:

```bash
npm create vite@latest . -- --template react-ts
```

The dev server runs on `http://localhost:5173`, which the backend already allows via `CORS_ORIGINS`.
Planned layout:

```
src/
├── api/          # typed client for /api/analyze and /api/flashcard/*
├── features/
│   ├── dictionary/   # word lookup, context and homonym views
│   ├── voice/        # Web Speech API capture -> /api/analyze (source="voice")
│   └── practice/     # cloze / translation drills from /api/flashcard/practice
└── main.tsx
```

Production hosting: S3 + CloudFront (Origin Access Control), with `/api/*` routed to the backend origin.
