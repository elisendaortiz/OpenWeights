# Record the UI demo

Record the actual UI yourself using macOS screen recording. Show a chat follow-up, an image question, audio chunk progress, and a time-agent tool call. Avoid recording personal files, secrets, or private conversations.

Convert the recording from the repository root:

```bash
ffmpeg -i demo.mov -vf "fps=10,scale=900:-1" docs/ui-demo.gif
```

Target a GIF under 5 MB; shorten the recording or reduce its frame rate or dimensions as needed. Then replace the README demo placeholder with:

```markdown
![OpenWeights Playground demo](docs/ui-demo.gif)
```

No demo recording is included or fabricated.
