# Record the FastAPI UI demo

Record the actual UI yourself using macOS screen recording. Show a streamed chat and Stop, an image follow-up, audio chunk progress, and an agent tool call. Avoid personal recordings, images and secrets.

From the repository root, convert your recording:

```bash
ffmpeg -i demo.mov -vf "fps=10,scale=900:-1" docs/ui-demo.gif
```

Keep the GIF under 5 MB by shortening the clip or reducing its frame rate or size. Replace the README demo placeholder with:

```markdown
![OpenWeights Playground demo](docs/ui-demo.gif)
```

No demo GIF is fabricated or included.
