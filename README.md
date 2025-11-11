# Presentations

A collection of technical presentations built with [Marp](https://marp.app/), focused on on-device AI, MLX, and Apple Silicon development.

## Overview

This repository hosts presentation decks covering various aspects of machine learning and AI development on Apple platforms. Each presentation is built using Marp for clean, version-controlled slides that can be viewed directly in the browser.

## Presentations

### 🎤 Jersey Shore Tech and Design
A deep dive into building production-ready on-device AI applications with MLX and Apple Silicon. Covers privacy-first architecture, performance optimization, and real-world implementation.

### 🤖 MLX
Technical exploration of Apple's MLX framework for machine learning on Apple Silicon.

### 🍎 Apple Foundation Model
Analysis and practical application of Apple's foundation models for on-device AI.

## Building

Each presentation is built from Markdown to HTML using Marp:

```bash
# Build individual presentations
marp jstd/presentation.md -o jstd/index.html
marp mlx/presentation.md -o mlx/index.html
marp apple-foundation-model/presentation.md -o apple-foundation-model/index.html

# Build all presentations at once
marp jstd/presentation.md -o jstd/index.html && \
marp mlx/presentation.md -o mlx/index.html && \
marp apple-foundation-model/presentation.md -o apple-foundation-model/index.html
```

## Development

### Prerequisites

- [Marp CLI](https://github.com/marp-team/marp-cli) installed globally:
  ```bash
  npm install -g @marp-team/marp-cli
  ```

### Local Preview

To preview presentations while editing:

```bash
# Watch mode with auto-reload
marp -w jstd/presentation.md -o jstd/index.html

# Preview in browser
marp -p jstd/presentation.md
```

## Viewing

Presentations are hosted via GitHub Pages and can be viewed at:
- [Browse all presentations](https://mpazaryna.github.io/presentation/)

## License

MIT

## Contact

- Email: matthew@paz.land
- GitHub: [@mpazaryna](https://github.com/mpazaryna)

## Load MCP Server

```bash
claude --mcp-config .mcp.json.context7
```