# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a presentation repository using Marp (Markdown Presentation Ecosystem) to build technical presentations about on-device AI, MLX, and Apple Silicon development. The presentations are hosted on GitHub Pages.

**Key Topics:**
- Jersey Shore Tech and Design (JSTD): Building production-ready on-device AI with MLX
- MLX Framework: Machine learning on Apple Silicon
- Apple Foundation Models: On-device AI capabilities

## Prerequisites

Marp CLI must be installed globally:
```bash
npm install -g @marp-team/marp-cli
```

## Building Presentations

### Build Individual Presentations
```bash
# JSTD presentation
marp jstd/presentation.md -o jstd/index.html

# MLX presentation
marp mlx/presentation.md -o mlx/index.html

# Apple Foundation Model presentation
marp apple-foundation-model/presentation.md -o apple-foundation-model/index.html
```

### Build All Presentations
```bash
marp jstd/presentation.md -o jstd/index.html && \
marp mlx/presentation.md -o mlx/index.html && \
marp apple-foundation-model/presentation.md -o apple-foundation-model/index.html
```

## Development Workflow

### Preview While Editing
```bash
# Watch mode with auto-reload
marp -w jstd/presentation.md -o jstd/index.html

# Live preview in browser
marp -p jstd/presentation.md
```

## Repository Structure

```
presentation/
├── index.html              # Landing page for all presentations
├── jstd/                   # Jersey Shore Tech and Design presentation
│   ├── presentation.md     # Main presentation source
│   ├── README.md          # Detailed presenter notes and study guide
│   ├── index.html         # Built presentation
│   └── image.png          # Integration screenshot
├── mlx/                    # MLX framework presentation
│   ├── presentation.md     # Main presentation source
│   └── index.html         # Built presentation
└── apple-foundation-model/ # Apple Foundation Models presentation
    ├── presentation.md     # Main presentation source
    └── index.html         # Built presentation
```

## Architecture Notes

### Presentation Format
- All presentations use **Marp** with YAML frontmatter configuration
- Output format: Markdown → HTML conversion
- Theme: Default Marp theme (MLX uses custom font sizing)
- Hosting: GitHub Pages at https://mpazaryna.github.io/presentation/

### Landing Page (index.html)
- Pure HTML/CSS with no build dependencies
- Responsive design using flexbox
- Gradient background (purple theme)
- Links to all presentation subdirectories

### Content Organization
Each presentation directory contains:
1. `presentation.md` - Source slides in Marp format
2. `index.html` - Built presentation (committed to repo)
3. Optional `README.md` - Presenter notes and detailed technical content (JSTD only)

### JSTD Presentation Details
The JSTD presentation is the most comprehensive and includes:
- **README.md**: Extensive presenter notes covering technical depth, Q&A preparation, and concept mapping
- **Key topics**: .safetensors format, MLX training pipeline, data processing workflow, HIPAA compliance
- **Real metrics**: Model loading (34ms), inference times (2-14ms), training date tracking
- **Architecture**: Pure MLX implementation with 3 specialized models (ICD-10, CPT, Vertebral)

## When Making Changes

### Editing Presentations
1. Edit the `presentation.md` file in the relevant subdirectory
2. Build using Marp CLI to regenerate the `index.html`
3. Test the built HTML in a browser before committing
4. Both source (.md) and built (.html) files are committed

### Adding New Presentations
1. Create a new subdirectory with the presentation name
2. Add `presentation.md` with Marp frontmatter
3. Build to generate `index.html`
4. Update root `index.html` to add a link to the new presentation
5. Update the root `README.md` to document the new presentation

### Marp Frontmatter Configuration
Standard frontmatter for presentations:
```yaml
---
marp: true
theme: default
---
```

For custom styling (see mlx/presentation.md):
```yaml
---
marp: true
theme: default
style: |
  section {
    font-size: 28px;
  }
---
```

## Deployment

Presentations are automatically served via GitHub Pages from the main branch. After building and committing HTML files, changes will be live at:
- Landing page: https://mpazaryna.github.io/presentation/
- Individual presentations: https://mpazaryna.github.io/presentation/[subdirectory]/

## Contact Information

- Email: matthew@paz.land
- GitHub: [@mpazaryna](https://github.com/mpazaryna)
