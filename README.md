# AI Art
AI Art is a plugin for DSS that allows you to generate images from text prompts.

# Usage
For usage instructions, see *INSERT LINK HERE*.

> *TODO*: Add link to the plugin documentation. For now, see the Sphinx
  documentation in [doc](doc). Build it by running `make html`

# Testing and linting
Install the development dependencies:
```bash
pip install -r requirements-dev.txt
```

Format the code using Black:
```bash
black .
```

Lint the code using Flake8:
```bash
flake8 .
```

> *TODO*: Add tests

# Known limitations
The weights folder must be stored on the local filesystem. Remote connections
such as S3 don't work. This is because the method used to load the weights
([from_pretrained]) and the method used to download the weights (git clone)
require a local filepath.

[from_pretrained]: https://huggingface.co/docs/diffusers/v0.6.0/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.from_pretrained
