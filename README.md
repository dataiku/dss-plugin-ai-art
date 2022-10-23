# AI Art
AI Art is a plugin for DSS that allows you to generate images from text prompts.

## Usage
For usage instructions, see *INSERT LINK HERE*.

> *TODO*: Add link to the plugin documentation. For now, see the Sphinx
  documentation in [doc](doc). Build it by running `make html`

## Testing and linting
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

## Known limitations
The weights must be stored on the local filesystem. If a remote folder (S3, etc)
is used, or if the recipe uses containerized execution, the weights will be
downloaded to a temporary directory every time the recipe is run. This is
because the method used to load the weights ([from_pretrained]) and the method
used to download the weights (git clone) require a local filepath.

> *TODO*: The macro for downloading weights is currently broken when using UIF
  with a local folder because it doesn't have write access to the local folder.
  This could potentially be fixed by downloading the weights to a local
  directory first like when using remote folders. Unsure if there's a better
  way.

[from_pretrained]: https://huggingface.co/docs/diffusers/v0.6.0/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.from_pretrained
