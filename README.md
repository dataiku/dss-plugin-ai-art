# AI Art
AI Art is a plugin for DSS that allows you to generate images from text prompts.

## Usage
For usage instructions, see the [documentation][plugin_documentation].

[plugin_documentation]: https://www.dataiku.com/product/plugins/ai-art/

## Testing and linting
Format the code using Black:
```bash
make black
```

Lint the code using Flake8, and verify that the code was formatted using Black:
```bash
make lint
```

Run unit tests:
```bash
make unit-tests
```

Run integration tests:
```bash
make integration-tests
```

Run all linters and tests at once:
```bash
make tests
```

## Building
Create a plugin archive that can be imported into DSS:
```bash
make plugin
```

## Known limitations
1.  Due to licensing restrictions, the weights must be acquired manually before
    you can use this plugin. See the [documentation][plugin_documentation] for
    details.

1.  The weights must be stored on the local filesystem. If a remote folder (S3,
    etc) is used, or if the recipe uses containerized execution, the weights
    will be downloaded to a temporary directory every time the recipe is run.
    This is because the method used to load the weights ([from_pretrained])
    requires a local filepath.

[from_pretrained]: https://huggingface.co/docs/diffusers/v0.6.0/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.from_pretrained
