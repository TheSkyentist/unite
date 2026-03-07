# Contributing

Contributions are welcome. The project is hosted on
[GitHub](https://github.com/TheSkyentist/unite).

## Reporting issues

If you find a bug or unexpected behaviour, please
[open an issue](https://github.com/TheSkyentist/unite/issues). Include a minimal
reproducible example and the version of `unite` you are using (`import unite; print(unite.__version__)`).

## Suggesting features

Feature requests are also tracked via
[GitHub Issues](https://github.com/TheSkyentist/unite/issues). Describe the use-case
and, if possible, sketch out what the API might look like.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Install the development environment with [Pixi](https://pixi.sh/):
   ```bash
   pixi install
   ```
3. Make your changes, then run the test suite and linter before pushing:
   ```bash
   pixi run test
   pixi run lint
   ```
4. Open a pull request against `main`. The CI will run automatically.

All contributions — bug fixes, new features, documentation improvements, and
additional instrument support — are appreciated.
