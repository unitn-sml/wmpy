

We use `uv` for the development of this library.

To set up the environment:

```
uv sync
uv run wmpy-install --all -y
```


Before pushing to remote
------------------------

- Run the type checker: `uv run mypy`

- Run the tests: `uv run pytest`

- Fix any resulting error

- Make sure to reformat the source code: `uv run black`