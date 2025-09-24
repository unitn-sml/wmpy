

Feedback, pull requests, active development.. any form of contribution
is much welcome!

If you want to get in touch, contact Paolo Morettin (paolo.morettin@unitn.it).


Setup
-----

We use `uv` for the development of this library.

To set up the environment:

```
uv sync
uv run wmpy-install --all -y
```

To build the documentation:

```
cd docs
uv run make html
```


Before pushing to remote
------------------------

- Run the type checker: `uv run mypy`

- Run the tests: `uv run pytest`

- Fix any resulting error

- Make sure to reformat the source code: `uv run black`


Conventions
-----------

We adopt the following conventions

- Python code format: [PEP8](https://peps.python.org/pep-0008/)

- Docstrings: [PEP257](https://peps.python.org/pep-0257/), Google [style](https://google.github.io/styleguide/pyguide.html)

- Version numbers: [semantic versioning](https://semver.org/)