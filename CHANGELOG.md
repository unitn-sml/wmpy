All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased / TODO]

### Added

- Bound computation functionalities
- Optimization functionalities
- Sampling functionalities
- redundant inequalities removal in Polytope
- Equalities support

### Fixed

- ...

### Changed

- ...

### Removed

- ...


## [0.1.1] - 2025-10-30

### Added

- `Inequality.to_numpy()` 

### Fixed

- Multiple bug fixes related to integration polynomial manipulations
- Multiple fixes in the unit tests

### Changed

- `Polytope.to_numpy()` now returns an additional numpy array with strictness information
- `AxisAlignedWrapper` now works with arbitrary polynomial integrands



## [0.1.0] - 2025-09-25

### Added

- First pre-release version.


[unreleased]: https://github.com/unitn-sml/wmpy/compare/v0.1.0...HEAD
[0.1.1]: https://github.com/unitn-sml/wmpy/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/unitn-sml/wmpy/releases/tag/v0.1.0
