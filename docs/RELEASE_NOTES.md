# Source package notes

Source was collected from the author's unified SPRINTracker server project. The public package excludes datasets, videos, checkpoints, credentials, and runtime output directories.

Release edits replace machine-specific absolute paths with repository-relative paths, resolve the shell project root from the script location, and allow `PYTHON_BIN` and `ACCELERATE_BIN` overrides. No model architecture or learned parameters were changed during packaging. New documentation, dependency guidance, the author-supplied figure, and a raw validation summary were added.

Static validation covers Python parsing, YAML parsing and parent-config resolution, and required release assets. It does not establish numerical reproduction or validate the unpinned dependency combination. The bundled legacy upstream documents remain historical references; follow the SPRINTracker README for the release entry points.
