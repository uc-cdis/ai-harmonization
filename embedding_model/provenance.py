"""What a result was made with: the code's commit, a model's revision and the libraries' versions.

``score`` records all three. ``train``, ``compare`` and ``experiment compare``
record the commit and the libraries' versions. So a result can be traced to
the code and the model that produced it.
"""

import os
import platform
from importlib import import_module
from pathlib import Path

from embedding_model import manifests

# The libraries that train and score a model, whose versions a run's record
# and a score's config.yaml hold.
MODEL_LIBRARIES = (
    "sentence_transformers",
    "transformers",
    "torch",
    "datasets",
    "tokenizers",
)

# The files whose changes can change a score or a trained model, as git
# pathspecs from the top of the checkout.
CODE_PATHS = [
    ":(top,glob)ai_harmonization/**/*.py",
    ":(top,glob)embedding_model/**/*.py",
    ":(top)pyproject.toml",
    ":(top)uv.lock",
]


def code_commit(folder=Path(__file__).resolve().parent):
    """The ai-harmonization commit of the code that made a result.

    Only what can change a result counts as an uncommitted change. That is
    CODE_PATHS: the Python files of the two packages and the files that pin the
    dependencies. A changed doc does not count.

    Args:
        folder (Path): A folder of the checkout to ask, by default this
            package's.

    Returns:
        dict: ``commit``, the commit at ``HEAD``, and ``uncommitted_changes``,
            whether a file of CODE_PATHS differs from it or is new. Both are
            None when git cannot answer.
    """
    try:
        commit = manifests.git(folder, "rev-parse", "HEAD").strip()
        status = manifests.git(
            folder, "status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS
        )
    except RuntimeError:
        return {"commit": None, "uncommitted_changes": None}
    return {"commit": commit, "uncommitted_changes": bool(status.strip())}


def model_revision(model):
    """The Hugging Face commit of a model loaded by its id, read from the cache.

    Loading a model by its id downloads the model into the Hugging Face cache,
    under its commit. model_revision reads the commit from there, with no
    network, in the cache that SentenceTransformer uses:
    ``SENTENCE_TRANSFORMERS_HOME``, where set.

    model_revision looks for the id as given. SentenceTransformer reads an id
    with no owner, for example ``all-MiniLM-L6-v2``, as a
    ``sentence-transformers/`` model. So such an id gives None.

    Args:
        model (str): A Hugging Face model id or a model folder on disk.

    Returns:
        str | None: The commit, or None for a model folder on disk or an id
            that the cache does not hold.
    """
    if Path(model).is_dir():
        return None
    from huggingface_hub import snapshot_download

    try:
        folder = snapshot_download(
            model,
            local_files_only=True,
            cache_dir=os.getenv("SENTENCE_TRANSFORMERS_HOME"),
        )
    except Exception:
        return None
    return Path(folder).name


def library_versions(libraries=MODEL_LIBRARIES):
    """The versions of Python and of some libraries.

    Each version is the module's ``__version__``, as a plain str. torch's own
    ``__version__`` is a TorchVersion, a kind of str that YAML cannot write. It
    also keeps torch's build, for example ``+cu130``, which the package's
    metadata leaves out.

    Args:
        libraries (tuple[str]): The libraries, by their modules' names.

    Returns:
        dict[str, str]: Each version, by the library's name.
    """
    return {
        "python": platform.python_version(),
        **{name: str(import_module(name).__version__) for name in libraries},
    }
