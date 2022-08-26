setup:
    python3 -m pip install poetry==1.2.0rc1
    poetry install
    mkdir -p .git/hooks
    ln -f -s `pwd`/hooks/* .git/hooks

setup-tpu:
    @just setup
    poetry run pip install --upgrade \
        jax[tpu]==`poetry export | grep jax== | cut -d';' -f1 | cut -d'=' -f3` \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

setup-cuda:
    @just setup
    poetry run pip install --upgrade \
        jax[cuda]==`poetry export | grep jax== | cut -d';' -f1 | cut -d'=' -f3` \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

test:
    poetry run pytest

lint:
    poetry run isort --check vit_jax
    poetry run black --check --include .py --exclude ".pyc|.pyi|.so" vit_jax
    poetry run black --check --pyi --include .pyi --exclude ".pyc|.py|.so" vit_jax
    poetry run pylint vit_jax
    poetry run pyright vit_jax

fix:
   poetry run isort vit_jax
   poetry run black --include .py --exclude ".pyc|.pyi|.so" vit_jax
   poetry run black --pyi --include .pyi --exclude ".pyc|.py|.so" vit_jax
