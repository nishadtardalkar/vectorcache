#!/bin/bash

export TMPDIR=/rsstu/users/d/dslalush/KneeProject/.cache/tmp
export RUSTUP_HOME=/rsstu/users/d/dslalush/KneeProject/.cache/rustup
export CARGO_HOME=/rsstu/users/d/dslalush/KneeProject/.cache/cargo

mkdir -p $TMPDIR $RUSTUP_HOME $CARGO_HOME

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh