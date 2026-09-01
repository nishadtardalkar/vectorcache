#!/bin/bash

export TMPDIR=./.tmp
export RUSTUP_HOME=./.rustup
export CARGO_HOME=./.cargo

mkdir -p $TMPDIR $RUSTUP_HOME $CARGO_HOME

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh