#!/bin/sh

cargo build --example elasticity2 --release --target wasm32-unknown-unknown --features dim2
#cargo build --example elasticity3 --release --target wasm32-unknown-unknown --features dim3
wasm-bindgen --no-typescript --target web --out-dir dist2d --out-name elasticity2 ./target/wasm32-unknown-unknown/release/examples/elasticity2.wasm
#wasm-bindgen --no-typescript --target web --out-dir dist3d --out-name elasticity3 ./target/wasm32-unknown-unknown/release/examples/elasticity3.wasm
~/.cargo/bin/wasm-opt -Oz -o ./dist2d/opt.wasm ./dist2d/elasticity2_bg.wasm && mv ./dist2d/opt.wasm ./dist2d/elasticity2_bg.wasm
#~/.cargo/bin/wasm-opt -Oz -o ./dist3d/opt.wasm ./dist3d/elasticity3_bg.wasm && mv ./dist3d/opt.wasm ./dist3d/elasticity3_bg.wasm

brotli ./dist2d/elasticity2_bg.wasm && mv ./dist2d/elasticity2_bg.wasm.br ./dist2d/elasticity2_bg.wasm
#brotli ./dist3d/elasticity3_bg.wasm && mv ./dist3d/elasticity3_bg.wasm.br ./dist3d/elasticity3_bg.wasm