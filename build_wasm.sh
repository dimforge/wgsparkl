#!/bin/sh

# cargo build -p wgsparkl-examples-2d --release --target wasm32-unknown-unknown
cargo build -p wgsparkl-examples-3d --release --target wasm32-unknown-unknown
# wasm-bindgen --no-typescript --target web --out-dir dist2d --out-name wgsparkl-examples-2d ./target/wasm32-unknown-unknown/release/wgsparkl-examples-2d.wasm
wasm-bindgen --no-typescript --target web --out-dir dist3d --out-name wgsparkl-examples-3d ./target/wasm32-unknown-unknown/release/wgsparkl-examples-3d.wasm
# wasm-opt -Oz -o ./dist2d/opt.wasm ./dist2d/wgsparkl-examples-2d_bg.wasm && mv ./dist2d/opt.wasm ./dist2d/wgsparkl-examples-2d_bg.wasm
wasm-opt -Oz -o ./dist3d/opt.wasm ./dist3d/wgsparkl-examples-3d_bg.wasm && mv ./dist3d/opt.wasm ./dist3d/wgsparkl-examples-3d_bg.wasm

# brotli ./dist2d/wgsparkl-examples-2d_bg.wasm && mv ./dist2d/wgsparkl-examples-2d_bg.wasm.br ./dist2d/wgsparkl-examples-2d_bg.wasm
brotli ./dist3d/wgsparkl-examples-3d_bg.wasm && mv ./dist3d/wgsparkl-examples-3d_bg.wasm.br ./dist3d/wgsparkl-examples-3d_bg.wasm