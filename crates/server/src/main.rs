// With fast 4g from chrome:
// - gzip: 8,1 MB: 8.9 seconds (fp) -> 10.1 (fcp)
// - br: 5,6 MB: 6.4 seconds (fp) -> 7.6 (fcp)
// - txt: 24,6 MB: 25.2 seconds (fp) -> 26.4 (fcp)

use axum::{
    response::{Html, IntoResponse},
    routing::get,
    Router,
};
use hyper::header::{HeaderValue, CONTENT_ENCODING, CONTENT_TYPE};
use std::net::SocketAddr;
use tower_http::compression::CompressionLayer;

#[tokio::main]
async fn main() {
    // Create the router with a compression layer for gzip
    let app = Router::new()
        // Route for the HTML file
        .route("/", get(serve_html))
        // Route for the JavaScript file
        .route("/elasticity2.js", get(serve_js))
        // Route for the gzipped .wasm file
        .route("/elasticity2_bg.wasm", get(serve_wasm))
        // Apply gzip compression to responses
        .layer(CompressionLayer::new());

    // Address and port for the server
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("Server listening on http://{}", addr);

    // Run the server
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

// Serve the HTML file
async fn serve_html() -> impl IntoResponse {
    Html(include_str!("../../../dist2d/index.html"))
}

// Serve the JavaScript file (example contents omitted for brevity)
async fn serve_js() -> impl IntoResponse {
    (
        [(
            CONTENT_TYPE,
            HeaderValue::from_static("application/javascript"),
        )],
        include_str!("../../../dist2d/elasticity2.js"),
    )
}

// Serve the gzipped WASM file
async fn serve_wasm() -> impl IntoResponse {
    // Load the pre-gzipped .wasm file
    let wasm_path = "dist2d/elasticity2_bg.wasm";
    let wasm_data = tokio::fs::read(wasm_path).await.unwrap();

    // Set headers for gzip encoding and wasm content type
    (
        [
            (CONTENT_TYPE, HeaderValue::from_static("application/wasm")),
            (CONTENT_ENCODING, HeaderValue::from_static("txt")),
        ],
        wasm_data,
    )
}
