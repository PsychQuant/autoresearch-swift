// swift-tools-version: 5.12

import PackageDescription

let package = Package(
    name: "autoresearch-swift",
    platforms: [
        .macOS("14.0"),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
    ],
    targets: [
        // Main executable
        .executableTarget(
            name: "AutoResearch",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
            ],
            path: "Sources/AutoResearch"
        ),
    ]
)
