import Foundation
import MLX

/// BPE Tokenizer that loads a pre-trained tokenizer from ~/.cache/autoresearch/tokenizer/.
///
/// Strategy: The tokenizer data was created by the reference Python prepare.py script.
/// The merge table is stored in Python's serialization format which we don't read directly.
/// Instead we:
/// 1. Load token_bytes.npy for evaluation (bytes-per-token lookup)
/// 2. Read pre-tokenized binary shards for training data
/// 3. Data preparation is a one-time step using the reference Python script
///
/// This means data prep uses Python once, but all training runs are pure Swift.

class BPETokenizer {
    let vocabSize: Int
    let tokenBytes: MLXArray  // [vocabSize] int32, bytes per token for BPB calculation
    let bosTokenId: Int

    private static let cacheDir = NSHomeDirectory() + "/.cache/autoresearch"
    private static let tokenizerDir = cacheDir + "/tokenizer"
    private static let dataDir = cacheDir + "/data"

    init(vocabSize: Int, tokenBytes: MLXArray, bosTokenId: Int) {
        self.vocabSize = vocabSize
        self.tokenBytes = tokenBytes
        self.bosTokenId = bosTokenId
    }

    static func fromDirectory() throws -> BPETokenizer {
        let tokenBytesPath = tokenizerDir + "/token_bytes.npy"

        guard FileManager.default.fileExists(atPath: tokenBytesPath) else {
            print("Error: Tokenizer not found at \(tokenizerDir)")
            print("Run data preparation first:")
            print("  cd references/autoresearch-mlx && uv run prepare.py")
            exit(1)
        }

        // Load token_bytes.npy (numpy int32 array)
        let tokenBytes = try loadNpyInt32(path: tokenBytesPath)
        let vocabSize = tokenBytes.shape[0]

        // BOS token is the first special token (index = vocab_size - 4)
        let bosTokenId = vocabSize - 4

        return BPETokenizer(vocabSize: vocabSize, tokenBytes: tokenBytes, bosTokenId: bosTokenId)
    }

    /// List parquet files in data directory
    static func listDataShards() -> [String] {
        guard let files = try? FileManager.default.contentsOfDirectory(atPath: dataDir) else {
            return []
        }
        return files
            .filter { $0.hasSuffix(".parquet") && !$0.hasSuffix(".tmp") }
            .sorted()
            .map { dataDir + "/" + $0 }
    }

    static func valShardPath() -> String {
        return dataDir + "/shard_06542.parquet"
    }
}

// MARK: - NPY loader for int32 arrays

func loadNpyInt32(path: String) throws -> MLXArray {
    guard let data = FileManager.default.contents(atPath: path) else {
        throw TokenizerError.fileNotFound(path)
    }

    // NPY format: 6-byte magic (\x93NUMPY) + version + header + data
    let magic = Array(data.prefix(6))
    guard magic == [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59] else {
        throw TokenizerError.invalidFormat("Not a valid .npy file: \(path)")
    }

    let majorVersion = data[6]

    let headerLen: Int
    let headerStart: Int

    if majorVersion == 1 {
        headerLen = Int(data[8]) | (Int(data[9]) << 8)
        headerStart = 10
    } else {
        headerLen = Int(data[8]) | (Int(data[9]) << 8) | (Int(data[10]) << 16) | (Int(data[11]) << 24)
        headerStart = 12
    }

    let dataStart = headerStart + headerLen

    // Parse shape from header string
    let headerData = data[headerStart..<(headerStart + headerLen)]
    let headerStr = String(data: headerData, encoding: .ascii) ?? ""

    var shape: [Int] = []
    if let shapeRange = headerStr.range(of: #"'shape':\s*\(([^)]*)\)"#, options: .regularExpression) {
        let shapeStr = String(headerStr[shapeRange])
        let numbers = shapeStr.components(separatedBy: CharacterSet.decimalDigits.inverted)
            .filter { !$0.isEmpty }
            .dropFirst() // drop "shape"
            .compactMap { Int($0) }
        shape = Array(numbers)
    }

    // Read int32 data
    let rawData = data.subdata(in: dataStart..<data.count)
    let count = rawData.count / 4

    let int32Array = rawData.withUnsafeBytes { buffer -> [Int32] in
        let ptr = buffer.bindMemory(to: Int32.self)
        return Array(ptr)
    }

    return MLXArray(int32Array, shape.isEmpty ? [count] : shape)
}

enum TokenizerError: Error {
    case fileNotFound(String)
    case invalidFormat(String)
}
