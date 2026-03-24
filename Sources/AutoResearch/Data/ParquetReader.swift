import Foundation

/// Read pre-tokenized binary shard files.
///
/// The reference implementation reads Parquet files containing text, then tokenizes on-the-fly.
/// For the Swift version, we use a two-phase approach:
/// 1. Pre-tokenize: A one-time Python script reads parquet shards, tokenizes with the trained
///    BPE tokenizer, and writes binary files (int32 token arrays) to ~/.cache/autoresearch/tokens/
/// 2. Training: Swift reads the binary token files directly — no Parquet or tokenizer needed.
///
/// Binary shard format: raw int32 array, one document's tokens after another,
/// separated by BOS tokens. File extension: .bin

class TokenShardReader {
    private static let tokenDir = NSHomeDirectory() + "/.cache/autoresearch/tokens"

    /// List all binary token shard files
    static func listShards(split: String, valShardName: String = "shard_06542") -> [String] {
        guard let files = try? FileManager.default.contentsOfDirectory(atPath: tokenDir) else {
            return []
        }
        let sorted = files.filter { $0.hasSuffix(".bin") }.sorted()

        if split == "train" {
            return sorted.filter { !$0.contains(valShardName) }.map { tokenDir + "/" + $0 }
        } else {
            return sorted.filter { $0.contains(valShardName) }.map { tokenDir + "/" + $0 }
        }
    }

    /// Read a binary shard file as an array of Int32 tokens
    static func readShard(path: String) -> [Int32] {
        guard let data = FileManager.default.contents(atPath: path) else {
            print("Warning: Could not read shard at \(path)")
            return []
        }
        let count = data.count / MemoryLayout<Int32>.size
        return data.withUnsafeBytes { buffer -> [Int32] in
            let ptr = buffer.bindMemory(to: Int32.self)
            return Array(ptr.prefix(count))
        }
    }

    /// Check if pre-tokenized shards exist
    static func shardsExist() -> Bool {
        let shards = listShards(split: "train")
        return !shards.isEmpty
    }
}
