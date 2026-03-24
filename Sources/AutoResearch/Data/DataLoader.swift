import Foundation
import MLX

/// Streaming DataLoader that yields batches of (input, target) token pairs.
///
/// Reads pre-tokenized binary shards and packs sequences to the configured
/// sequence length with BOS-aligned document boundaries.

class DataLoader: Sequence, IteratorProtocol {
    let batchSize: Int
    let seqLen: Int
    let split: String
    let bosTokenId: Int
    private(set) var epoch: Int = 1

    private var shardPaths: [String]
    private var currentShardIdx: Int = 0
    private var tokenBuffer: [Int32] = []
    private var bufferPos: Int = 0

    init(tokenizer: BPETokenizer, batchSize: Int, seqLen: Int, split: String) {
        self.batchSize = batchSize
        self.seqLen = seqLen
        self.split = split
        self.bosTokenId = tokenizer.bosTokenId

        self.shardPaths = TokenShardReader.listShards(split: split)

        if shardPaths.isEmpty {
            print("Error: No pre-tokenized shards found for split '\(split)'")
            print("Run the prepare script first:")
            print("  python3 scripts/prepare_tokens.py")
            exit(1)
        }

        loadNextShard()
    }

    typealias Element = (x: MLXArray, y: MLXArray, epoch: Int)

    func next() -> Element? {
        let rowCapacity = seqLen + 1
        var allRows: [[Int32]] = []

        for _ in 0..<batchSize {
            var row: [Int32] = []
            while row.count < rowCapacity {
                ensureTokens(needed: rowCapacity - row.count)
                let remaining = rowCapacity - row.count
                let available = tokenBuffer.count - bufferPos
                let take = Swift.min(remaining, available)
                row.append(contentsOf: tokenBuffer[bufferPos..<(bufferPos + take)])
                bufferPos += take
            }
            allRows.append(Array(row.prefix(rowCapacity)))
        }

        // Convert to MLXArray [batchSize, seqLen+1] then split
        let flat = allRows.flatMap { $0 }
        let rowArray = MLXArray(flat, [batchSize, rowCapacity])
        let x = rowArray[0..., 0..<seqLen]
        let y = rowArray[0..., 1..<(seqLen + 1)]

        return (x, y, epoch)
    }

    /// Convenience to get one batch (matching Python's next(loader) pattern)
    func nextBatch() -> (x: MLXArray, y: MLXArray, epoch: Int) {
        return next()!
    }

    private func ensureTokens(needed: Int) {
        while (tokenBuffer.count - bufferPos) < needed {
            if !loadNextShard() {
                // Wrapped around — increment epoch
                epoch += 1
                currentShardIdx = 0
                _ = loadNextShard()
            }
        }
    }

    @discardableResult
    private func loadNextShard() -> Bool {
        guard currentShardIdx < shardPaths.count else {
            return false
        }
        let path = shardPaths[currentShardIdx]
        currentShardIdx += 1

        // Compact buffer: keep unread portion
        if bufferPos > 0 {
            tokenBuffer = Array(tokenBuffer[bufferPos...])
            bufferPos = 0
        }

        let newTokens = TokenShardReader.readShard(path: path)
        tokenBuffer.append(contentsOf: newTokens)
        return true
    }
}
