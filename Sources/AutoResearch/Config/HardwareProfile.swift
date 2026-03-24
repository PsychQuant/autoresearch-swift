import Foundation
import Metal
import IOKit

// MARK: - Chip Profile Database

struct ChipProfile {
    let deviceBatchSize: Int
    let depth: Int
    let totalBatchLog2: Int
    let sequenceLen: Int
    let evalBatchSize: Int
}

private let chipProfiles: [String: [Int: ChipProfile]] = {
    // [chipFamily: [memoryGB: profile]]
    // Memory tiers determine profile within each chip family
    var db: [String: [Int: ChipProfile]] = [:]

    // M1 family
    let m1Base = ChipProfile(deviceBatchSize: 8, depth: 4, totalBatchLog2: 14, sequenceLen: 512, evalBatchSize: 64)
    let m1Pro  = ChipProfile(deviceBatchSize: 16, depth: 4, totalBatchLog2: 15, sequenceLen: 1024, evalBatchSize: 64)
    let m1Max  = ChipProfile(deviceBatchSize: 32, depth: 6, totalBatchLog2: 16, sequenceLen: 2048, evalBatchSize: 128)
    let m1Ultra = ChipProfile(deviceBatchSize: 64, depth: 8, totalBatchLog2: 18, sequenceLen: 2048, evalBatchSize: 256)
    db["M1"] = [8: m1Base, 16: m1Base, 32: m1Pro, 64: m1Max, 128: m1Ultra]

    // M2 family
    db["M2"] = [8: m1Base, 16: m1Base, 24: m1Pro, 32: m1Pro, 64: m1Max, 96: m1Ultra, 128: m1Ultra, 192: m1Ultra]

    // M3 family
    let m3Pro = ChipProfile(deviceBatchSize: 24, depth: 6, totalBatchLog2: 16, sequenceLen: 2048, evalBatchSize: 128)
    let m3Max = ChipProfile(deviceBatchSize: 48, depth: 8, totalBatchLog2: 17, sequenceLen: 2048, evalBatchSize: 192)
    db["M3"] = [8: m1Base, 16: m1Base, 18: m1Pro, 36: m3Pro, 48: m3Pro, 64: m3Max, 96: m3Max, 128: m1Ultra]

    // M4 family
    let m4Base = ChipProfile(deviceBatchSize: 12, depth: 4, totalBatchLog2: 15, sequenceLen: 1024, evalBatchSize: 64)
    let m4Pro  = ChipProfile(deviceBatchSize: 32, depth: 6, totalBatchLog2: 17, sequenceLen: 2048, evalBatchSize: 128)
    let m4Max  = ChipProfile(deviceBatchSize: 64, depth: 8, totalBatchLog2: 18, sequenceLen: 2048, evalBatchSize: 256)
    db["M4"] = [16: m4Base, 24: m4Base, 32: m4Pro, 48: m4Pro, 64: m4Max, 128: m4Max]

    return db
}()

// MARK: - Hardware Profile

struct HardwareProfile {
    let chipName: String
    let chipFamily: String
    let gpuCores: Int
    let totalMemoryGB: Int
    let availableMemoryGB: Int

    static func detect() -> HardwareProfile {
        let chipName = detectChipName()
        let chipFamily = extractChipFamily(from: chipName)

        var gpuCores = 0
        if let device = MTLCreateSystemDefaultDevice() {
            gpuCores = device.maxThreadsPerThreadgroup.width
        }

        let totalMemoryBytes = ProcessInfo.processInfo.physicalMemory
        let totalMemoryGB = Int(totalMemoryBytes / (1024 * 1024 * 1024))

        // Available memory via mach_task_basic_info
        var taskInfo = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &taskInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        let usedBytes = kr == KERN_SUCCESS ? Int(taskInfo.resident_size) : 0
        let availableGB = max(0, totalMemoryGB - Int(usedBytes / (1024 * 1024 * 1024)))

        return HardwareProfile(
            chipName: chipName,
            chipFamily: chipFamily,
            gpuCores: gpuCores,
            totalMemoryGB: totalMemoryGB,
            availableMemoryGB: availableGB
        )
    }

    func applyDefaults(to config: inout ExperimentConfig) {
        print("Hardware: \(chipName) (\(chipFamily)), \(totalMemoryGB)GB RAM, GPU cores: ~\(gpuCores)")
        print("Available memory: ~\(availableMemoryGB)GB")

        // Try chip profile database first
        if let familyProfiles = chipProfiles[chipFamily] {
            // Find closest memory tier (round down)
            let sortedTiers = familyProfiles.keys.sorted()
            let matchedTier = sortedTiers.last(where: { $0 <= totalMemoryGB }) ?? sortedTiers.first!
            let profile = familyProfiles[matchedTier]!

            config.deviceBatchSize = profile.deviceBatchSize
            config.depth = profile.depth
            config.totalBatchLog2 = profile.totalBatchLog2
            config.sequenceLen = profile.sequenceLen
            config.evalBatchSize = profile.evalBatchSize

            print("Profile: \(chipFamily) \(matchedTier)GB tier")
        } else {
            // Unknown chip — fallback to memory-based heuristics
            print("Unknown chip family '\(chipFamily)' — using memory-based defaults")
            applyMemoryFallback(to: &config)
        }

        // Available memory check
        if availableMemoryGB < totalMemoryGB / 2 {
            let ratio = max(0.25, Double(availableMemoryGB) / Double(totalMemoryGB))
            let reduced = max(4, Int(Double(config.deviceBatchSize) * ratio))
            print("Warning: Available memory is below 50% — reducing device_batch_size \(config.deviceBatchSize) → \(reduced)")
            config.deviceBatchSize = reduced
        }
    }

    private func applyMemoryFallback(to config: inout ExperimentConfig) {
        if totalMemoryGB >= 96 {
            config.deviceBatchSize = 64; config.depth = 8; config.totalBatchLog2 = 18; config.evalBatchSize = 256
        } else if totalMemoryGB >= 32 {
            config.deviceBatchSize = 32; config.depth = 6; config.totalBatchLog2 = 17; config.evalBatchSize = 128
        } else if totalMemoryGB >= 16 {
            config.deviceBatchSize = 8; config.depth = 4; config.sequenceLen = 512; config.totalBatchLog2 = 14; config.evalBatchSize = 64
        } else {
            config.deviceBatchSize = 4; config.depth = 2; config.sequenceLen = 256; config.totalBatchLog2 = 13; config.evalBatchSize = 32
        }
    }

    // MARK: - Chip detection

    private static func detectChipName() -> String {
        // Try IORegistry for Apple Silicon chip name
        if let name = ioRegistryChipName() {
            return name
        }
        // Fallback to sysctl
        var size: Int = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        if size > 0 {
            var buffer = [CChar](repeating: 0, count: size)
            sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0)
            return String(cString: buffer)
        }
        return "Unknown"
    }

    private static func ioRegistryChipName() -> String? {
        let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("IOPlatformExpertDevice"))
        guard service != 0 else { return nil }
        defer { IOObjectRelease(service) }

        if let data = IORegistryEntryCreateCFProperty(service, "chip-id" as CFString, kCFAllocatorDefault, 0) {
            // chip-id exists — we're on Apple Silicon
            // Get the model name from product-name or compatible
            if let nameData = IORegistryEntryCreateCFProperty(service, "product-name" as CFString, kCFAllocatorDefault, 0) {
                if let data = nameData.takeRetainedValue() as? Data {
                    return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .controlCharacters)
                }
            }
        }

        // Try hw.chip sysctl (available on newer macOS)
        var size: Int = 0
        if sysctlbyname("hw.chip", nil, &size, nil, 0) == 0, size > 0 {
            var buffer = [CChar](repeating: 0, count: size)
            sysctlbyname("hw.chip", &buffer, &size, nil, 0)
            return String(cString: buffer)
        }

        return nil
    }

    private static func extractChipFamily(from name: String) -> String {
        // Extract "M4", "M3", "M2", "M1" from chip name like "Apple M4 Max"
        let pattern = #"M\d+"#
        if let range = name.range(of: pattern, options: .regularExpression) {
            return String(name[range])
        }
        return "Unknown"
    }
}
