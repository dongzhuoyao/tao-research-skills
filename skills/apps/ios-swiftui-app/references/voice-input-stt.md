# Voice Input & STT Pipeline on iOS

Deep-dive on implementing speech-to-text with Apple Speech and custom server backends.

## Architecture Overview

```
User taps VoiceButton
    → STTService.startRecording()
    → AVAudioEngine captures PCM audio
    → User taps again to stop
    → STTService.transcribe(engine, serverURL)
        ├── Apple Speech (on-device)
        └── Custom Server (network POST)
    → Transcribed text
    → SSHService.send(text) → PTY input
```

## Audio Recording

### Setup

```swift
@MainActor
final class STTService: ObservableObject {
    @Published var state: STTState = .idle  // idle | recording | processing | error
    @Published var lastTranscription: String = ""

    private let audioEngine = AVAudioEngine()
    private var audioBuffers: [AVAudioPCMBuffer] = []
}
```

### Permission Request

```swift
func requestPermissions() async -> Bool {
    // Microphone permission
    let micGranted = await withCheckedContinuation { continuation in
        AVAudioApplication.requestRecordPermission { granted in
            continuation.resume(returning: granted)
        }
    }

    // Speech recognition permission
    let speechGranted = await withCheckedContinuation { continuation in
        SFSpeechRecognizer.requestAuthorization { status in
            continuation.resume(returning: status == .authorized)
        }
    }

    return micGranted && speechGranted
}
```

### Start Recording

```swift
func startRecording() {
    audioBuffers.removeAll()

    do {
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.record, mode: .measurement)
        try audioSession.setActive(true)  // MUST be before inputNode access

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) {
            [weak self] buffer, _ in
            self?.audioBuffers.append(buffer)
        }

        audioEngine.prepare()
        try audioEngine.start()
        state = .recording
    } catch {
        state = .error("Recording failed: \(error.localizedDescription)")
    }
}
```

### Stop Recording & Convert

```swift
func stopRecording() -> Data {
    audioEngine.inputNode.removeTap(onBus: 0)
    audioEngine.stop()

    // Convert to 16kHz mono WAV
    let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 16000,
        channels: 1,
        interleaved: false
    )!

    let converter = AVAudioConverter(from: audioBuffers.first!.format, to: targetFormat)!
    var convertedBuffers: [AVAudioPCMBuffer] = []

    for buffer in audioBuffers {
        let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: targetFormat,
            frameCapacity: AVAudioFrameCount(
                Double(buffer.frameLength) * 16000.0 / buffer.format.sampleRate
            )
        )!

        try? converter.convert(to: outputBuffer, from: buffer)
        convertedBuffers.append(outputBuffer)
    }

    return encodeWAV(buffers: convertedBuffers, sampleRate: 16000)
}
```

### WAV Encoding

```swift
func encodeWAV(buffers: [AVAudioPCMBuffer], sampleRate: Int) -> Data {
    var pcmData = Data()
    for buffer in buffers {
        let ptr = buffer.floatChannelData![0]
        for i in 0..<Int(buffer.frameLength) {
            // Convert float32 to int16
            let sample = Int16(max(-1, min(1, ptr[i])) * 32767)
            withUnsafeBytes(of: sample.littleEndian) { pcmData.append(contentsOf: $0) }
        }
    }

    // RIFF WAV header
    var header = Data()
    header.append("RIFF".data(using: .ascii)!)
    header.append(UInt32(36 + pcmData.count).littleEndianData)
    header.append("WAVE".data(using: .ascii)!)
    header.append("fmt ".data(using: .ascii)!)
    header.append(UInt32(16).littleEndianData)          // Chunk size
    header.append(UInt16(1).littleEndianData)            // PCM format
    header.append(UInt16(1).littleEndianData)            // Mono
    header.append(UInt32(sampleRate).littleEndianData)   // Sample rate
    header.append(UInt32(sampleRate * 2).littleEndianData) // Byte rate
    header.append(UInt16(2).littleEndianData)             // Block align
    header.append(UInt16(16).littleEndianData)            // Bits per sample
    header.append("data".data(using: .ascii)!)
    header.append(UInt32(pcmData.count).littleEndianData)

    return header + pcmData
}
```

## Transcription Engines

### Apple Speech (On-Device)

```swift
func transcribeWithAppleSpeech(audioData: Data) async -> String? {
    guard let recognizer = SFSpeechRecognizer(), recognizer.isAvailable else { return nil }

    // Write WAV to temp file (SFSpeechRecognizer requires file URL)
    let tempURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString + ".wav")
    try? audioData.write(to: tempURL)
    defer { try? FileManager.default.removeItem(at: tempURL) }

    let request = SFSpeechURLRecognitionRequest(url: tempURL)
    request.shouldReportPartialResults = false

    return await withCheckedContinuation { continuation in
        var hasResumed = false  // Guard against double-resume

        recognizer.recognitionTask(with: request) { result, error in
            guard !hasResumed else { return }

            if let result = result, result.isFinal {
                hasResumed = true
                continuation.resume(returning: result.bestTranscription.formattedString)
            } else if let error = error {
                hasResumed = true
                continuation.resume(returning: nil)
            }
        }
    }
}
```

**Critical:** The `hasResumed` guard prevents a double-resume crash — Apple Speech callbacks can fire multiple times with partial results even when `shouldReportPartialResults = false`.

### Custom Server (VibeWave / Whisper)

```swift
func transcribeWithServer(audioData: Data, serverURL: String) async -> String? {
    guard let url = URL(string: serverURL + "/transcribe") else { return nil }

    var request = URLRequest(url: url)
    request.httpMethod = "POST"

    let boundary = UUID().uuidString
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

    var body = Data()
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n".data(using: .utf8)!)
    body.append("Content-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
    body.append(audioData)
    body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
    request.httpBody = body

    let (data, _) = try await URLSession.shared.data(for: request)
    let response = try JSONDecoder().decode(TranscriptionResponse.self, from: data)
    return response.text
}
```

## VoiceButton UI Pattern

```swift
struct VoiceButton: View {
    @EnvironmentObject var sttService: STTService
    @EnvironmentObject var sshService: SSHService

    @State private var outerPulse = false
    @State private var innerRing = false

    var body: some View {
        Button(action: toggleRecording) {
            ZStack {
                // Outer pulse (recording indicator)
                if sttService.state == .recording {
                    Circle()
                        .fill(Color.red.opacity(0.2))
                        .scaleEffect(outerPulse ? 1.8 : 1.0)
                        .opacity(outerPulse ? 0 : 0.5)
                        .animation(.easeOut(duration: 1.5).repeatForever(autoreverses: false), value: outerPulse)
                }

                // Main button
                Image(systemName: iconName)
                    .font(.title2)
                    .foregroundColor(iconColor)
                    .contentTransition(.symbolEffect(.replace))
            }
        }
        .disabled(!canRecord)
    }

    private var canRecord: Bool {
        sshService.state == .connected && sttService.state != .processing
    }

    private var iconName: String {
        switch sttService.state {
        case .recording: return "stop.circle.fill"
        case .processing: return "waveform"
        default: return "mic.fill"
        }
    }
}
```

**Design decisions:**
- **Tap-to-toggle** (not push-to-talk) — more ergonomic for longer dictation on mobile
- **Disabled when disconnected** — prevents recording audio that can't be delivered
- **Per-profile engine selection** — no implicit fallback between engines
- **Spring animation** (response 0.35, dampingFraction 0.7) — feels natural, not robotic

## STT Engine Selection

```swift
enum STTEngine: String, Codable, CaseIterable {
    case appleSpeech = "apple_speech"
    case vibewave = "vibewave"

    var displayName: String {
        switch self {
        case .appleSpeech: return "Apple Speech"
        case .vibewave: return "VibeWave Server"
        }
    }
}
```

Store per-profile in `ServerProfile.sttEngine`. The settings view also provides a global default via `AppSettings.defaultSTTEngine`.

## Timeout Protection

Always wrap transcription in a timeout:

```swift
func transcribeWithTimeout(engine: STTEngine, url: String?, audioData: Data) async -> String? {
    state = .processing
    defer { state = .idle }

    return await withThrowingTaskGroup(of: String?.self) { group in
        group.addTask {
            switch engine {
            case .appleSpeech:
                return await self.transcribeWithAppleSpeech(audioData: audioData)
            case .vibewave:
                return try await self.transcribeWithServer(audioData: audioData, serverURL: url!)
            }
        }
        group.addTask {
            try await Task.sleep(for: .seconds(30))
            return nil  // Timeout sentinel
        }

        let result = try? await group.next() ?? nil
        group.cancelAll()
        return result
    }
}
```
