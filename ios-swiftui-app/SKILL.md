---
name: ios-swiftui-app
description: Use when building iOS apps with SwiftUI, UIKit bridging, SSH terminal integration, or voice input pipelines. Applies to Swift 5.9+/iOS 17+ projects using SPM, xcodegen, Citadel, SwiftTerm, or AVFoundation. Triggers: "SwiftUI", "UIViewRepresentable", "ObservableObject", "@MainActor", "xcodegen", "SwiftTerm", "Citadel", "SSH", "STT", "AVAudioEngine", "iOS app", "terminal emulator", "voice input"
---

# iOS SwiftUI App Development

Battle-tested patterns for building production iOS apps with SwiftUI, UIKit integration, SSH connectivity, terminal emulation, and voice input pipelines. Extracted from building VibeWave Mobile — a voice-controlled SSH terminal client.

## When to Use

- Building an iOS app with SwiftUI + UIKit bridging (UIViewRepresentable)
- Integrating a terminal emulator (SwiftTerm) into a SwiftUI app
- Implementing SSH connectivity with Citadel (NIO-based)
- Adding voice input / speech-to-text (Apple Speech or custom server)
- Setting up iOS project with xcodegen + SPM
- Managing connection profiles with Keychain credential storage
- Handling `@MainActor` concurrency in ObservableObject services
- Wrapping AVAudioEngine for audio capture and transcription

## Core Architecture

### Layered Structure

```
SwiftUI Views Layer
    ├── MainView (primary UI + status bar)
    ├── Feature Views (profile mgmt, settings, pickers)
    └── UIViewRepresentable wrappers (UIKit bridging)
        ↓
Service Layer (@MainActor ObservableObjects)
    ├── ConnectionService (SSH/network)
    ├── InputService (voice/STT)
    └── DataStore (persistence)
        ↓
Model Layer (Codable structs + enums)
    ├── Domain models (profiles, settings, state enums)
    └── Keychain/UserDefaults helpers
        ↓
External Libraries (SPM)
    ├── SwiftTerm, Citadel, etc.
    └── iOS Frameworks (AVFoundation, Speech, Security)
```

### File Organization

```
mobile/
├── Package.swift              # SPM dependencies
├── project.yml                # xcodegen spec
├── AppName/
│   ├── App.swift              # @main, bootstrap, DI
│   ├── Views/                 # SwiftUI views + UIKit wrappers
│   ├── Services/              # @MainActor ObservableObject services
│   ├── Models/                # Data types, enums, stores
│   └── Info.plist             # Permissions + config
└── AppNameUITests/            # XCTest UI tests
```

## Patterns

### 1. Service Architecture (@MainActor + ObservableObject)

```swift
@MainActor
final class SSHService: ObservableObject {
    @Published var state: ConnectionState = .disconnected
    @Published var currentProfile: ServerProfile?

    // Callback for data from remote — set by view layer
    var onDataReceived: ((Data) -> Void)?

    func connect(profile: ServerProfile, credential: String) async {
        state = .connecting
        do {
            // ... connection logic
            state = .connected
        } catch {
            state = .error(error.localizedDescription)
        }
    }

    func disconnect() {
        // cleanup
        state = .disconnected
    }
}
```

**Rules:**
- Mark service classes `@MainActor` — all `@Published` mutations must be on main thread
- Use `Task.detached` for background I/O loops (shell read, audio capture) to avoid blocking MainActor
- Use `weak self` in closures that outlive the service lifetime
- State enums with associated error strings: `.error(String)` not Bool flags

### 2. UIViewRepresentable + Coordinator

```swift
struct TerminalUIView: UIViewRepresentable {
    @Binding var fontSize: CGFloat
    @EnvironmentObject var sshService: SSHService

    func makeUIView(context: Context) -> TerminalView {
        let tv = TerminalView(frame: .zero)
        tv.terminalDelegate = context.coordinator
        // Configure once
        return tv
    }

    func updateUIView(_ tv: TerminalView, context: Context) {
        // React to SwiftUI state changes (theme, font size)
        if tv.font.pointSize != fontSize {
            tv.font = UIFont.monospacedSystemFont(ofSize: fontSize, weight: .regular)
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    class Coordinator: NSObject, TerminalViewDelegate {
        var parent: TerminalUIView
        init(_ parent: TerminalUIView) { self.parent = parent }

        // Implement ALL required delegate methods
        func send(source: TerminalView, data: ArraySlice<UInt8>) {
            Task { @MainActor in
                await parent.sshService.send(String(bytes: data, encoding: .utf8) ?? "")
            }
        }
        // ... all other delegate methods
    }
}
```

**Rules:**
- Implement **all** required delegate methods — partial conformance causes silent failures
- Avoid naming your wrapper the same as the UIKit class it wraps (e.g., `SSHTerminalView` not `TerminalView`)
- Use Coordinator for delegate callbacks, not the struct itself
- Bridge delegate callbacks to `@MainActor` with `Task { @MainActor in ... }`

### 3. Dependency Injection via Environment

```swift
// App.swift — create services once at root
@main
struct VibeWaveApp: App {
    @StateObject private var sshService = SSHService()
    @StateObject private var sttService = STTService()
    @StateObject private var profileStore = ProfileStore()
    @StateObject private var appSettings = AppSettings.shared

    var body: some Scene {
        WindowGroup {
            MainView()
                .environmentObject(sshService)
                .environmentObject(sttService)
                .environmentObject(profileStore)
                .environmentObject(appSettings)
        }
    }
}

// Child views receive via @EnvironmentObject
struct MainView: View {
    @EnvironmentObject var sshService: SSHService
    // ...
}
```

### 4. Async/Await Patterns

```swift
// Callback-to-async bridging
func requestPermission() async -> Bool {
    await withCheckedContinuation { continuation in
        AVAudioApplication.requestRecordPermission { granted in
            continuation.resume(returning: granted)
        }
    }
}

// Background I/O loop (detached from MainActor)
func startShellReadLoop() {
    Task.detached { [weak self] in
        for try await data in shellStream {
            await MainActor.run {
                self?.onDataReceived?(data)
            }
        }
    }
}

// Timeout wrapper
func transcribeWithTimeout() async -> String? {
    await withThrowingTaskGroup(of: String?.self) { group in
        group.addTask { try await self.transcribe() }
        group.addTask { try await Task.sleep(for: .seconds(30)); return nil }
        // Return first result, cancel the other
        let result = try await group.next()
        group.cancelAll()
        return result ?? nil
    }
}
```

**Rules:**
- Never call `continuation.resume()` twice — guard with a flag (`hasResumed`)
- `Task.detached` for I/O loops; regular `Task` inherits MainActor context
- Always provide timeout wrappers for external service calls (STT, network)

### 5. Keychain Credential Storage

```swift
enum KeychainHelper {
    static func save(service: String, data: String) -> Bool {
        guard let data = data.data(using: .utf8) else { return false }
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecValueData as String: data
        ]
        SecItemDelete(query as CFDictionary)  // Remove existing
        return SecItemAdd(query as CFDictionary, nil) == errSecSuccess
    }

    static func load(service: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }
}
```

**Rules:**
- Index by UUID string (not hostname — allows multiple profiles per host)
- Delete Keychain entry when profile is deleted
- Use `kSecClassGenericPassword` for credentials
- Always `SecItemDelete` before `SecItemAdd` (upsert pattern)

### 6. UserDefaults Persistence for Models

```swift
class ProfileStore: ObservableObject {
    @Published var profiles: [ServerProfile] = []

    private let key = "saved_profiles"

    func load() {
        guard let data = UserDefaults.standard.data(forKey: key),
              let decoded = try? JSONDecoder().decode([ServerProfile].self, from: data)
        else { return }
        profiles = decoded
    }

    func save() {
        if let data = try? JSONEncoder().encode(profiles) {
            UserDefaults.standard.set(data, forKey: key)
        }
    }
}
```

### 7. Audio Capture (AVAudioEngine)

```swift
func startRecording() {
    let audioSession = AVAudioSession.sharedInstance()
    try audioSession.setCategory(.record, mode: .measurement)
    try audioSession.setActive(true)  // MUST be before inputNode access

    let inputNode = audioEngine.inputNode
    let format = inputNode.outputFormat(forBus: 0)

    inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) {
        [weak self] buffer, _ in
        self?.audioBuffers.append(buffer)
    }

    audioEngine.prepare()
    try audioEngine.start()
}
```

**Rules:**
- Set up audio session **before** accessing `audioEngine.inputNode` (crash otherwise)
- Record at hardware sample rate, convert to target rate (16kHz) afterward
- Use `installTap` for streaming capture, `removeTap` on stop
- Convert to WAV format with proper RIFF headers for compatibility

### 8. Build Configuration (xcodegen + SPM)

**project.yml:**
```yaml
name: AppName
options:
  bundleIdPrefix: com.company
  deploymentTarget:
    iOS: "17.0"
packages:
  SwiftTerm:
    url: https://github.com/migueldeicaza/SwiftTerm
    from: "1.2.0"
  Citadel:
    url: https://github.com/orlandos-nl/Citadel
    from: "0.7.0"
targets:
  AppName:
    type: application
    platform: iOS
    sources: [AppName]
    dependencies:
      - package: SwiftTerm
      - package: Citadel
    info:
      path: AppName/Info.plist
```

**Workflow:**
```bash
# Generate → Build → Install
xcodegen generate
xcodebuild -scheme AppName \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  build
xcrun simctl install booted build/Build/Products/Debug-iphonesimulator/AppName.app
```

### 9. Terminal Rendering Fixes

See [references/terminal-rendering.md](references/terminal-rendering.md) for:
- ANSI escape sequence filtering (background color stripping)
- DEC line attribute removal
- Font cascade for CJK characters
- Preventing vertical drift with custom block glyphs
- Safe-area inset handling

### 10. Testing Patterns (XCUITest)

```swift
class AppUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testConnectionFlow() throws {
        let app = XCUIApplication()
        app.launchEnvironment["VW_UI_TEST_MODE"] = "1"
        app.launch()

        // Wait for elements with timeout
        let button = app.buttons["connectButton"]
        XCTAssertTrue(button.waitForExistence(timeout: 5))
        button.tap()

        // Screenshot for debugging
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
```

## Anti-Patterns

### Architecture
- **Don't use `@Observable` macro for services with UIKit delegates** — stick with `ObservableObject` + `@Published` when bridging UIKit
- **Don't use `NavigationView`** — deprecated in iOS 16+, use `NavigationStack`
- **Don't use `.autocapitalization()`** — deprecated in iOS 15+, use `.textInputAutocapitalization()`
- **Don't name UIViewRepresentable wrapper the same as the UIKit class** — causes ambiguous type resolution
- **Don't use Bool flags for multi-state conditions** — use enums with associated values (`.connecting`, `.connected`, `.error(String)`)

### Concurrency
- **Don't mutate `@Published` from background threads** — always `@MainActor` or `await MainActor.run`
- **Don't use regular `Task { }` for background I/O** — it inherits MainActor; use `Task.detached`
- **Don't forget `[weak self]` in escaping closures** — especially `onDataReceived` callbacks that outlive views
- **Don't resume a continuation twice** — guard with `hasResumed` flag (Apple Speech callbacks can fire multiple times)

### Terminal Integration
- **Don't enable `customBlockGlyphs`** — causes vertical drift on iOS
- **Don't skip `contentInsetAdjustmentBehavior = .never`** — iOS safe-area insets corrupt terminal row positioning
- **Don't pass raw ANSI sequences to mobile terminal** — filter background colors (SGR 40-47, 48, 100-107) for readability on small screens
- **Don't hardcode terminal columns** — calculate from screen width and font metrics

### Audio/STT
- **Don't access `audioEngine.inputNode` before setting audio session** — crashes on iOS
- **Don't mix STT engine results silently** — if user selected Apple Speech, don't fall back to server without explicit opt-in
- **Don't skip timeout wrappers** — Apple Speech recognition can hang indefinitely

### Credentials
- **Don't store passwords in UserDefaults** — use iOS Keychain
- **Don't index Keychain by hostname** — use profile UUID (supports multiple accounts per host)
- **Don't forget to delete Keychain entries when profile is deleted** — stale credentials accumulate

### Build
- **Don't use CocoaPods for libraries with pre-built arm64 binaries** — they conflict with arm64 simulator on Apple Silicon; prefer SPM or build from source
- **Don't set podspec `platform` higher than Podfile target** — causes silent build failures

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `@Published` updates not reflecting in UI | Ensure service is `@MainActor` and mutations happen on main thread |
| Terminal text drifts vertically | Set `customBlockGlyphs = false` and `contentInsetAdjustmentBehavior = .never` |
| UIViewRepresentable delegate methods not called | Implement ALL required protocol methods, not just the ones you need |
| Audio recording crashes on launch | Set `AVAudioSession.setActive(true)` before accessing `inputNode` |
| Keychain save succeeds but load returns nil | Check `kSecAttrService` key matches between save and load |
| Simulator build fails on Apple Silicon | Use SPM instead of CocoaPods; avoid pre-built arm64 static libs |
| `Task.detached` closure can't access `self` | Capture `[weak self]` explicitly |
| Speech recognition hangs | Wrap in `withThrowingTaskGroup` with 30s timeout task |
| SwiftTerm keyboard doesn't appear | Post `toggleTerminalKeyboard` notification + call `becomeFirstResponder()` |
| xcodegen `generate` fails | Ensure `project.yml` has correct indentation and valid SPM package URLs |

## See Also

- `tmux` — terminal multiplexer patterns for remote sessions
- `zsh` — shell configuration and scripting
- `claude-code-config` — Claude Code integration patterns
- [references/terminal-rendering.md](references/terminal-rendering.md) — SwiftTerm rendering fixes
- [references/ssh-connectivity.md](references/ssh-connectivity.md) — Citadel SSH patterns
- [references/voice-input-stt.md](references/voice-input-stt.md) — STT pipeline architecture
