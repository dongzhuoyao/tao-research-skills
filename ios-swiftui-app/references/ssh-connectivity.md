# SSH Connectivity on iOS (Citadel / NIO)

Deep-dive on implementing SSH with PTY support using Citadel (pure Swift, NIO-based).

## Library Choice

| Library | Pros | Cons |
|---------|------|------|
| **Citadel** | Pure Swift, SPM, NIO-based, active development | PTY stdin requires workaround in some versions |
| **NMSSH** | Mature, full-featured | CocoaPods-only, pre-built arm64 conflicts with simulator |
| **libssh2** | C library, widely used | Requires building from source for iOS simulator |

**Recommendation:** Citadel for new projects. NMSSH if you need battle-tested ObjC compatibility (but expect build pain on Apple Silicon simulators).

## Connection Flow

```swift
@MainActor
func connect(profile: ServerProfile, credential: String) async {
    state = .connecting
    do {
        let client = try await SSHClient.connect(
            host: profile.host,
            port: profile.port,
            authenticationMethod: .password(
                username: profile.username,
                password: credential
            ),
            hostKeyValidator: .acceptAnything()  // Dev only — validate in production
        )
        self.sshClient = client
        state = .connected
    } catch {
        state = .error("Connection failed: \(error.localizedDescription)")
    }
}
```

## PTY Shell Setup

```swift
func startShell(cols: Int, rows: Int, startupCommand: String?, workingDirectory: String?) async {
    guard let client = sshClient else { return }

    do {
        let channel = try await client.requestPTY(
            .init(
                wantReply: true,
                term: "xterm-256color",  // Required for Claude Code
                terminalCharacterWidth: UInt32(cols),
                terminalRowHeight: UInt32(rows),
                terminalPixelWidth: 0,
                terminalPixelHeight: 0
            )
        )

        // Start shell
        try await channel.requestShell()

        // Set working directory
        if let dir = workingDirectory {
            try await channel.sendCommand("cd \(dir)\n")
        }

        // Run startup command (e.g., "claude")
        if let cmd = startupCommand {
            try await channel.sendCommand("\(cmd)\n")
        }

        // Start read loop
        startReadLoop(channel: channel)

    } catch {
        state = .error("Shell failed: \(error.localizedDescription)")
    }
}
```

## Read Loop (Background)

```swift
private func startReadLoop(channel: SSHChannel) {
    Task.detached { [weak self] in
        do {
            for try await data in channel.stream {
                let processed = self?.filterEscapeSequences(data) ?? data
                await MainActor.run {
                    self?.onDataReceived?(processed)
                }
            }
        } catch {
            await MainActor.run {
                self?.state = .disconnected
            }
        }
    }
}
```

**Critical:** Use `Task.detached` — a regular `Task` inherits `@MainActor` and blocks the UI.

## Text Input (PTY stdin)

```swift
func send(_ text: String) async {
    guard let channel = shellChannel else {
        // Fallback: executeCommand workaround
        return
    }
    if let data = text.data(using: .utf8) {
        try? await channel.send(data)
    }
}
```

**Known limitation (Citadel):** Some versions require an `executeCommand` workaround when direct channel stdin isn't exposed. Check your Citadel version's API surface.

## Window Resize

```swift
func resize(cols: Int, rows: Int) async {
    guard let channel = shellChannel else { return }
    try? await channel.sendWindowChange(
        columns: UInt32(cols),
        rows: UInt32(rows),
        pixelWidth: 0,
        pixelHeight: 0
    )
}
```

## Authentication Methods

### Password
```swift
.password(username: profile.username, password: credential)
```

### SSH Key (Ed25519)
```swift
let keyData = credential.data(using: .utf8)!
.privateKey(username: profile.username, privateKey: .init(ed25519Key: keyData))
```

### SSH Key from Base64
```swift
let decoded = Data(base64Encoded: base64String)!
let keyString = String(data: decoded, encoding: .utf8)!
// Then use .privateKey(...)
```

## Host Key Validation

```swift
// Development: accept anything (INSECURE)
hostKeyValidator: .acceptAnything()

// Production: pin known host keys
hostKeyValidator: .custom { hostKey in
    return hostKey == expectedPublicKey
}
```

## Disconnect & Cleanup

```swift
func disconnect() {
    shellChannel?.close()
    shellChannel = nil
    sshClient?.close()
    sshClient = nil
    state = .disconnected
    currentProfile = nil
}
```

Always nil out references to prevent retain cycles and stale state.

## NMSSH on Apple Silicon (Legacy)

If you must use NMSSH:

1. NMSSH CocoaPod bundles pre-built `arm64` static libs for **device only**
2. Apple Silicon simulators also target `arm64` → linker conflict
3. `EXCLUDED_ARCHS[sdk=iphonesimulator*] = arm64` does NOT work (modern simulators are arm64-only)
4. **Solution:** Build OpenSSL + libssh2 from source targeting arm64-simulator, create xcframeworks, vendor NMSSH ObjC source directly

Build script pattern:
```bash
# Build OpenSSL for simulator
./Configure ios64-cross --prefix=$OUTPUT -isysroot $(xcrun --sdk iphonesimulator --show-sdk-path)
# Patch Makefile to inject -isysroot
sed -i '' "s|-isysroot|-isysroot $(xcrun --sdk iphonesimulator --show-sdk-path)|g" Makefile
make -j$(sysctl -n hw.ncpu)
```
