# Terminal Rendering on iOS (SwiftTerm)

Deep-dive on making SwiftTerm render correctly on mobile screens.

## ANSI Background Color Stripping

Claude Code's TUI emits ANSI SGR sequences that set background colors. On small mobile screens, these create visual clutter. Strip background-only params from ESC[...m sequences:

```swift
func stripBackgroundColors(_ data: Data) -> Data {
    // Parse ESC [ params m sequences
    // Remove params: 40-47 (standard bg), 48 (extended bg), 100-107 (bright bg)
    // Preserve foreground, bold, underline, etc.
    var output = Data()
    var i = 0
    while i < data.count {
        if data[i] == 0x1B, i + 1 < data.count, data[i + 1] == 0x5B {
            // Found ESC[ — parse SGR params
            let (filtered, advance) = filterSGR(data, from: i)
            output.append(contentsOf: filtered)
            i += advance
        } else {
            output.append(data[i])
            i += 1
        }
    }
    return output
}
```

**Important:** Preserve UTF-8 multi-byte sequences — never inspect continuation bytes (0x80-0xBF) as potential escape characters.

## DEC Line Attribute Filtering

DEC line attributes (`ESC # 3` through `ESC # 8`) change line height and width. SwiftTerm renders these but they cause layout drift on mobile. Filter them:

```swift
// Strip ESC # [3-8] sequences
if data[i] == 0x1B, i + 1 < data.count, data[i + 1] == 0x23 {
    if i + 2 < data.count, (0x33...0x38).contains(data[i + 2]) {
        i += 3  // Skip entire sequence
        continue
    }
}
```

## Synchronized Output Mode

Some tools emit `ESC[?2026h` (begin) / `ESC[?2026l` (end) for synchronized output. SwiftTerm may not handle these gracefully on iOS. Filter them:

```swift
// Strip ESC[?2026h and ESC[?2026l
```

## CJK Font Cascade

For CJK character support, set a font cascade list:

```swift
let baseFontDesc = UIFont.monospacedSystemFont(ofSize: size, weight: .regular).fontDescriptor
let cascadeList = [
    UIFontDescriptor(fontAttributes: [.name: "PingFang SC"]),    // Simplified Chinese
    UIFontDescriptor(fontAttributes: [.name: "PingFang TC"]),    // Traditional Chinese
    UIFontDescriptor(fontAttributes: [.name: "Hiragino Sans"]),  // Japanese
    UIFontDescriptor(fontAttributes: [.name: "Apple SD Gothic Neo"]) // Korean
]
let cascadedDesc = baseFontDesc.addingAttributes([
    .cascadeList: cascadeList
])
terminalView.font = UIFont(descriptor: cascadedDesc, size: size)
```

## Preventing Vertical Drift

Two critical settings:

```swift
// 1. Disable custom box/block drawing glyphs
terminalView.customBlockGlyphs = false

// 2. Prevent safe-area insets from corrupting row positioning
terminalView.contentInsetAdjustmentBehavior = .never
```

Without these, terminal content drifts downward over time as lines accumulate.

## Adaptive Column Sizing

Calculate terminal columns from device screen width and font metrics:

```swift
func estimateColumns(screenWidth: CGFloat, fontSize: CGFloat) -> Int {
    let font = UIFont.monospacedSystemFont(ofSize: fontSize, weight: .regular)
    let charWidth = font.monospacedDigitWidth  // or measure "M" manually
    let usableWidth = screenWidth - (horizontalPadding * 2)
    return max(40, Int(usableWidth / charWidth))
}
```

Typical values: iPhone 16 Pro at 9pt Menlo yields ~62-65 columns.

## Theme Palettes

Define ANSI color palettes as arrays of 16 `UIColor` values (8 standard + 8 bright):

```swift
enum TerminalTheme: String, CaseIterable {
    case dark, solarizedDark, monokai

    var colors: [UIColor] {
        switch self {
        case .dark:
            return [/* black, red, green, yellow, blue, magenta, cyan, white,
                       brightBlack, brightRed, ... */]
        // ...
        }
    }

    var background: UIColor { /* ... */ }
    var foreground: UIColor { /* ... */ }
}
```

**Tip:** Mute white and bright-white colors slightly to prevent oversaturation on OLED screens.

## Terminal Size Reporting

Report terminal size at multiple points for reliability:

```swift
func onAppear() {
    // Immediate
    reportSize()
    // After initial layout
    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { reportSize() }
    // After animations settle
    DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) { reportSize() }
}
```

This handles cases where the terminal view's frame isn't finalized immediately.
