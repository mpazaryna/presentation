# Apple Intelligence: Developer Integration Guide

## Getting Started with Apple Intelligence

### System Requirements

**Minimum Requirements:**
- iOS 18, iPadOS 18, macOS Sequoia, visionOS 2.0, or later
- Apple silicon device with appropriate generation
- Latest Xcode for development

**Supported Devices:**
- **iPhone:** A17 Pro chip or later
- **iPad:** M1 or later (except iPad mini, which requires A17 Pro)
- **Mac:** M1 or later
- **Apple Vision Pro:** Full support
- **Apple Watch:** Limited on-device capabilities

### Development Environment Setup

```bash
# Ensure Xcode is up to date
xcode-select --install

# Verify Swift version (Swift 5.9 or later recommended)
swift --version

# Clone Apple's sample repositories (optional)
git clone https://github.com/apple/swift-samples.git
```

## Foundation Models Framework

### Overview

The Foundation Models Framework provides direct access to Apple's on-device foundation model through Swift APIs. Developers can integrate sophisticated language understanding with just a few lines of code.

### Basic Integration (3 Lines of Code)

```swift
import Foundation
import AppKit

// 1. Create a request for text summarization
let request = GenerativeContentRequest(text: userInput)

// 2. Create the generator
let generator = GenerativeContentGenerator()

// 3. Generate result
let response = try await generator.generate(request: request)
```

### Supported Operations

#### Text Extraction

Extract structured information from unstructured text:

```swift
import Foundation

let request = TextExtractionRequest(
    input: "Email: john@example.com, Phone: 555-1234",
    format: .structured
)

let result = try await generator.extractText(request: request)
// Returns structured: ["email": "john@example.com", "phone": "555-1234"]
```

#### Summarization

Create concise summaries of longer content:

```swift
let request = SummarizationRequest(
    input: longDocumentText,
    summaryLength: .medium  // .short, .medium, .long
)

let summary = try await generator.summarize(request: request)
print(summary.text)  // Concise summary of document
```

#### Guided Generation

Generate text following specific patterns and constraints:

```swift
let guidedRequest = GuidedGenerationRequest(
    prompt: "Complete this story:",
    style: .creative,
    constraints: [
        .maxLength(200),
        .tone(.professional),
        .format(.paragraph)
    ]
)

let generated = try await generator.generate(request: guidedRequest)
```

#### Tool Calling

Integrate model with app-specific functions:

```swift
let toolRequest = ToolCallingRequest(
    query: "What's the weather tomorrow?",
    tools: [
        Tool(name: "getWeather",
             description: "Get weather information",
             parameters: ["date": .string, "location": .string])
    ]
)

let result = try await generator.callTools(request: toolRequest)
// Returns: {"tool": "getWeather", "parameters": {"date": "tomorrow", "location": "user_location"}}
```

### Error Handling

```swift
do {
    let result = try await generator.generate(request: request)
    // Use result
} catch GenerativeContentError.invalidInput {
    // Handle invalid input
    print("Invalid input provided")
} catch GenerativeContentError.serverError {
    // Handle server-side errors
    print("Private Cloud Compute temporarily unavailable")
} catch GenerativeContentError.unsupported {
    // Handle unsupported operations on this device
    print("Feature not available on this device")
} catch {
    // Handle other errors
    print("Error: \(error.localizedDescription)")
}
```

### Performance Considerations

```swift
// Optimal request batching
let requests = [request1, request2, request3]
let results = try await generator.generateBatch(requests: requests)
// More efficient than serial requests

// Caching results for repeated queries
let cachedGenerator = GenerativeContentGenerator(cachingEnabled: true)
let result1 = try await cachedGenerator.generate(request: identicalRequest)  // Fresh compute
let result2 = try await cachedGenerator.generate(request: identicalRequest)  // From cache
```

## Writing Tools Integration

### Native Text View Integration

#### UITextView (iOS/iPadOS)

```swift
import UIKit

class DocumentViewController: UIViewController {
    @IBOutlet weak var textView: UITextView!

    override func viewDidLoad() {
        super.viewDidLoad()

        // Configure Writing Tools
        textView.writingToolsAllowedInputOptions = [.plainText, .richText]
        textView.writingToolsDelegate = self

        // Optional: Set behavior mode
        textView.writingToolsBehavior = .default  // or .limited, .none
    }
}

extension DocumentViewController: UITextViewDelegate {
    func textViewWritingToolsWillBegin(_ textView: UITextView) {
        // Pause syncing, disable other edits
        pauseBackendSync()
    }

    func textViewWritingToolsDidEnd(_ textView: UITextView) {
        // Resume syncing, update UI
        resumeBackendSync()
    }

    func textView(_ textView: UITextView,
                  writingToolsIgnoredRangesIn range: NSRange) -> [NSRange] {
        // Return ranges to exclude from Writing Tools (e.g., code blocks)
        return findCodeBlockRanges(in: range)
    }
}
```

#### NSTextView (macOS)

```swift
import AppKit

class DocumentWindow: NSWindow {
    @IBOutlet weak var textView: NSTextView!

    override func awakeFromNib() {
        super.awakeFromNib()

        // NSTextView requires NSServicesMenuRequestor protocol
        textView.delegate = self
    }
}

extension DocumentWindow: NSServicesMenuRequestor {
    override func validRequestor(
        forSendType sendType: NSPasteboard.PasteboardType?,
        returnType: NSPasteboard.PasteboardType?
    ) -> Any? {
        // Enable Writing Tools for this view
        return self
    }
}
```

#### WKWebView (Web Content)

```swift
import WebKit

class WebContentViewController: UIViewController {
    @IBOutlet weak var webView: WKWebView!

    // WKWebView automatically supports Writing Tools for
    // contenteditable elements in HTML

    func loadDocument() {
        let html = """
        <html>
            <body>
                <textarea id="editor" contenteditable="true">
                    Edit me with Writing Tools...
                </textarea>
            </body>
        </html>
        """
        webView.loadHTMLString(html, baseURL: nil)
    }
}
```

### Custom Text Input Integration

```swift
import UIKit

class CustomTextView: UIView, UITextInput {
    var text: String = ""

    // Required UITextInput properties and methods
    var selectedTextRange: UITextRange?
    var markedTextRange: UITextRange?
    var markedTextStyle: [NSAttributedString.Key: Any]?

    // Implement required methods...
    func replace(_ range: UITextRange, withText text: String) {
        // Handle text replacement
    }

    // Enable Writing Tools through UITextInteraction
    func enableWritingTools() {
        let interaction = UITextInteraction(mode: .editable)
        addInteraction(interaction)
    }
}
```

## Image Playground Integration

### Image Generation API

```swift
import ImagePlayground

// Request image generation
let request = ImagePlaygroundRequest(
    concept: "A serene mountain landscape at sunset",
    style: .illustration,
    userPhotos: selectedPhotosFromLibrary
)

// Get generated image
let image = try await ImagePlayground.generate(request: request)

// Use generated image
imageView.image = image
```

### Image Playground Presentation

```swift
import ImagePlayground

// Present Image Playground UI
let playgroundViewController = ImagePlaygroundViewController(delegate: self)
present(playgroundViewController, animated: true)

// Handle generated images
extension ViewController: ImagePlaygroundViewControllerDelegate {
    func imagePlaygroundViewController(
        _ viewController: ImagePlaygroundViewController,
        didGenerateImage image: UIImage
    ) {
        // Use generated image
        handleGeneratedImage(image)
        dismiss(animated: true)
    }
}
```

## Siri and Shortcuts Integration

### App Intents Framework

```swift
import AppIntents

// Define custom intent for Siri
struct DocumentSummaryIntent: AppIntent {
    static var title: LocalizedStringResource = "Summarize Document"
    static var description: LocalizedStringResource =
        "Create a summary of the current document"

    @Parameter(title: "Document Content")
    var content: String

    @Parameter(title: "Summary Length", default: "medium")
    var length: String

    func perform() async throws -> some IntentResult {
        let summary = try await summarizeContent(content, length: length)
        return .result(value: summary)
    }
}

// Enable in Siri and Shortcuts
extension DocumentSummaryIntent: PredictableIntent {
    static var assetColoring: AssetColoring = .blue

    nonisolated static var allCases: [DocumentSummaryIntent] = [
        DocumentSummaryIntent(content: "", length: "medium")
    ]
}
```

### Voice Command Handling

```swift
import SiriKit

// Implement INRequestHandler for voice interactions
class SiriIntentHandler: INExtension {
    override func handler(for intent: INIntent) -> Any {
        if intent is DocumentSummaryIntent {
            return DocumentSummaryIntentHandler()
        }
        return self
    }
}

class DocumentSummaryIntentHandler: NSObject, DocumentSummaryIntentHandling {
    func handle(intent: DocumentSummaryIntent) async -> DocumentSummaryIntentResponse {
        do {
            let summary = try await performSummary(intent: intent)
            return DocumentSummaryIntentResponse(code: .success, userActivity: nil)
        } catch {
            return DocumentSummaryIntentResponse(code: .failure, userActivity: nil)
        }
    }
}
```

## Visual Intelligence Integration

### Using App Intents for Visual Search

```swift
import AppIntents

struct VisualSearchIntent: AppIntent {
    static var title: LocalizedStringResource = "Search with Photo"
    static var description: LocalizedStringResource =
        "Search the app using a visual reference"

    @Parameter(title: "Image")
    var image: IntentFile

    func perform() async throws -> some IntentResult {
        let uiImage = try UIImage(data: image.data)
        let results = try await searchByImage(uiImage)
        return .result(value: results)
    }
}
```

## Genmoji Integration

### Custom Emoji Generation

```swift
import EmojiUIKit

// Request Genmoji generation
let request = GenmojiRequest(
    description: "A happy robot developer",
    style: .animated
)

let emoji = try await Genmoji.generate(request: request)

// Use as text or image
textView.insertText(emoji)
```

## Best Practices for Apple Intelligence Integration

### 1. Progressive Enhancement

```swift
// Check device capability
if #available(iOS 18, *) {
    // Use Apple Intelligence features
    enableWritingTools()
    enableVisualIntelligence()
} else {
    // Fallback for older devices
    showBasicTextEditing()
}
```

### 2. Privacy-First Design

```swift
// Always use on-device processing when possible
let request = GenerativeContentRequest(
    text: userInput,
    preferLocalProcessing: true  // Use on-device model
)

// Only use Private Cloud Compute for necessary operations
let advancedRequest = AdvancedProcessingRequest(
    text: complexQuery,
    allowCloudProcessing: true  // Only if needed
)
```

### 3. Error Handling and Fallbacks

```swift
do {
    let result = try await generator.generate(request: request)
    updateUI(with: result)
} catch GenerativeContentError.unsupported {
    // Gracefully disable feature on unsupported devices
    disableFeature()
} catch GenerativeContentError.serverError {
    // Retry or show offline experience
    showOfflineMessage()
} catch {
    // Generic error handling
    showError("Processing failed. Please try again.")
}
```

### 4. Performance Optimization

```swift
// Batch requests for efficiency
let results = try await generator.generateBatch(
    requests: [req1, req2, req3]
)

// Cache results to avoid redundant processing
let cache = NSCache<NSString, NSString>()
if let cached = cache.object(forKey: query as NSString) {
    return cached as String
}

// Use async/await for responsive UI
Task {
    let result = try await performExpensiveOperation()
    DispatchQueue.main.async {
        updateUI(with: result)
    }
}
```

### 5. User Consent and Control

```swift
// Request permission for features
AVAuthorizationStatus.requestAccess(mediaType: .photo) { granted in
    if granted {
        enableImageAnalysis()
    }
}

// Provide feature toggles
UserDefaults.standard.set(true, forKey: "writingToolsEnabled")

// Respect user choices
if UserDefaults.standard.bool(forKey: "writingToolsEnabled") {
    configureWritingTools()
}
```

## Debugging and Testing

### Enable Debug Logging

```swift
// Enable verbose logging
import OSLog

let logger = Logger(subsystem: "com.example.app", category: "AI")

logger.debug("Starting text generation with prompt: \(prompt)")
logger.error("Generation failed: \(error)")
```

### Test on Real Devices

```swift
// Test on actual Apple silicon devices
// Simulator support limited for some features
// Device testing critical for:
// - Writing Tools interaction
// - Private Cloud Compute
// - Real-time performance

// Use Xcode's device simulator
// Test on:
// - iPad with M1
// - Mac with Apple silicon
// - iPhone with A17 Pro
```

### Monitor Performance

```swift
import MetricKit

// Track performance metrics
let metricsManager = MetricsManager()

// Measure inference time
let startTime = CFAbsoluteTimeGetCurrent()
let result = try await generator.generate(request: request)
let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime

logger.info("Generation took \(timeElapsed)s")
```

## Deployment Considerations

### App Store Requirements

- Clearly document Apple Intelligence features
- Disclose use of Private Cloud Compute
- Handle devices without support gracefully
- Provide fallback functionality

### System Requirements in App Store

```xml
<!-- Manifest for App Store Connect -->
<requires>
    <minimum_os version="18.0" />
    <requires_device type="iphone" model="A17Pro" />
    <requires_device type="ipad" model="M1" />
</requires>
```

### User Documentation

- Explain what Apple Intelligence features do
- Clarify on-device vs. cloud processing
- Document privacy guarantees
- Provide feature settings documentation

## Resources and References

### Official Documentation
- Apple Intelligence Framework documentation
- Writing Tools Integration Guide
- App Intents Framework reference
- WWDC 2024 sessions on Apple Intelligence

### Sample Code
- Apple's GitHub repositories
- WWDC sample code projects
- Third-party integration examples

### Support
- Developer Forums
- Apple Support Resources
- WWDC Office Hours (during conference)
- Technical Support Incidents (TSIs)

---

**Source:** Official Apple Intelligence Developer Documentation
