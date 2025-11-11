# Writing Tools: Apple Intelligence Text Integration

## Overview

Writing Tools is a built-in Apple Intelligence feature that helps users refine text through intelligent processing. The system processes text using Apple Intelligence's language capabilities and integrates seamlessly with existing text interfaces in apps. Developers can enable Writing Tools with automatic support or implement custom integration for specialized text handling.

## Key Capabilities

Writing Tools enables users to:

### Proofread
- Identify spelling and grammar errors
- Provide correction suggestions
- Improve text clarity
- Enforce writing style consistency

### Rewrite
- Adjust tone of text (friendly, professional, concise, etc.)
- Maintain meaning while changing style
- Support multiple output variations
- Generate alternative phrasings

### Summarize
- Extract key points from longer texts
- Generate concise summaries
- Identify important information
- Reduce text length while preserving meaning

### Transform
- Create lists from descriptive text
- Generate tables from content
- Reorganize information structure
- Format text for specific use cases

## Native Text View Support

Writing Tools integrates automatically with standard Apple text controls without requiring developer implementation:

### UITextView (iOS, iPadOS)

**Requirements:**
- TextKit 2 required for full experience
- TextKit 1 provides basic support

**Automatic Behavior:**
- Writing Tools appear in text editing context menus
- Works automatically without code
- Standard interaction model for users

### NSTextView (macOS)

**Features:**
- Automatic integration
- Native macOS text editing experience
- Full Writing Tools support
- Standard document editing integration

### WKWebView (Web Content)

**Support:**
- Works with web text content
- Supports editable web form fields
- Standard web text interaction
- JavaScript-based text handling supported

### Implementation Statement

According to Apple's documentation: "if you are using a UITextView, NSTextView or WKWebView, it just works."

## Configuration Options

For apps using native text views, developers can control Writing Tools behavior:

### Behavior Configuration

**`.default` (Default Behavior)**
- Inline editing experience
- Writing Tools appear in context menus
- Standard user interaction model
- Recommended for most applications

**`.limited`**
- Panel-only view without inline editing
- Processing occurs in separate panel
- Results displayed separately
- Useful for constrained UI spaces

**`.none`**
- Opt-out of Writing Tools completely
- Feature disabled for specific text view
- Used for sensitive or special cases
- Can be set per-text-view

### Format Support Declaration

Developers specify supported text formats via `writingToolsAllowedInputOptions`:

**Plain Text Support**
- Basic text without formatting
- Standard text input
- Simplest format support

**Rich Text Support**
- Formatted text (bold, italic, etc.)
- Styling information preserved
- Complex document handling
- Full formatting support

**Table Support**
- Structured table content
- Row and column handling
- Table formatting preservation
- Useful for data-heavy applications

## Custom Text View Implementation

For apps implementing custom text handling, developers can integrate Writing Tools through standard protocols:

### iOS/iPadOS Implementation

#### UITextInteraction Adoption

Adopt the `UITextInteraction` protocol to enable Writing Tools:

```swift
// Adopt UITextInteraction protocol
class CustomTextView: UIView, UITextInteraction {
    // Implement required methods
    // Automatically gains Writing Tools support
}
```

#### UITextInput Protocol

Alternatively, implement the `UITextInput` protocol:

```swift
class CustomTextView: UIView, UITextInput {
    // Implement text input protocol
    // Enables Writing Tools integration
}
```

### macOS Implementation

#### NSServicesMenuRequestor

Conform to `NSServicesMenuRequestor` protocol:

```swift
class CustomTextView: NSView, NSServicesMenuRequestor {
    override func validRequestor(forSendType sendType: NSPasteboard.PasteboardType?,
                                returnType: NSPasteboard.PasteboardType?) -> Any? {
        // Implement validation logic
        // Enables Writing Tools support
    }
}
```

## Lifecycle Methods and Delegates

### Delegate Protocol

Writing Tools provides delegate callbacks for lifecycle management:

#### textViewWritingToolsWillBegin()

Called when Writing Tools processing starts:

```swift
func textViewWritingToolsWillBegin() {
    // Pause syncing with backend
    // Disable other edits
    // Save current state
    // Pause animation
}
```

**Use Cases:**
- Pause real-time syncing to servers
- Disable other text edits
- Save application state
- Pause UI animations

#### textViewWritingToolsDidEnd()

Called when Writing Tools processing completes:

```swift
func textViewWritingToolsDidEnd() {
    // Restore app state
    // Resume syncing
    // Update UI
    // Re-enable edits
}
```

**Use Cases:**
- Restore application state
- Resume real-time syncing
- Update UI elements
- Re-enable user interactions

### Status Checking

**isWritingToolsActive Property**

Check current Writing Tools status at any time:

```swift
if textView.isWritingToolsActive {
    // Writing Tools processing in progress
    // Don't modify text
} else {
    // Safe to modify text
}
```

## Protected Ranges

Developers can exclude specific text from Writing Tools processing:

### Use Cases

**Code Blocks**
- Exclude programming code from rewriting
- Preserve syntax and structure
- Prevent language transformations

**Quotations**
- Preserve exact quoted text
- Prevent quote modification
- Maintain attribution

**Technical Terms**
- Protect specific terminology
- Preserve product names
- Keep technical jargon unchanged

**Special Formatting**
- Preserve manually styled text
- Keep specific formatting intact
- Protect data structures

### Implementation

Use the `textView(_:writingToolsIgnoredRangesIn:)` delegate method:

```swift
func textView(_ textView: UITextView,
              writingToolsIgnoredRangesIn range: NSRange) -> [NSRange] {
    var ignoredRanges: [NSRange] = []

    // Identify code blocks in range
    // Add code block ranges to ignoredRanges

    // Identify quotations
    // Add quotation ranges to ignoredRanges

    return ignoredRanges
}
```

## User Interaction Model

### Text Selection

1. User selects text in application
2. Context menu appears with Writing Tools option
3. User taps Writing Tools option
4. Selection panel appears with capabilities

### Options Presentation

Writing Tools presents available options:

**For Selected Text:**
- Proofread: Suggests corrections
- Rewrite: Offers tone options (friendly, professional, concise, etc.)
- Summarize: Generates summary text
- Transform: Lists available transformations

**User Selection:**
- User selects desired option
- System processes text
- Displays results for review
- User accepts or rejects changes

### Result Handling

**User Review:**
- Results displayed for user confirmation
- Before/after comparison available
- User can accept or reject
- Can iterate on different options

**Acceptance:**
- User accepts result
- Text replaced with processed version
- Edit integrated into document
- Undo available if needed

## Integration Best Practices

### Standard UI Approach

**Recommendation:** Use standard UITextView, NSTextView, or WKWebView when possible
- Automatic Writing Tools support
- No custom implementation needed
- Better user experience consistency
- Easier maintenance

### Custom Text Handling

**When Needed:**
- Custom text rendering
- Specialized text types
- Unique interaction model
- Performance requirements

**Implementation Guidelines:**
- Adopt appropriate protocol (UITextInput or UITextInteraction)
- Implement delegate methods properly
- Handle lifecycle events correctly
- Test thoroughly on target devices

### Format Configuration

**Choose Appropriate Formats:**
- Enable plain text if not using formatting
- Enable rich text if supporting styled text
- Enable table support if handling structured data
- Reduce scope for specialized documents

### State Management

**Proper Cleanup:**
- Pause syncing during Writing Tools processing
- Save state before processing starts
- Restore state after processing completes
- Handle interruptions gracefully

## Privacy and Security

### On-Device Processing

Writing Tools processes text entirely on-device:
- No text transmission to servers
- All processing local to device
- Complete user privacy
- No data retention

### Protected Content

Users can control which content is processed:
- Opt-out via `.none` configuration
- Protect sensitive ranges
- Exclude specific text types
- Disable for certain views

## Developer Resources

### Documentation

- Writing Tools implementation guide
- API reference documentation
- Code samples for UITextView and NSTextView
- macOS implementation examples

### Sample Code

- Basic implementation example
- Custom text view integration
- Delegate method implementation
- Protected range examples

### Related Sessions

WWDC 2024 sessions covering Writing Tools:
- "Get started with Writing Tools" (11:44)
- "Bring your machine learning and AI models to Apple silicon" (30:09)
- Integration with other Apple Intelligence features

## Limitations and Considerations

### Text Size

Writing Tools works efficiently with:
- Single paragraphs
- Multiple paragraphs
- Document-length text
- No strict size limits

**Performance Factors:**
- Longer texts may take more processing time
- Device capabilities affect processing speed
- Network state affects Private Cloud Compute features

### Language Support

Writing Tools availability depends on:
- Device language settings
- Regional availability
- Feature rollout phase
- Beta availability status

### Supported Applications

Works with:
- Built-in Apple apps (Mail, Messages, Notes, etc.)
- Third-party apps with standard text views
- Custom apps with proper implementation
- Web content in WKWebView

## Integration Roadmap

### Recommended Implementation Steps

**Phase 1: Assessment**
- Identify text areas in application
- Determine if standard text views can be used
- Assess customization requirements
- Plan implementation approach

**Phase 2: Implementation**
- Update to standard text views where possible
- Implement custom integration if needed
- Add delegate methods for lifecycle handling
- Test Writing Tools functionality

**Phase 3: Optimization**
- Configure format support appropriately
- Implement protected ranges if needed
- Handle edge cases and errors
- Optimize performance

**Phase 4: Testing and Deployment**
- Test across target devices
- Verify with various text types
- Test lifecycle handling
- Deploy with confidence

---

**Source:** Official Apple Writing Tools documentation and WWDC 2024 sessions
