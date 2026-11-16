import SwiftUI

// Button that switches between light mode and dark mode (Regis)
struct AppearanceButton: View {
    // Saved on the iPhone, so the app remembers it after a restart.
    // "system" = like the iPhone, "light" or "dark" = chosen by the user
    @AppStorage("appearance") private var appearance = "system"

    // The mode the app shows right now
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        Button {
            // Switch to the opposite of what is shown right now
            appearance = colorScheme == .dark ? "light" : "dark"
        } label: {
            // In dark mode show a sun (to switch to light), in light mode a moon
            Image(systemName: colorScheme == .dark ? "sun.max.fill" : "moon.fill")
        }
        .accessibilityLabel(Text("Change appearance"))
    }
}
