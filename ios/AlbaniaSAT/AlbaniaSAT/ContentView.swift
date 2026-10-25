import SwiftUI

struct ContentView: View {
    // false = show the language screen, true = show the welcome screen
    @State private var showWelcome = false

    var body: some View {
        if showWelcome {
            WelcomeView()
        } else {
            LanguageView(showWelcome: $showWelcome)
        }
    }
}

#Preview {
    ContentView()
}
