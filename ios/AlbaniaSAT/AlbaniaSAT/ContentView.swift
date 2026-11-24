import SwiftUI

struct ContentView: View {
    // The chosen language
    @AppStorage("language") private var language = "en"

    // true = dark mode, false = light mode
    @AppStorage("darkMode") private var darkMode = false

    // true = the user has already seen the welcome pages
    @AppStorage("onboardingDone") private var onboardingDone = false

    var body: some View {
        Group {
            if onboardingDone {
                MainTabView()
            } else {
                NavigationStack {
                    LanguageView()
                }
            }
        }
        .environment(\.locale, Locale(identifier: language))
        .preferredColorScheme(darkMode ? .dark : .light)
    }
}

#Preview {
    ContentView()
}
