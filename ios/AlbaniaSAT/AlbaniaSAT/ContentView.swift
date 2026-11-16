import SwiftUI

struct ContentView: View {
    // The chosen language
    @AppStorage("language") private var language = "en"

    // Light or dark mode chosen with the AppearanceButton ("system" = like the iPhone)
    @AppStorage("appearance") private var appearance = "system"

    private var chosenScheme: ColorScheme? {
        switch appearance {
        case "light": return .light
        case "dark": return .dark
        default: return nil
        }
    }

    var body: some View {
        // NavigationStack: screens are "stacked", swiping back works automatically
        NavigationStack {
            LanguageView()
        }
        .environment(\.locale, Locale(identifier: language))
        .preferredColorScheme(chosenScheme)
    }
}

#Preview {
    ContentView()
}
