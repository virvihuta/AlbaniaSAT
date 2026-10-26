import SwiftUI

struct ContentView: View {
    // Die gewählte Sprache
    @AppStorage("language") private var language = "en"

    var body: some View {
        // NavigationStack: Screens werden "gestapelt", Zurück-Wischen geht automatisch
        NavigationStack {
            LanguageView()
        }
        .environment(\.locale, Locale(identifier: language))
    }
}

#Preview {
    ContentView()
}

