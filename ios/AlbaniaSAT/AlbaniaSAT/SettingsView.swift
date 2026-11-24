import SwiftUI

// Settings tab: language, appearance and the welcome pages
struct SettingsView: View {
    // The same saved values as in ContentView
    @AppStorage("language") private var language = "en"
    @AppStorage("darkMode") private var darkMode = false
    @AppStorage("onboardingDone") private var onboardingDone = false

    let languages = [("sq", "Shqip"), ("en", "English"), ("de", "Deutsch")]

    var body: some View {
        Form {
            Section {
                Picker("Language", selection: $language) {
                    ForEach(languages, id: \.0) { code, name in
                        Text(name).tag(code)
                    }
                }
            }

            Section("Appearance") {
                Picker("Appearance", selection: $darkMode) {
                    Text("Light").tag(false)
                    Text("Dark").tag(true)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }

            Section {
                Button("Show the welcome pages again") {
                    onboardingDone = false
                }
            }
        }
        .navigationTitle("Settings")
    }
}
