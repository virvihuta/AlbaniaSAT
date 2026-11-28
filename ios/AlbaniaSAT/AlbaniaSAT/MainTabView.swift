import SwiftUI

// The main screen with 4 tabs at the bottom
struct MainTabView: View {
    var body: some View {
        TabView {
            NavigationStack {
                MapScreen()
            }
            .tabItem {
                Label("Map", systemImage: "map")
            }

            NavigationStack {
                ExploreScreen()
            }
            .tabItem {
                Label("Explore", systemImage: "list.bullet")
            }

            NavigationStack {
                NewsScreen()
            }
            .tabItem {
                Label("Stories", systemImage: "newspaper")
            }

            NavigationStack {
                SettingsView()
            }
            .tabItem {
                Label("Settings", systemImage: "gearshape")
            }
        }
        .tint(Color.brandRed)
    }
}

#Preview {
    MainTabView()
}
