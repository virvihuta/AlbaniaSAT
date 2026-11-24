import SwiftUI

// Map tab, will show the classification map with MapKit
struct MapScreen: View {
    var body: some View {
        Text("The map is coming soon.")
            .font(.montserrat(17))
            .foregroundStyle(.secondary)
            .navigationTitle("Map")
    }
}

// Explore tab, will show the counties and their land cover
struct ExploreScreen: View {
    var body: some View {
        Text("The counties are coming soon.")
            .font(.montserrat(17))
            .foregroundStyle(.secondary)
            .navigationTitle("Explore")
    }
}

// News tab, will show the articles from the API
struct NewsScreen: View {
    var body: some View {
        Text("The news are coming soon.")
            .font(.montserrat(17))
            .foregroundStyle(.secondary)
            .navigationTitle("News")
    }
}
