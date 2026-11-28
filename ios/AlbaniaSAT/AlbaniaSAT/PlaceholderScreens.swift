import SwiftUI

// Map tab, will show the classification map with MapKit (Regis)
struct MapScreen: View {
    var body: some View {
        Text("The map is coming soon.")
            .font(.montserrat(17))
            .foregroundStyle(.secondary)
            .navigationTitle("Map")
    }
}

// Explore tab, will show the counties and their land cover (Regis)
struct ExploreScreen: View {
    var body: some View {
        Text("The counties are coming soon.")
            .font(.montserrat(17))
            .foregroundStyle(.secondary)
            .navigationTitle("Explore")
    }
}
