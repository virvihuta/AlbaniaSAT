import SwiftUI

struct WelcomeView: View {
    // Which page is showing right now (0, 1 or 2)
    @State private var page = 0

    // The 3 pages: image, headline, text
    let pages = [
        ("welcome1", "See Albania from space",
         "Sentinel-2 imagery of the whole country, from the Alps to Butrint."),
        ("welcome2", "Seven classes on one map",
         "Forest, grassland, fields, towns and water, sorted by a model trained on Albanian terrain."),
        ("welcome3", "Compare the years",
         "See how the land changes, county by county, from 2018 to today.")
    ]

    var body: some View {
        VStack(spacing: 16) {

            // "Skip" at the top right (not on the last page)
            HStack {
                Spacer()
                if page < 2 {
                    Button("Skip") {
                        withAnimation { page = 2 }
                    }
                    .font(.montserrat(17))
                    .foregroundStyle(Color.brandRed)
                }
            }
            .frame(height: 30)

            // The 3 pages you can swipe
            TabView(selection: $page) {
                ForEach(0..<3, id: \.self) { i in
                    VStack(alignment: .leading, spacing: 16) {
                        Image(pages[i].0)
                            .resizable()
                            .scaledToFit()
                            .clipShape(RoundedRectangle(cornerRadius: 24))
                        Text(pages[i].1)
                            .font(.montserratBold(28))
                        Text(pages[i].2)
                            .font(.montserrat(17))
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                    .tag(i)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .never))

            // The 3 dots at the bottom
            HStack(spacing: 12) {
                ForEach(0..<3, id: \.self) { i in
                    Capsule()
                        .fill(i == page ? Color.primary : Color.gray.opacity(0.4))
                        .frame(width: i == page ? 20 : 8, height: 8)
                }
            }

            // Button: "Next" or on the last page "Open the map"
            Button {
                if page < 2 {
                    withAnimation { page += 1 }
                } else {
                    print("Open the map")
                }
            } label: {
                Text(page < 2 ? "Next" : "Open the map")
                    .font(.montserrat(20))
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.brandRed)
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
            }
        }
        .padding(.horizontal, 20)
    }
}

#Preview {
    WelcomeView()
}
