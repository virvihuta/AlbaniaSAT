import SwiftUI

// Stories tab: the articles from the API as cards, like on the website
struct NewsScreen: View {
    @State private var model = NewsViewModel()

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                ForEach(model.articles) { article in
                    ArticleCard(article: article)
                }

                // Button at the end, only if there are more articles
                if model.hasMore {
                    Button {
                        Task { await model.loadMore() }
                    } label: {
                        HStack(spacing: 8) {
                            if model.isLoading {
                                ProgressView()
                            }
                            Text("Load more")
                                .font(.montserratBold(16))
                        }
                        .foregroundStyle(Color.brandRed)
                        .padding(.horizontal, 24)
                        .padding(.vertical, 12)
                        .background(Color.brandRed.opacity(0.1))
                        .clipShape(Capsule())
                    }
                }
            }
            .padding(16)
        }
        .background(Color(.systemGroupedBackground))
        .overlay {
            // First loading: a circle in the middle
            if model.articles.isEmpty && model.isLoading {
                ProgressView()
            }
            // No internet or server error: message and "Try again"
            if model.articles.isEmpty && model.failed {
                VStack(spacing: 16) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.largeTitle)
                    Text("The stories didn't load")
                        .font(.montserratBold(20))
                    Text("Check your connection and try again.")
                        .font(.montserrat(15))
                        .foregroundStyle(.secondary)
                    Button {
                        Task { await model.reload() }
                    } label: {
                        Text("Try again")
                            .font(.montserratBold(16))
                            .padding(.horizontal, 24)
                            .padding(.vertical, 12)
                            .background(Color.brandRed)
                            .foregroundStyle(.white)
                            .clipShape(Capsule())
                    }
                }
                .padding()
            }
        }
        .navigationTitle("Stories")
        // Load the first page when the tab opens
        .task {
            if model.articles.isEmpty {
                await model.loadMore()
            }
        }
        // Pull down to load again
        .refreshable {
            await model.reload()
        }
    }
}

// One article as a card: color stripes on top, then the text
struct ArticleCard: View {
    let article: Article

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {

            ArticleCover()
                .frame(height: 160)

            VStack(alignment: .leading, spacing: 14) {
                // Category: grey, capital letters, with space between the letters
                if let categoryName = article.categoryName {
                    Text(categoryName)
                        .font(.montserrat(12))
                        .tracking(2)
                        .textCase(.uppercase)
                        .foregroundStyle(.secondary)
                }

                // Big thin title
                Text(article.title)
                    .font(.montserratLight(28))

                if let summary = article.summary {
                    Text(summary)
                        .font(.montserrat(15))
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                }

                // Author: circle with initials, name and date
                HStack(spacing: 12) {
                    Text(article.initials)
                        .font(.montserratBold(13))
                        .frame(width: 40, height: 40)
                        .background(Color(.systemGroupedBackground))
                        .clipShape(Circle())
                    VStack(alignment: .leading, spacing: 2) {
                        Text(article.authorName)
                            .font(.montserratBold(14))
                        if let date = article.date {
                            Text(date, format: .dateTime.day().month(.abbreviated).year())
                                .font(.montserrat(13))
                                .foregroundStyle(.secondary)
                        }
                    }
                    Spacer()
                    Image(systemName: "chevron.right")
                        .foregroundStyle(Color.brandRed)
                }
                .padding(.top, 4)
            }
            .padding(20)
        }
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 20))
        .shadow(color: .black.opacity(0.06), radius: 8, y: 4)
    }
}

// Diagonal stripes in the land cover colors, the same as the website when there is no image
struct ArticleCover: View {
    var body: some View {
        LinearGradient(
            stops: [
                .init(color: .lcCropland, location: 0),
                .init(color: .lcCropland, location: 0.35),
                .init(color: .lcBuiltUp, location: 0.35),
                .init(color: .lcBuiltUp, location: 0.45),
                .init(color: .lcGrassShrub, location: 0.45),
                .init(color: .lcGrassShrub, location: 0.60),
                .init(color: .lcConiferous, location: 0.60),
                .init(color: .lcConiferous, location: 1)
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }
}
