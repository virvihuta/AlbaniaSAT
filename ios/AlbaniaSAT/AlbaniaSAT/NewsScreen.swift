import SwiftUI

// News tab: the articles from the API as cards (Regis)
struct NewsScreen: View {
    @State private var model = NewsViewModel()

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
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
                            } else {
                                Image(systemName: "arrow.down.circle.fill")
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
                    .padding(.top, 8)
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
                    Text("The news didn't load")
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
        .navigationTitle("News")
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

// One article as a card
struct ArticleCard: View {
    let article: Article

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {

            // Top: category badge and date
            HStack {
                if let categoryName = article.categoryName {
                    Label(categoryName, systemImage: article.categoryIcon)
                        .font(.montserratBold(12))
                        .foregroundStyle(Color.brandRed)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Color.brandRed.opacity(0.1))
                        .clipShape(Capsule())
                }
                Spacer()
                if let date = article.date {
                    Text(date, format: .dateTime.day().month(.abbreviated).year())
                        .font(.montserrat(13))
                        .foregroundStyle(.secondary)
                }
            }

            // Title and summary
            Text(article.title)
                .font(.montserratBold(19))
            if let summary = article.summary {
                Text(summary)
                    .font(.montserrat(15))
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
            }

            // Bottom: author and "Read"
            HStack {
                if let author = article.author {
                    Label(author, systemImage: "person.circle.fill")
                        .font(.montserrat(13))
                        .foregroundStyle(.secondary)
                }
                Spacer()
                HStack(spacing: 4) {
                    Text("Read")
                    Image(systemName: "chevron.right")
                }
                .font(.montserratBold(14))
                .foregroundStyle(Color.brandRed)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 20))
        .shadow(color: .black.opacity(0.06), radius: 8, y: 4)
    }
}
