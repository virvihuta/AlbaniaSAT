import SwiftUI

// News tab: the list of articles from the API (Regis)
struct NewsScreen: View {
    @State private var model = NewsViewModel()

    var body: some View {
        List {
            ForEach(model.articles) { article in
                ArticleRow(article: article)
            }

            // Button at the end of the list, only if there are more articles
            if model.hasMore {
                Button {
                    Task { await model.loadMore() }
                } label: {
                    if model.isLoading {
                        ProgressView()
                            .frame(maxWidth: .infinity)
                    } else {
                        Text("Load more")
                            .font(.montserrat(17))
                            .foregroundStyle(Color.brandRed)
                            .frame(maxWidth: .infinity)
                    }
                }
            }
        }
        .listStyle(.plain)
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
                            .font(.montserrat(17))
                            .padding(.horizontal, 24)
                            .padding(.vertical, 12)
                            .background(Color.brandRed)
                            .foregroundStyle(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
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
        // Pull the list down to load again
        .refreshable {
            await model.reload()
        }
    }
}

// One row in the list: category, title, summary, date
struct ArticleRow: View {
    let article: Article

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let categoryName = article.categoryName {
                Text(categoryName)
                    .font(.montserratBold(12))
                    .foregroundStyle(Color.brandRed)
                    .textCase(.uppercase)
            }
            Text(article.title)
                .font(.montserratBold(18))
            if let summary = article.summary {
                Text(summary)
                    .font(.montserrat(15))
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
            }
            if let date = article.date {
                Text(date, format: .dateTime.day().month(.wide).year())
                    .font(.montserrat(13))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 8)
    }
}
