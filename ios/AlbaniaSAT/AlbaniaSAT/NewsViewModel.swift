import Foundation
import Observation

// Loads the articles for the News tab, page by page (Regis)
@Observable
final class NewsViewModel {
    var articles: [Article] = []
    var total = 0
    var isLoading = false
    var failed = false

    private var page = 0

    // Are there more articles on the server?
    var hasMore: Bool {
        articles.count < total
    }

    // Loads the next page and adds it to the list
    func loadMore() async {
        if isLoading { return }
        isLoading = true
        failed = false
        do {
            let result = try await APIClient.get("/articles?page=\(page + 1)", as: ArticlePage.self)
            articles += result.items
            total = result.total
            page += 1
        } catch {
            failed = true
        }
        isLoading = false
    }

    // Starts again from page 1 (pull to refresh)
    func reload() async {
        articles = []
        total = 0
        page = 0
        await loadMore()
    }
}
