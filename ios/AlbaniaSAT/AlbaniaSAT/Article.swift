import SwiftUI

// One news article from GET /articles (Regis)
struct Article: Codable, Identifiable {
    let id: Int
    let title: String
    let slug: String
    let summary: String?
    let imageUrl: String?
    let publishedAt: String
    let category: String?
    let author: String?

    // The date from "2026-11-14T15:26:43..." (only the day is used)
    var date: Date? {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.date(from: String(publishedAt.prefix(10)))
    }

    // Readable name of the category
    var categoryName: LocalizedStringKey? {
        switch category {
        case "urban_growth": return "Urban growth"
        case "forests": return "Forests"
        case "coast": return "Coast"
        case "agriculture": return "Agriculture"
        case "methodology": return "Methodology"
        default: return nil
        }
    }
}

// One page of articles: {"items": [...], "total": 21, "page": 1, "page_size": 9}
struct ArticlePage: Codable {
    let items: [Article]
    let total: Int
    let page: Int
    let pageSize: Int
}
