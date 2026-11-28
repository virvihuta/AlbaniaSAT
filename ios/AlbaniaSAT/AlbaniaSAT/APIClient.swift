import Foundation

// Gets data from our REST API (Regis)
enum APIClient {
    static let baseURL = "https://albaniasat-api.onrender.com"

    // Loads JSON from a path like "/articles?page=1" and turns it into Swift data
    static func get<T: Decodable>(_ path: String, as type: T.Type) async throws -> T {
        guard let url = URL(string: baseURL + path) else {
            throw URLError(.badURL)
        }

        let (data, response) = try await URLSession.shared.data(from: url)

        // Only 200 means everything is ok
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }

        let decoder = JSONDecoder()
        // "image_url" in the JSON becomes imageUrl in Swift
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(T.self, from: data)
    }
}
