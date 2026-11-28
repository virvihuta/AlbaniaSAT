import SwiftUI

// The 7 land cover colors, the same as on the website and in the thesis
extension Color {
    static let lcBroadleaf = Color(red: 76 / 255, green: 154 / 255, blue: 60 / 255)    // #4C9A3C
    static let lcConiferous = Color(red: 23 / 255, green: 100 / 255, blue: 59 / 255)   // #17643B
    static let lcCropland = Color(red: 240 / 255, green: 169 / 255, blue: 59 / 255)    // #F0A93B
    static let lcGrassShrub = Color(red: 203 / 255, green: 208 / 255, blue: 90 / 255)  // #CBD05A
    static let lcOlive = Color(red: 143 / 255, green: 158 / 255, blue: 107 / 255)      // #8F9E6B
    static let lcBuiltUp = Color(red: 140 / 255, green: 94 / 255, blue: 196 / 255)     // #8C5EC4
    static let lcWater = Color(red: 43 / 255, green: 127 / 255, blue: 212 / 255)       // #2B7FD4
}

// The thin Montserrat for big titles, like on the website
extension Font {
    static func montserratLight(_ size: CGFloat) -> Font {
        .custom("Montserrat-Light", size: size)
    }
}
