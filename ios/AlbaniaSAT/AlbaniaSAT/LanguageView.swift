import SwiftUI

struct LanguageView: View {
    // The chosen language is saved on the iPhone
    @AppStorage("language") private var language = "en"

    // The three languages: code and name
    let languages = [("sq", "Shqip"), ("en", "English"), ("de", "Deutsch")]

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {

            // Logo and name next to each other
            HStack(spacing: 16) {
                // Logo: red rectangle, the white lines are cut out (transparent)
                Rectangle()
                    .fill(Color.brandRed)
                    .frame(width: 54, height: 72)
                    .mask {
                        ZStack {
                            Rectangle()
                            Image("contours")
                                .resizable()
                                .scaledToFit()
                                .blendMode(.destinationOut)
                        }
                        .compositingGroup()
                    }
                VStack(alignment: .leading) {
                    Text("AlbaniaSAT")
                        .font(.montserratBold(34))
                    Text("Europe couldn't see Albania. Now it can.")
                        .font(.montserrat(15))
                        .foregroundStyle(Color.brandRed)
                }
            }
            .padding(.top, 40)
            .padding(.bottom, 40)

            // Title
            Text("Choose your language")
                .font(.montserratBold(28))
            Text("You can change this later in Settings.")
                .font(.montserrat(15))
                .foregroundStyle(.secondary)

            // One row per language
            ForEach(languages, id: \.0) { code, name in
                let selected = (language == code)

                Button {
                    language = code
                } label: {
                    HStack {
                        Text(name).font(.montserratBold(20))
                        Spacer()
                        if selected {
                            Image(systemName: "checkmark")
                        }
                    }
                    .padding(20)
                    .foregroundStyle(selected ? Color.brandRed : Color.primary)
                    .background(
                        RoundedRectangle(cornerRadius: 16)
                            .fill(selected ? Color.brandRed.opacity(0.1) : Color.clear)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(selected ? Color.brandRed : Color.gray.opacity(0.3))
                    )
                }
            }

            Spacer()

            // Continue: opens the welcome pages
            NavigationLink {
                WelcomeView()
            } label: {
                Text("Continue")
                    .font(.montserrat(20))
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.brandRed)
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
            }
        }
        .padding(.horizontal, 20)
        // Light/dark mode button at the top right
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                AppearanceButton()
            }
        }
    }
}

// Our red brand color from the mockup
extension Color {
    static let brandRed = Color(red: 0.75, green: 0.12, blue: 0.18)
}

// Our font Montserrat
extension Font {
    static func montserrat(_ size: CGFloat) -> Font {
        .custom("Montserrat-Regular", size: size)
    }
    static func montserratBold(_ size: CGFloat) -> Font {
        .custom("Montserrat-Bold", size: size)
    }
}

#Preview {
    NavigationStack {
        LanguageView()
    }
}
