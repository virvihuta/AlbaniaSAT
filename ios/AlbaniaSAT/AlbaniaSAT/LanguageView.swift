import SwiftUI

struct LanguageView: View {
    // Die gewählte Sprache wird auf dem iPhone gespeichert
    @AppStorage("language") private var language = "en"
    
    @Binding var showWelcome: Bool

    // Die drei Sprachen: Code und Name
    let languages = [("sq", "Shqip"), ("en", "English"), ("de", "Deutsch")]

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            
            
            // Logo und Name nebeneinander
            HStack(spacing: 16) {
                // Logo: rotes Rechteck hinten, weiße Linien vorne
                ZStack {
                    Rectangle()
                        .fill(Color.brandRed)
                    Image("contours")
                        .resizable()
                        .scaledToFit()
                }
                .frame(width: 54, height: 72)
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

            // Überschrift
            Text("Choose your language")
                .font(.montserratBold(28))
            Text("You can change this later in Settings.")
                .font(.montserrat(15))
                .foregroundStyle(.secondary)

            // Eine Zeile pro Sprache
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

            // Weiter-Button
            Button {
                showWelcome = true
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
    }
}

// Unsere rote Markenfarbe aus dem Mockup
extension Color {
    static let brandRed = Color(red: 0.75, green: 0.12, blue: 0.18)
}

// Unsere Schrift Montserrat
extension Font {
    static func montserrat(_ size: CGFloat) -> Font {
        .custom("Montserrat-Regular", size: size)
    }
    static func montserratBold(_ size: CGFloat) -> Font {
        .custom("Montserrat-Bold", size: size)
    }
}

#Preview {
    LanguageView(showWelcome: .constant(false))
}
