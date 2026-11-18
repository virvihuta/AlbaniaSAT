// All fixed texts of the website in English (en), Albanian (sq) and German (de).
// {name} in a text is replaced by a value, for example {f1} by "81.38%".
export const translations = {
  en: {
    'nav.overview': 'Overview',
    'nav.explorer': 'Explorer',
    'nav.story': 'Story',
    'nav.contact': 'Contact',
    'language.label': 'Language',

    'tagline': "Europe couldn't see Albania. Now it can.",

    'home.eyebrow': 'Land cover of Albania · Sentinel-2 · 2021, more years coming',
    'home.intro':
      'AlbaniaSAT maps land cover across the whole country from Sentinel-2 satellite imagery, with a model trained on Albanian ground truth instead of borrowed European data.',
    'home.readStories': 'Read the stories →',
    'home.mapTitle': 'Explore the map',
    'home.mapText': 'Regions, classes and Sentinel-2 patches',
    'home.mapButton': 'Open the explorer →',
    'home.classes': 'Classes',
    'home.statAccuracy': 'Test accuracy of the AlbaniaSAT model on held-out Albanian patches (macro F1 {f1})',
    'home.statEuroSat': 'Accuracy of a EuroSAT-trained model on the same Albanian data',
    'home.statClasses': 'Land cover classes, from forest to built-up land',
    'home.statPatches': 'Labelled 64 × 64 px Sentinel-2 patches',

    'footer.about':
      'Land cover maps of Albania at 10 m resolution, built from Copernicus Sentinel-2 imagery with a model trained on Albanian ground truth.',
    'footer.platform': 'Platform',
    'footer.sources': 'Data sources',
    'footer.rights': '© 2026 AlbaniaSAT. All rights reserved.',
    'footer.copernicus': 'Contains modified Copernicus Sentinel data 2021',
  },

  sq: {
    'nav.overview': 'Përmbledhje',
    'nav.explorer': 'Eksploro',
    'nav.story': 'Histori',
    'nav.contact': 'Kontakt',
    'language.label': 'Gjuha',

    'tagline': 'Evropa nuk e shihte Shqipërinë. Tani e sheh.',

    'home.eyebrow': 'Mbulesa e tokës në Shqipëri · Sentinel-2 · 2021, vite të tjera së shpejti',
    'home.intro':
      'AlbaniaSAT harton mbulesën e tokës në të gjithë vendin nga imazhet satelitore Sentinel-2, me një model të trajnuar mbi të dhëna reale shqiptare dhe jo mbi të dhëna të huazuara evropiane.',
    'home.readStories': 'Lexo historitë →',
    'home.mapTitle': 'Eksploro hartën',
    'home.mapText': 'Rajone, klasa dhe imazhe Sentinel-2',
    'home.mapButton': 'Hap eksploruesin →',
    'home.classes': 'Klasat',
    'home.statAccuracy': 'Saktësia e modelit AlbaniaSAT në imazhet shqiptare të testimit (macro F1 {f1})',
    'home.statEuroSat': 'Saktësia e një modeli të trajnuar me EuroSAT në të njëjtat të dhëna shqiptare',
    'home.statClasses': 'Klasa të mbulesës së tokës, nga pylli te zonat e ndërtuara',
    'home.statPatches': 'Imazhe Sentinel-2 të etiketuara, 64 × 64 px',

    'footer.about':
      'Harta të mbulesës së tokës në Shqipëri me rezolucion 10 m, të ndërtuara nga imazhet Copernicus Sentinel-2 me një model të trajnuar mbi të dhëna reale shqiptare.',
    'footer.platform': 'Platforma',
    'footer.sources': 'Burimet e të dhënave',
    'footer.rights': '© 2026 AlbaniaSAT. Të gjitha të drejtat e rezervuara.',
    'footer.copernicus': 'Përmban të dhëna të modifikuara Copernicus Sentinel 2021',
  },

  de: {
    'nav.overview': 'Übersicht',
    'nav.explorer': 'Explorer',
    'nav.story': 'Berichte',
    'nav.contact': 'Kontakt',
    'language.label': 'Sprache',

    'tagline': 'Europa konnte Albanien nicht sehen. Jetzt schon.',

    'home.eyebrow': 'Landbedeckung Albaniens · Sentinel-2 · 2021, weitere Jahre folgen',
    'home.intro':
      'AlbaniaSAT kartiert die Landbedeckung des ganzen Landes aus Sentinel-2-Satellitenbildern, mit einem Modell, das auf albanischen Referenzdaten trainiert wurde statt auf übernommenen europäischen Daten.',
    'home.readStories': 'Berichte lesen →',
    'home.mapTitle': 'Karte erkunden',
    'home.mapText': 'Regionen, Klassen und Sentinel-2-Ausschnitte',
    'home.mapButton': 'Explorer öffnen →',
    'home.classes': 'Klassen',
    'home.statAccuracy': 'Testgenauigkeit des AlbaniaSAT-Modells auf albanischen Testausschnitten (Makro-F1 {f1})',
    'home.statEuroSat': 'Genauigkeit eines mit EuroSAT trainierten Modells auf denselben albanischen Daten',
    'home.statClasses': 'Landbedeckungsklassen, von Wald bis zur bebauten Fläche',
    'home.statPatches': 'Gelabelte Sentinel-2-Ausschnitte mit 64 × 64 px',

    'footer.about':
      'Landbedeckungskarten Albaniens mit 10 m Auflösung, erstellt aus Copernicus-Sentinel-2-Bildern mit einem Modell, das auf albanischen Referenzdaten trainiert wurde.',
    'footer.platform': 'Plattform',
    'footer.sources': 'Datenquellen',
    'footer.rights': '© 2026 AlbaniaSAT. Alle Rechte vorbehalten.',
    'footer.copernicus': 'Enthält veränderte Copernicus-Sentinel-Daten 2021',
  },
}