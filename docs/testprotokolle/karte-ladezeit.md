# Testprotokoll: Ladezeit der interaktiven Karte (Ziel M-3.3)

**Datum:** 16.11.2026
**Testerin:** Suisa Shehi
**Anforderung:** Klick auf eine Region lädt Klassifikationsdaten und Sentinel-2-Bilder aus der API in unter drei Sekunden.

## Ablauf

Auf der Live-Seite https://albaniasat.onrender.com den Explorer öffnen, eine Region in der Liste oder auf der Karte wählen und die angezeigte Ladezeit ablesen. Die Zeit wird im Browser gemessen, vom Start der Anfrage bis die Patches und die Zusammenfassung angekommen sind. Danach dieselbe Region noch einmal öffnen, um den Cache zu prüfen.

## Ergebnisse

| Region | Patches | Ladezeit | Ergebnis |
|---|---|---|---|
| Shkodër | 1049 | 0,5 s | bestanden |
| Kukës | 384 | 0,1 s | bestanden |
| Lezhë | 449 | 0,2 s | bestanden |
| Tiranë | 617 | 0,2 s | bestanden |

**Cache:** Beim zweiten Öffnen einer Region erscheinen die Patches sofort, ohne neue Anfrage an die API („From cache“).

**Hinweis:** Das Backend läuft im Gratis-Tarif von Render. Nach 15 Minuten ohne Anfragen schläft der Server, und die erste Anfrage danach dauert 30 bis 60 Sekunden. Die Messungen wurden mit laufendem Server gemacht.

## Ergebnis

Alle gemessenen Regionen laden in unter einer Sekunde. Die Anforderung von unter drei Sekunden ist erfüllt.

## Screenshots

![Shkodër](bilder/karte-shkoder.png)
![Kukës](bilder/karte-kukes.png)
![Lezhë](bilder/karte-lezhe.png)
![Tiranë](bilder/karte-tirane.png)