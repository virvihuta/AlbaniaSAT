-- =====================================================================
-- AlbaniaSAT - Datenbankschema
-- PostgreSQL 18 + PostGIS 3.6
-- Arbeitspaket 1.1 / 1.2 - Regis Balla
-- Stand: September 2026
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS postgis;


-- =====================================================================
-- 1. Aufzaehlungstypen
-- =====================================================================

CREATE TYPE user_role    AS ENUM ('admin', 'redakteur', 'besucher');
CREATE TYPE article_status AS ENUM ('draft', 'published');
CREATE TYPE message_status AS ENUM ('new', 'read', 'done');
CREATE TYPE split_type   AS ENUM ('train', 'val', 'test');
CREATE TYPE season_type  AS ENUM ('spring', 'summer', 'winter');


-- =====================================================================
-- 2. Stammdaten
-- =====================================================================

-- Beschreibt einen Datensatz-Stand aus Virvis Pipeline.
CREATE TABLE dataset_versions (
    id              SERIAL       PRIMARY KEY,
    version         VARCHAR(20)  NOT NULL UNIQUE,
    year            SMALLINT     NOT NULL,
    patch_count     INTEGER,
    band_count      SMALLINT     NOT NULL DEFAULT 18,
    patch_size_px   SMALLINT     NOT NULL DEFAULT 64,
    resolution_m    SMALLINT     NOT NULL DEFAULT 10,
    extent_m        SMALLINT     NOT NULL DEFAULT 640,
    label_source    TEXT,
    array_path      TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT dataset_versions_year_ck CHECK (year BETWEEN 2015 AND 2100)
);

COMMENT ON TABLE  dataset_versions IS 'Ein Stand des albanischen Sentinel-2-Datensatzes';
COMMENT ON COLUMN dataset_versions.extent_m IS 'Kantenlaenge eines Patches am Boden in Metern';


-- Die drei saisonalen Komposite, aus denen der 18-Band-Tensor besteht.
CREATE TABLE composites (
    id                  SERIAL        PRIMARY KEY,
    dataset_version_id  INTEGER       NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
    season              season_type   NOT NULL,
    period_start        DATE          NOT NULL,
    period_end          DATE          NOT NULL,
    cloud_threshold     NUMERIC(4,3),
    band_offset         SMALLINT      NOT NULL,
    masking_notes       TEXT,
    CONSTRAINT composites_unique      UNIQUE (dataset_version_id, season),
    CONSTRAINT composites_period_ck   CHECK (period_end >= period_start)
);

COMMENT ON COLUMN composites.band_offset IS 'Startposition dieser Saison im 18-Band-Tensor: summer 0, spring 6, winter 12';


-- Die sieben Landbedeckungsklassen.
CREATE TABLE land_cover_classes (
    id           SMALLINT     PRIMARY KEY,
    name_en      VARCHAR(50)  NOT NULL,
    name_de      VARCHAR(50),
    name_sq      VARCHAR(50),
    color_hex    CHAR(7)      NOT NULL,
    description  TEXT,
    CONSTRAINT land_cover_classes_color_ck CHECK (color_hex ~ '^#[0-9A-Fa-f]{6}$')
);

COMMENT ON TABLE land_cover_classes IS 'Zentrale Farbdefinition fuer Karte, Diagramme und iOS-App';


-- Albanische Verwaltungseinheiten. parent_id bildet Qark -> Bashki ab.
CREATE TABLE regions (
    id         SERIAL       PRIMARY KEY,
    name       VARCHAR(100) NOT NULL,
    code       VARCHAR(10)  UNIQUE,
    level      VARCHAR(20)  NOT NULL,
    parent_id  INTEGER      REFERENCES regions(id) ON DELETE SET NULL,
    geom       GEOMETRY(MULTIPOLYGON, 4326) NOT NULL,
    area_km2   NUMERIC(10,2),
    CONSTRAINT regions_level_ck CHECK (level IN ('qark', 'bashki'))
);


-- =====================================================================
-- 3. Kerndaten
-- =====================================================================

-- Ein Bildausschnitt. array_index verweist auf die Zeile in patches.npy.
CREATE TABLE patches (
    id                  BIGSERIAL  PRIMARY KEY,
    dataset_version_id  INTEGER    NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
    array_index         INTEGER    NOT NULL,
    center_geom         GEOMETRY(POINT, 4326)   NOT NULL,
    extent_geom         GEOMETRY(POLYGON, 4326),
    region_id           INTEGER    REFERENCES regions(id) ON DELETE SET NULL,
    year                SMALLINT   NOT NULL,
    reference_class_id  SMALLINT   NOT NULL REFERENCES land_cover_classes(id),
    split               split_type,
    preview_path        TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT patches_array_index_ck CHECK (array_index >= 0),
    CONSTRAINT patches_dataset_index_unique UNIQUE (dataset_version_id, array_index)
);

COMMENT ON COLUMN patches.reference_class_id IS 'Ground Truth aus WorldCover und CORINE, NICHT die Modellvorhersage';
COMMENT ON COLUMN patches.region_id IS 'Wird beim Ingest per raeumlichem Join gesetzt';


-- Jeder trainierte Modellstand.
CREATE TABLE model_versions (
    id                  SERIAL       PRIMARY KEY,
    label               VARCHAR(80)  NOT NULL UNIQUE,
    architecture        VARCHAR(80),
    input_bands         SMALLINT,
    accuracy            NUMERIC(6,5),
    f1_macro            NUMERIC(6,5),
    dataset_version_id  INTEGER      REFERENCES dataset_versions(id) ON DELETE SET NULL,
    trained_at          DATE,
    notes               TEXT,
    CONSTRAINT model_versions_accuracy_ck CHECK (accuracy BETWEEN 0 AND 1),
    CONSTRAINT model_versions_f1_ck       CHECK (f1_macro BETWEEN 0 AND 1)
);


-- Die Aussage eines Modells ueber einen Patch.
CREATE TABLE predictions (
    id                  BIGSERIAL    PRIMARY KEY,
    patch_id            BIGINT       NOT NULL REFERENCES patches(id) ON DELETE CASCADE,
    model_version_id    INTEGER      NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    predicted_class_id  SMALLINT     NOT NULL REFERENCES land_cover_classes(id),
    confidence          NUMERIC(6,5) NOT NULL,
    probabilities       JSONB,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT predictions_unique     UNIQUE (patch_id, model_version_id),
    CONSTRAINT predictions_conf_ck    CHECK (confidence BETWEEN 0 AND 1)
);

COMMENT ON COLUMN predictions.probabilities IS 'Softmax ueber alle sieben Klassen';


-- Vorberechnete Auswertung je Region, Jahr und Modellversion.
CREATE TABLE region_year_summaries (
    id                  SERIAL       PRIMARY KEY,
    region_id           INTEGER      NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
    year                SMALLINT     NOT NULL,
    model_version_id    INTEGER      NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    class_distribution  JSONB        NOT NULL,
    dominant_class_id   SMALLINT     REFERENCES land_cover_classes(id),
    patch_count         INTEGER,
    change_vs_previous  JSONB,
    ai_summary          TEXT,
    generated_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT region_year_summaries_unique UNIQUE (region_id, year, model_version_id)
);

COMMENT ON TABLE region_year_summaries IS
    'Bewusst redundant: Aggregation beim Ingest statt bei jeder Anfrage, damit die API unter zwei Sekunden bleibt';


-- =====================================================================
-- 4. Auth- und Redaktionsmodul (Suisa)
-- =====================================================================

CREATE TABLE users (
    id             SERIAL       PRIMARY KEY,
    email          VARCHAR(255) NOT NULL UNIQUE,
    password_hash  VARCHAR(60)  NOT NULL,
    role           user_role    NOT NULL DEFAULT 'besucher',
    display_name   VARCHAR(100),
    is_active      BOOLEAN      NOT NULL DEFAULT true,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    last_login_at  TIMESTAMPTZ,
    CONSTRAINT users_email_lower_ck CHECK (email = lower(email))
);


CREATE TABLE articles (
    id            SERIAL         PRIMARY KEY,
    title         VARCHAR(200)   NOT NULL,
    slug          VARCHAR(220)   NOT NULL UNIQUE,
    summary       VARCHAR(300),
    content       TEXT           NOT NULL,
    image_url     TEXT,
    author_id     INTEGER        REFERENCES users(id) ON DELETE SET NULL,
    status        article_status NOT NULL DEFAULT 'draft',
    published_at  TIMESTAMPTZ,
    created_at    TIMESTAMPTZ    NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ    NOT NULL DEFAULT now(),
    CONSTRAINT articles_published_ck
        CHECK (status = 'draft' OR published_at IS NOT NULL)
);


CREATE TABLE contact_messages (
    id          SERIAL         PRIMARY KEY,
    name        VARCHAR(100)   NOT NULL,
    email       VARCHAR(255)   NOT NULL,
    subject     VARCHAR(200),
    message     TEXT           NOT NULL,
    status      message_status NOT NULL DEFAULT 'new',
    created_at  TIMESTAMPTZ    NOT NULL DEFAULT now()
);


-- =====================================================================
-- 5. Indizes
-- =====================================================================

CREATE INDEX idx_patches_region_year   ON patches (region_id, year);
CREATE INDEX idx_patches_center_geom   ON patches USING GIST (center_geom);
CREATE INDEX idx_patches_extent_geom   ON patches USING GIST (extent_geom);
CREATE INDEX idx_patches_class         ON patches (reference_class_id);

CREATE INDEX idx_regions_geom          ON regions USING GIST (geom);
CREATE INDEX idx_regions_parent        ON regions (parent_id);

CREATE INDEX idx_predictions_patch     ON predictions (patch_id);
CREATE INDEX idx_predictions_model     ON predictions (model_version_id);

CREATE INDEX idx_summaries_region_year ON region_year_summaries (region_id, year);

CREATE INDEX idx_articles_published    ON articles (status, published_at DESC);
CREATE INDEX idx_contact_status        ON contact_messages (status, created_at DESC);


-- =====================================================================
-- 6. Trigger fuer updated_at
-- =====================================================================

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER users_set_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER articles_set_updated_at
    BEFORE UPDATE ON articles
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- =====================================================================
-- 7. Stammdaten befuellen
-- =====================================================================

INSERT INTO land_cover_classes (id, name_en, name_de, name_sq, color_hex) VALUES
    (0, 'Broad-leaved Forest',    'Laubwald',              'Pyje gjethegjere', '#2E7D32'),
    (1, 'Coniferous Forest',      'Nadelwald',             'Pyje halore',      '#1B5E20'),
    (2, 'Agricultural',           'Ackerland',             'Toke bujqesore',   '#F9A825'),
    (3, 'Grassland and Shrubland','Gruenland und Buschland','Kullota dhe shkurre', '#9CCC65'),
    (4, 'Olive Groves',           'Olivenhaine',           'Ullishta',         '#827717'),
    (5, 'Urban',                  'Siedlung',              'Zona urbane',      '#C62828'),
    (6, 'Water',                  'Gewaesser',             'Ujera',            '#1565C0');


-- =====================================================================
-- 8. Hilfsabfragen
-- =====================================================================

-- Patch-Umriss aus dem Mittelpunkt berechnen (640m Kantenlaenge).
-- Nach dem Import der Mittelpunkte einmal ausfuehren.
--
-- UPDATE patches p
-- SET extent_geom = ST_Envelope(
--         ST_Buffer(p.center_geom::geography, (d.extent_m / 2.0))::geometry)
-- FROM dataset_versions d
-- WHERE p.dataset_version_id = d.id
--   AND p.extent_geom IS NULL;

-- Region je Patch per raeumlichem Join setzen.
--
-- UPDATE patches p
-- SET region_id = r.id
-- FROM regions r
-- WHERE r.level = 'bashki'
--   AND ST_Contains(r.geom, p.center_geom)
--   AND p.region_id IS NULL;

-- Klassenverteilung einer Region fuer ein Jahr.
--
-- SELECT c.name_de, COUNT(*) AS anzahl,
--        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS prozent
-- FROM predictions pr
-- JOIN patches p            ON p.id = pr.patch_id
-- JOIN land_cover_classes c ON c.id = pr.predicted_class_id
-- WHERE p.region_id = 7
--   AND p.year = 2021
--   AND pr.model_version_id = 1
-- GROUP BY c.name_de
-- ORDER BY anzahl DESC;
