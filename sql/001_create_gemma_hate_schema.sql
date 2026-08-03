-- Esquema de duas etapas:
-- 1) hate / nao_hate
-- 2) categoria(s), somente quando hate

CREATE TABLE IF NOT EXISTS gemma_hate_categories (
    category_code  TEXT PRIMARY KEY,
    category_name  TEXT NOT NULL,
    description    TEXT,
    sort_order     SMALLINT NOT NULL,
    active         BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS gemma_hate_runs (
    run_id              BIGSERIAL PRIMARY KEY,
    run_name            TEXT NOT NULL UNIQUE,
    model_name          TEXT NOT NULL,
    prompt_version      TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running', 'completed', 'interrupted', 'failed')),
    filters             JSONB NOT NULL DEFAULT '{}'::jsonb,
    total_selected      INTEGER NOT NULL DEFAULT 0 CHECK (total_selected >= 0),
    processed_count     INTEGER NOT NULL DEFAULT 0 CHECK (processed_count >= 0),
    success_count       INTEGER NOT NULL DEFAULT 0 CHECK (success_count >= 0),
    error_count         INTEGER NOT NULL DEFAULT 0 CHECK (error_count >= 0),
    reused_count        INTEGER NOT NULL DEFAULT 0 CHECK (reused_count >= 0),
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at         TIMESTAMPTZ,
    error_message       TEXT,
    CHECK (processed_count = success_count + error_count),
    CHECK (reused_count <= success_count)
);

ALTER TABLE gemma_hate_runs
    ADD COLUMN IF NOT EXISTS reused_count INTEGER NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS gemma_hate_analyses (
    analysis_id          BIGSERIAL PRIMARY KEY,
    run_id               BIGINT NOT NULL REFERENCES gemma_hate_runs(run_id) ON DELETE CASCADE,
    content_kind         TEXT NOT NULL CHECK (content_kind IN ('post', 'comment')),
    post_id              INTEGER REFERENCES posts(id) ON DELETE CASCADE,
    comment_id           INTEGER REFERENCES comments(id) ON DELETE CASCADE,
    platform             VARCHAR(20) NOT NULL,
    external_id          VARCHAR(255) NOT NULL,
    text_hash            VARCHAR(64) NOT NULL,
    text_snapshot        TEXT NOT NULL,
    model_name           TEXT NOT NULL,
    prompt_version       TEXT NOT NULL,
    status               TEXT NOT NULL CHECK (status IN ('success', 'error')),
    is_hate              BOOLEAN,
    pred_label           TEXT GENERATED ALWAYS AS (
        CASE
            WHEN is_hate IS TRUE THEN 'hate'
            WHEN is_hate IS FALSE THEN 'nao_hate'
            ELSE NULL
        END
    ) STORED,
    pred_hate            SMALLINT GENERATED ALWAYS AS (
        CASE
            WHEN is_hate IS TRUE THEN 1
            WHEN is_hate IS FALSE THEN 0
            ELSE NULL
        END
    ) STORED,
    hate_probability     NUMERIC(7,6),
    hate_types           TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    target_identified    TEXT,
    textual_evidence     TEXT,
    justification        TEXT,
    raw_response         TEXT,
    error_message        TEXT,
    attempt_count        SMALLINT NOT NULL DEFAULT 1,
    inference_seconds    NUMERIC(12,4),
    reused_from_analysis_id BIGINT REFERENCES gemma_hate_analyses(analysis_id) ON DELETE SET NULL,
    analyzed_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE gemma_hate_analyses
    ADD COLUMN IF NOT EXISTS reused_from_analysis_id BIGINT
        REFERENCES gemma_hate_analyses(analysis_id) ON DELETE SET NULL;

CREATE TABLE IF NOT EXISTS gemma_hate_analysis_categories (
    analysis_id      BIGINT NOT NULL REFERENCES gemma_hate_analyses(analysis_id) ON DELETE CASCADE,
    category_code    TEXT NOT NULL REFERENCES gemma_hate_categories(category_code),
    probability      NUMERIC(7,6) NOT NULL CHECK (probability >= 0 AND probability <= 1),
    PRIMARY KEY (analysis_id, category_code)
);

INSERT INTO gemma_hate_categories
    (category_code, category_name, description, sort_order)
VALUES
    ('ageism', 'Etarismo', 'Hostilidade baseada em idade ou geração.', 1),
    ('aporophobia', 'Aporofobia', 'Hostilidade contra pessoas pobres ou em vulnerabilidade econômica.', 2),
    ('body_shame', 'Discriminação corporal', 'Ataque baseado no corpo ou aparência.', 3),
    ('capacitism', 'Capacitismo', 'Hostilidade baseada em deficiência ou capacidade.', 4),
    ('lgbtphobia', 'LGBTfobia', 'Hostilidade baseada em orientação sexual ou identidade de gênero.', 5),
    ('political', 'Hostilidade política', 'Hostilidade generalizante contra grupo político ou social.', 6),
    ('racism', 'Racismo', 'Hostilidade baseada em raça, cor ou etnia.', 7),
    ('religious_intolerance', 'Intolerância religiosa', 'Hostilidade baseada em religião ou crença.', 8),
    ('misogyny', 'Misoginia', 'Hostilidade contra mulheres enquanto grupo social.', 9),
    ('xenophobia', 'Xenofobia', 'Hostilidade baseada em nacionalidade ou origem regional.', 10),
    ('other', 'Outra categoria', 'Discurso de ódio fora das categorias anteriores.', 11)
ON CONFLICT (category_code) DO UPDATE SET
    category_name = EXCLUDED.category_name,
    description = EXCLUDED.description,
    sort_order = EXCLUDED.sort_order,
    active = TRUE;

-- Coluna desnormalizada para consulta direta e suporte a múltiplos tipos.
ALTER TABLE gemma_hate_analyses
    ADD COLUMN IF NOT EXISTS hate_types TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[];

-- Preenche/sincroniza instalações existentes a partir dos vínculos normalizados.
UPDATE gemma_hate_analyses a
SET hate_types = COALESCE(
    (
        SELECT ARRAY_AGG(ac.category_code ORDER BY ac.category_code)
        FROM gemma_hate_analysis_categories ac
        WHERE ac.analysis_id = a.analysis_id
    ),
    ARRAY[]::TEXT[]
)
WHERE a.hate_types IS DISTINCT FROM COALESCE(
    (
        SELECT ARRAY_AGG(ac.category_code ORDER BY ac.category_code)
        FROM gemma_hate_analysis_categories ac
        WHERE ac.analysis_id = a.analysis_id
    ),
    ARRAY[]::TEXT[]
);
