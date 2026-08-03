-- Migra instalações da versão anterior (quatro camadas) para somente duas etapas.

DROP VIEW IF EXISTS v_gemma_hate_summary;
DROP VIEW IF EXISTS v_gemma_hate_results;

DROP TRIGGER IF EXISTS trg_validate_gemma_hate_categories ON gemma_hate_analyses;
DROP FUNCTION IF EXISTS validate_gemma_hate_categories();

DROP INDEX IF EXISTS idx_gemma_hate_analyses_hostility;
DROP INDEX IF EXISTS idx_gemma_hate_dimensions_code;
DROP INDEX IF EXISTS uq_gemma_hate_one_primary_category;
DROP INDEX IF EXISTS idx_gemma_hate_categories_code;

DO $$
DECLARE
    constraint_row RECORD;
BEGIN
    FOR constraint_row IN
        SELECT conname
        FROM pg_constraint
        WHERE conrelid = 'gemma_hate_analyses'::regclass
          AND contype = 'c'
    LOOP
        EXECUTE FORMAT(
            'ALTER TABLE gemma_hate_analyses DROP CONSTRAINT %I',
            constraint_row.conname
        );
    END LOOP;
END;
$$;

ALTER TABLE gemma_hate_analyses
    DROP COLUMN IF EXISTS hostility_level_code CASCADE,
    DROP COLUMN IF EXISTS hostility_score CASCADE;

ALTER TABLE gemma_hate_analysis_categories
    DROP COLUMN IF EXISTS is_primary;

DROP TABLE IF EXISTS gemma_hate_analysis_dimensions;
DROP TABLE IF EXISTS gemma_hate_dimensions;
DROP TABLE IF EXISTS gemma_hate_hostility_levels;

ALTER TABLE gemma_hate_analyses
    ADD COLUMN IF NOT EXISTS pred_label TEXT GENERATED ALWAYS AS (
        CASE
            WHEN is_hate IS TRUE THEN 'hate'
            WHEN is_hate IS FALSE THEN 'nao_hate'
            ELSE NULL
        END
    ) STORED,
    ADD COLUMN IF NOT EXISTS pred_hate SMALLINT GENERATED ALWAYS AS (
        CASE
            WHEN is_hate IS TRUE THEN 1
            WHEN is_hate IS FALSE THEN 0
            ELSE NULL
        END
    ) STORED;

ALTER TABLE gemma_hate_analyses
    ADD CONSTRAINT ck_gemma_hate_content_reference CHECK (
        (content_kind = 'post' AND post_id IS NOT NULL AND comment_id IS NULL)
        OR
        (content_kind = 'comment' AND comment_id IS NOT NULL AND post_id IS NULL)
    ),
    ADD CONSTRAINT ck_gemma_hate_content_kind CHECK (
        content_kind IN ('post', 'comment')
    ),
    ADD CONSTRAINT ck_gemma_hate_status CHECK (
        status IN ('success', 'error')
    ),
    ADD CONSTRAINT ck_gemma_hate_probability CHECK (
        hate_probability IS NULL
        OR (hate_probability >= 0 AND hate_probability <= 1)
    ),
    ADD CONSTRAINT ck_gemma_hate_types CHECK (
        array_position(hate_types, NULL) IS NULL
        AND hate_types <@ ARRAY[
            'ageism', 'aporophobia', 'body_shame', 'capacitism',
            'lgbtphobia', 'political', 'racism', 'religious_intolerance',
            'misogyny', 'xenophobia', 'other'
        ]::TEXT[]
        AND (
            status <> 'success'
            OR (is_hate IS TRUE AND cardinality(hate_types) > 0)
            OR (is_hate IS FALSE AND cardinality(hate_types) = 0)
        )
    ),
    ADD CONSTRAINT ck_gemma_hate_attempt_count CHECK (attempt_count > 0),
    ADD CONSTRAINT ck_gemma_hate_inference_seconds CHECK (
        inference_seconds IS NULL OR inference_seconds >= 0
    ),
    ADD CONSTRAINT ck_gemma_hate_success_fields CHECK (
        status <> 'success'
        OR (
            is_hate IS NOT NULL
            AND hate_probability IS NOT NULL
            AND NULLIF(BTRIM(justification), '') IS NOT NULL
        )
    ),
    ADD CONSTRAINT ck_gemma_hate_error_message CHECK (
        status <> 'error' OR NULLIF(BTRIM(error_message), '') IS NOT NULL
    );

CREATE UNIQUE INDEX IF NOT EXISTS uq_gemma_hate_success_post
    ON gemma_hate_analyses (post_id, model_name, prompt_version, text_hash)
    WHERE content_kind = 'post' AND status = 'success';

CREATE UNIQUE INDEX IF NOT EXISTS uq_gemma_hate_success_comment
    ON gemma_hate_analyses (comment_id, model_name, prompt_version, text_hash)
    WHERE content_kind = 'comment' AND status = 'success';

CREATE INDEX IF NOT EXISTS idx_gemma_hate_analyses_run
    ON gemma_hate_analyses (run_id);
CREATE INDEX IF NOT EXISTS idx_gemma_hate_analyses_label
    ON gemma_hate_analyses (is_hate) WHERE status = 'success';
CREATE INDEX IF NOT EXISTS idx_gemma_hate_reuse
    ON gemma_hate_analyses (model_name, prompt_version, text_hash)
    WHERE status = 'success';
CREATE INDEX IF NOT EXISTS idx_gemma_hate_categories_code
    ON gemma_hate_analysis_categories (category_code);

CREATE INDEX IF NOT EXISTS idx_posts_inference_text_hash
    ON posts (MD5(
        CASE
            WHEN NULLIF(BTRIM(title), '') IS NOT NULL
             AND NULLIF(BTRIM(text), '') IS NOT NULL
                THEN BTRIM(title) || E'\n\n' || BTRIM(text)
            ELSE COALESCE(
                NULLIF(BTRIM(title), ''),
                NULLIF(BTRIM(text), ''),
                ''
            )
        END
    ));
CREATE INDEX IF NOT EXISTS idx_comments_inference_text_hash
    ON comments (MD5(BTRIM(text)));

CREATE OR REPLACE FUNCTION validate_gemma_hate_categories()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    category_count INTEGER;
    stored_categories TEXT[];
BEGIN
    IF NEW.status <> 'success' THEN
        RETURN NEW;
    END IF;

    SELECT COUNT(*), COALESCE(
        ARRAY_AGG(category_code ORDER BY category_code),
        ARRAY[]::TEXT[]
    )
    INTO category_count, stored_categories
    FROM gemma_hate_analysis_categories
    WHERE analysis_id = NEW.analysis_id;

    IF NEW.is_hate AND category_count = 0 THEN
        RAISE EXCEPTION
            'Análise hate % deve ter ao menos uma categoria', NEW.analysis_id;
    END IF;

    IF NOT NEW.is_hate AND category_count <> 0 THEN
        RAISE EXCEPTION
            'Análise nao_hate % não pode ter categorias', NEW.analysis_id;
    END IF;

    IF NEW.hate_types IS DISTINCT FROM stored_categories THEN
        RAISE EXCEPTION
            'hate_types da análise % não corresponde às categorias detalhadas',
            NEW.analysis_id;
    END IF;

    RETURN NEW;
END;
$$;

CREATE CONSTRAINT TRIGGER trg_validate_gemma_hate_categories
AFTER INSERT OR UPDATE OF status, is_hate, hate_types ON gemma_hate_analyses
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW
EXECUTE FUNCTION validate_gemma_hate_categories();

CREATE OR REPLACE VIEW v_gemma_hate_results AS
SELECT
    a.analysis_id,
    a.run_id,
    a.content_kind,
    a.platform,
    a.external_id,
    a.post_id AS source_post_id,
    a.comment_id AS source_comment_id,
    COALESCE(a.post_id, c.post_pk) AS root_post_id,
    pc.id AS parent_comment_id,
    c.parent_external_id,
    c.is_reply,
    c.depth,
    a.status,
    a.pred_label,
    a.pred_hate,
    a.hate_probability,
    a.hate_types,
    cats.pred_category,
    cats.category_probability,
    cats.categories,
    a.target_identified AS alvo_identificado,
    a.textual_evidence AS evidencia_textual,
    a.justification AS justificativa_curta,
    a.model_name,
    a.prompt_version,
    a.inference_seconds,
    a.reused_from_analysis_id,
    a.analyzed_at
FROM gemma_hate_analyses a
LEFT JOIN comments c ON c.id = a.comment_id
LEFT JOIN comments pc
    ON pc.platform = c.platform
   AND pc.external_id = c.parent_external_id
LEFT JOIN LATERAL (
    SELECT
        STRING_AGG(ac.category_code, '; ' ORDER BY ac.category_code) AS pred_category,
        MAX(ac.probability) AS category_probability,
        ARRAY_AGG(ac.category_code ORDER BY ac.category_code) AS categories
    FROM gemma_hate_analysis_categories ac
    WHERE ac.analysis_id = a.analysis_id
) cats ON TRUE;

CREATE OR REPLACE VIEW v_gemma_hate_summary AS
SELECT
    r.model_name,
    r.prompt_version,
    r.platform,
    r.content_kind,
    r.pred_label,
    r.pred_category,
    COUNT(*) AS total
FROM v_gemma_hate_results r
WHERE r.status = 'success'
GROUP BY
    r.model_name, r.prompt_version, r.platform, r.content_kind,
    r.pred_label, r.pred_category;
