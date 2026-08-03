-- Mantém uma única inferência bem-sucedida por texto/modelo/prompt.
-- As ocorrências sociais continuam preservadas em posts/comments; somente as
-- cópias de resultados do modelo são removidas.
--
-- O índice da autorreferência precisa existir antes da exclusão. Sem ele, o
-- PostgreSQL varre toda a tabela para validar a FK a cada linha removida.
CREATE INDEX IF NOT EXISTS idx_gemma_hate_reused_from
    ON gemma_hate_analyses (reused_from_analysis_id)
    WHERE reused_from_analysis_id IS NOT NULL;

CREATE TEMPORARY TABLE gemma_duplicate_analyses
ON COMMIT DROP
AS
WITH ranked AS (
    SELECT
        analysis_id,
        ROW_NUMBER() OVER (
            PARTITION BY model_name, prompt_version, text_hash
            ORDER BY
                (reused_from_analysis_id IS NULL) DESC,
                analysis_id
        ) AS position
    FROM gemma_hate_analyses
    WHERE status = 'success'
)
SELECT analysis_id
FROM gemma_hate_analyses
WHERE reused_from_analysis_id IS NOT NULL
UNION
SELECT analysis_id
FROM ranked
WHERE position > 1;

CREATE UNIQUE INDEX ON gemma_duplicate_analyses (analysis_id);

-- A exclusão explícita das categorias evita o custo de acionar a cascata
-- individualmente para dezenas de milhares de análises.
DELETE FROM gemma_hate_analysis_categories category
USING gemma_duplicate_analyses duplicate
WHERE category.analysis_id = duplicate.analysis_id;

DELETE FROM gemma_hate_analyses analysis
USING gemma_duplicate_analyses duplicate
WHERE analysis.analysis_id = duplicate.analysis_id;

CREATE UNIQUE INDEX IF NOT EXISTS uq_gemma_success_text_inference
    ON gemma_hate_analyses (model_name, prompt_version, text_hash)
    WHERE status = 'success';
