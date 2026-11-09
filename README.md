# Gemma Hate Inference — Radar Ódio

Projeto de inferência com `gemma3:4b` via Ollama sobre os textos do PostgreSQL
de `../social_data_pipeline`.

Por padrão, a execução inclui Reddit, Twitter, YouTube, Facebook, TikTok,
WhatsApp e Telegram. As mensagens de WhatsApp e Telegram são armazenadas pelo
pipeline como `posts` e entram automaticamente em `python run.py run`.

O classificador usa somente as duas etapas do prompt
`gemma3_4b_structured_v2`:

1. classificação binária: `hate` ou `nao_hate`;
2. se for `hate`, uma ou mais categorias válidas.

Não existem níveis de hostilidade, escala de Allport ou dimensões adicionais
neste projeto.

## Dados gravados

Para cada post ou comentário processado com sucesso são persistidos:

- `pred_hate` e o `pred_label` correspondente, derivado da mesma decisão;
- `hate_probability`;
- `hate_types`, coluna `TEXT[]` em `gemma_hate_analyses` com zero, um ou vários tipos;
- uma ou mais categorias e `category_probability`, somente para `hate`;
- `alvo_identificado`;
- `evidencia_textual`;
- `justificativa_curta`;
- modelo, versão exata do prompt, texto analisado e hash;
- resposta bruta, tempo e número de tentativas.

Erros de servidor, JSON inválido ou resposta inconsistente são registrados como
`status = 'error'`. Eles não são convertidos em `nao_hate`.

A API do Ollama recebe um JSON Schema compacto. A primeira etapa exige somente
decisão, probabilidade e código de justificativa; a segunda exige as sete
propriedades de detalhamento e só é chamada para candidatos a `hate`.
`pred_label` é derivado de `pred_hate`, evitando respostas redundantes e
contraditórias. Textos de entrada continuam limitados por `MAX_TEXT_CHARS`
(padrão: 1.800 caracteres).

## Categorias

`ageism`, `aporophobia`, `body_shame`, `capacitism`, `lgbtphobia`, `political`,
`racism`, `religious_intolerance`, `misogyny`, `xenophobia` e `other`.

O formato aceito é o mesmo do script-base:

```json
{
  "pred_label": "hate",
  "pred_hate": 1,
  "hate_probability": 0.8,
  "pred_category": "misogyny",
  "category_probability": 0.8,
  "alvo_identificado": "mulheres/feministas",
  "evidencia_textual": "trecho usado como evidência",
  "justificativa_curta": "ataque generalizante contra mulheres"
}
```

## Hierarquia social preservada

As análises referenciam diretamente `posts.id` ou `comments.id`. A view
`v_gemma_hate_results` também resolve:

```text
post raiz (posts.id)
└── comentário (comments.post_pk)
    └── resposta (comments.parent_external_id → comments.external_id)
```

## Instalação

```powershell
cd gemma_hate_inference
py -V:Astral/CPython3.11.15 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

O `.env` local é opcional. Sem ele, o projeto carrega `POSTGRES_URI` de
`../social_data_pipeline/.env`.

O padrão `OLLAMA_NUM_GPU=999` solicita o máximo de offload possível. Na GTX 1650
de 4 GB deste ambiente, o `gemma3:4b` foi validado com 100% das camadas na GPU e
contexto de 2.304 tokens. Use `OLLAMA_NUM_GPU=0` somente para forçar CPU.

## Banco de dados

```powershell
python run.py migrate
```

Objetos principais:

| Objeto | Finalidade |
|---|---|
| `gemma_hate_runs` | execução, filtros, progresso e erros |
| `gemma_hate_analyses` | classificação binária, justificativa e auditoria |
| `gemma_hate_analysis_categories` | tipos de hate multirrótulo |
| `gemma_hate_categories` | catálogo das 11 categorias |
| `v_gemma_hate_results` | resultados com a hierarquia social resolvida |
| `v_gemma_hate_summary` | resumo por plataforma, rótulo e categoria |

## Execução

### Escolher uma rede social

No Windows, use o script `inferir_rede.cmd` seguido do nome da rede. Ele conta
os textos pendentes, pede confirmação e executa somente a plataforma escolhida:

```powershell
# Processar tudo que estiver pendente no Reddit
.\inferir_rede.cmd reddit

# Também aceita o nome abreviado digitado como "reddi"
.\inferir_rede.cmd reddi

# Processar somente comentários do TikTok
.\inferir_rede.cmd tiktok -Conteudo comments

# Testar somente 10 textos do YouTube
.\inferir_rede.cmd youtube -Limite 10

# Apenas mostrar a quantidade pendente, sem iniciar o Gemma
.\inferir_rede.cmd whatsapp -SomenteContar
```

As redes aceitas são `reddit`, `twitter` (ou `x`), `youtube`, `facebook`,
`tiktok`, `whatsapp` e `telegram`. Ao abrir `inferir_rede.cmd` com dois cliques,
o próprio script pergunta qual rede deve ser processada. Por segurança, ele não
inicia um segundo worker enquanto outra inferência estiver usando o modelo.

```powershell
# Somente contar os registros pendentes
python run.py run --dry-run

# Testar dez comentários do Reddit
python run.py run --content comments --platform reddit --limit 10

# Processar todo texto pendente
python run.py run

# Processar somente as mensagens de WhatsApp e Telegram
python run.py run --content posts --platform whatsapp --platform telegram

# Execução otimizada: mensageria primeiro e comentários em seguida
.\run_gpu_priority.ps1

# Consultar os runs
python run.py status
```

O `status` também mostra o progresso global, a meta exata de 70% e quantos
registros ainda faltam para alcançá-la.

Cada fila processa somente uma ocorrência canônica por texto distinto. Posts e
comentários repetidos continuam preservados nas tabelas sociais para manter a
frequência e a hierarquia originais, mas não geram cópias na tabela de
inferência. Uma restrição única por modelo, prompt e hash do texto também
impede duplicações em execuções simultâneas ou retomadas.

Depois das duplicatas, a etapa binária agrupa dinamicamente até 16 textos,
respeitando `INFERENCE_BATCH_MAX_CHARS` para não exceder o contexto. Candidatos
a `hate` continuam passando individualmente pela segunda etapa completa. Essa
etapa usa a mesma rubrica em formato compacto, restringe a saída às categorias
válidas e reserva 512 tokens para fechar o JSON mesmo em textos longos.

No terminal, a cada lote de 25 itens, o run informa tempo decorrido, velocidade
média, tempo restante estimado e horário previsto de término. O comando
`python run.py status` mostra a mesma estimativa com o progresso já persistido
no banco.

Não use `--only-usable` enquanto `is_usable` estiver nulo no banco atual. Sem
essa opção, todo texto não vazio dentro do recorte de idioma/data do
`social_data_pipeline` é elegível.

## Idempotência

Resultados `success` do mesmo modelo, versão do prompt e hash não são
reprocessados. Resultados com erro permanecem pendentes para uma nova tentativa.
Se o texto mudar, seu hash muda e uma nova avaliação é permitida.

Cada execução faz um preflight curto no Ollama. O circuit breaker interrompe o
run após 25 erros consecutivos; erros isolados ficam pendentes para nova
tentativa sem derrubar todo o processamento.

## Consultas

```sql
SELECT pred_label, pred_hate, hate_probability, pred_category,
       category_probability, alvo_identificado,
       evidencia_textual, justificativa_curta
FROM v_gemma_hate_results
WHERE status = 'success';

SELECT *
FROM v_gemma_hate_summary
ORDER BY total DESC;

SELECT analysis_id, model_name, external_id, error_message
FROM gemma_hate_analyses
WHERE status = 'error';
```

## Testes

```powershell
python -m pytest -q
```
