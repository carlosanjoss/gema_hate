from __future__ import annotations


def build_prompt(text: str, max_chars: int = 1800) -> str:
    texto = str(text).strip()
    if len(texto) > max_chars:
        texto = texto[:max_chars]

    return f'''Você é um classificador de discurso de ódio em português brasileiro.

Classifique o comentário em duas etapas obrigatórias.

ETAPA 1 — Classificação binária:
- hate
- nao_hate

ETAPA 2 — Categoria:
Se, e somente se, a classificação binária for hate, escolha uma ou mais categorias válidas.

Categorias válidas:
ageism; aporophobia; body_shame; capacitism; lgbtphobia; political; racism; religious_intolerance; misogyny; xenophobia; other

DEFINIÇÃO GERAL:
Classifique como hate quando houver ataque, inferiorização, desumanização, hostilidade, ridicularização depreciativa, estereótipo negativo, desejo de dano, exclusão ou generalização ofensiva contra grupo social, identitário, político, religioso, racial, étnico, de gênero, sexualidade, deficiência, classe social, nacionalidade, origem regional ou aparência corporal.

Classifique como nao_hate quando o comentário apenas:
- discute preconceito;
- denuncia preconceito;
- cita fala ofensiva como exemplo;
- relata violência;
- critica discurso de ódio;
- fala sobre leis, conceitos, pesquisa ou notícia;
- contém palavrão sem alvo de grupo;
- faz ofensa individual sem relação clara com grupo social;
- faz crítica política comum sem desumanização, exclusão ou generalização hostil.

REGRA OBRIGATÓRIA:
Se h = 0 (nao_hate):
- c = null
- q = null

Se h = 1 (hate):
- c deve conter ao menos uma categoria válida
- q deve ser número entre 0 e 1

ATENÇÃO — MOVIMENTOS SOCIAIS IDENTITÁRIOS:
Ataques a movimentos sociais cujas pautas são definidas pelo pertencimento a grupos identitários não devem ser tratados automaticamente como crítica política comum.

Exemplos:
- feminismo;
- movimento negro;
- movimento LGBT+;
- movimentos de pessoas com deficiência;
- movimentos indígenas;
- movimentos contra racismo, misoginia, LGBTfobia, capacitismo ou xenofobia.

O feminismo é um movimento cujas protagonistas e objeto de defesa são as mulheres enquanto grupo social.
Ataques ao feminismo que generalizem negativamente sobre mulheres que o integram ou defendem constituem misoginia mediada pelo movimento.

Classifique como hate/misogyny quando o comentário:
- generaliza negativamente sobre mulheres por meio do ataque ao feminismo;
- afirma que feministas são manipuladoras, doentes mentais, degeneradas, promíscuas, enganadoras ou inimigas dos homens;
- diz que mulheres usam feminismo para enganar, controlar, explorar ou destruir homens;
- usa pseudoargumentos para desmoralizar, deslegitimar ou inferiorizar mulheres como grupo;
- afirma que mulheres não sofrem discriminação real como forma de ridicularizar ou negar a experiência coletiva das mulheres;
- descreve feminismo como doutrina de ódio aos homens, conspiração para destruir a família ou ferramenta para controlar homens;
- usa termos como feminazi, femismo, lacração feminista ou similares em contexto de demonização coletiva;
- usa vocabulário de manosfera/red pill com conteúdo misógino implícito.

NÃO classifique como hate/misogyny quando:
- há discordância de pauta específica sem generalização sobre mulheres;
- há crítica institucional ao movimento sem ataque às pessoas que o integram;
- o comentário defende posição conservadora sobre papéis de gênero sem hostilidade explícita ao grupo;
- o texto discute feminismo de modo analítico, histórico, acadêmico ou neutro.

ATENÇÃO — LINGUAGEM CODIFICADA E ALGOSPEAK:
Alguns grupos usam termos aparentemente neutros para mascarar discurso de ódio.

Termos e sinais de manosfera/red pill que podem carregar misoginia implícita:
redpilado; red pill; pílula vermelha; pílula azul; blue pill; stacy; chad; beta; alfa; hipergamia; valor sexual de mercado; colher usado como substituto de mulher; feminazi; femismo; msol quando usado de forma depreciativa; rodada quando usado como ataque generalizante; marmita quando usado como objetificação degradante; 304 ou outros códigos usados para insultar mulheres.

Esses termos isolados não bastam para classificar como hate.
Mas classifique como hate/misogyny quando aparecem junto de:
- generalização sobre mulheres;
- inferiorização feminina;
- desprezo por feministas;
- negação hostil de discriminação contra mulheres;
- teoria conspiratória sobre mulheres/feminismo;
- defesa de superioridade masculina;
- objetificação ou desumanização de mulheres.

POLÍTICA:
Crítica política comum não é automaticamente hate.
Classifique como political apenas quando houver hostilidade generalizante, desumanização, inferiorização forte, desejo de dano ou exclusão contra grupo político/social.

FORMATO OBRIGATÓRIO:
Retorne somente JSON válido, sem markdown e sem texto fora do JSON.
Retorne sempre as sete chaves compactas do formato, inclusive quando h for 0.

Exemplo para nao_hate:

{{
  "h": 0,
  "p": 0.10,
  "c": null,
  "q": null,
  "t": null,
  "e": "trecho relevante ou null quando não houver",
  "j": "explicação breve de por que o texto não constitui discurso de ódio"
}}

Exemplo para hate:

{{
  "h": 1,
  "p": 0.80,
  "c": "misogyny",
  "q": 0.80,
  "t": "mulheres/feministas",
  "e": "trecho usado como evidência",
  "j": "ataque generalizante contra mulheres mediado pelo feminismo"
}}

Na saída estruturada compacta: h=pred_hate (1 para hate, 0 para nao_hate);
p=hate_probability; c=pred_category; q=category_probability;
t=alvo_identificado; e=evidencia_textual; j=justificativa_curta.

Comentário:
"""{texto}"""'''.strip()


def build_binary_prompt(text: str, max_chars: int = 1800) -> str:
    texto = str(text).strip()
    if len(texto) > max_chars:
        texto = texto[:max_chars]

    return f'''Você é um classificador de discurso de ódio em português brasileiro.

DECISÃO BINÁRIA OBRIGATÓRIA:
- hate: ataque, inferiorização, desumanização, hostilidade, ridicularização
  depreciativa, estereótipo negativo, desejo de dano, exclusão ou
  generalização ofensiva contra grupo social, identitário, político,
  religioso, racial, étnico, de gênero, sexualidade, deficiência, classe,
  nacionalidade, origem regional, idade ou aparência corporal.
- nao_hate: discussão, denúncia ou citação de preconceito sem endosso; relato
  de violência; notícia, pesquisa ou conceito; palavrão sem alvo de grupo;
  ofensa apenas individual; crítica política comum sem desumanização,
  exclusão ou generalização hostil.

REGRAS QUE DEVEM SER PRESERVADAS:
- Ataques generalizantes a movimentos identitários podem atingir o grupo que
  representam. Ataques ao feminismo/feministas que demonizem, inferiorizem ou
  generalizem mulheres são hate, inclusive negação hostil da discriminação.
- Linguagem manosfera/red pill e códigos como feminazi, femismo, 304, rodada,
  marmita, hipergamia, stacy, chad ou colher só indicam hate quando o contexto
  expressa misoginia, objetificação, conspiração ou superioridade masculina.
- Crítica institucional ou discordância específica sem ataque ao grupo é
  nao_hate. Hostilidade generalizante contra grupo político é hate.

SAÍDA ESTRUTURADA:
Retorne somente h, p e j.
h é 1 para hate ou 0 para nao_hate; p é a probabilidade de o texto ser hate,
entre 0 e 1 (para h=0, p deve ser no máximo 0.49).
j deve ser exatamente um código:
no_group_attack; reported_or_discussed_hate; individual_offense;
ordinary_political_criticism; profanity_without_group;
neutral_or_irrelevant; potential_group_attack.
Use potential_group_attack sempre que h=1.

Texto:
"""{texto}"""'''.strip()


def build_binary_batch_prompt(texts: list[str], max_chars: int = 1800) -> str:
    if not texts:
        raise ValueError("O lote binário não pode ser vazio.")

    # Reutiliza literalmente a mesma rubrica da inferência unitária. Somente o
    # contrato de saída e a quantidade de textos mudam.
    single_prompt = build_binary_prompt("", max_chars)
    rubric = single_prompt.split("SAÍDA ESTRUTURADA:", 1)[0].rstrip()
    blocks = []
    for index, text in enumerate(texts):
        truncated = str(text).strip()[:max_chars]
        blocks.append(f'ITEM {index}:\n"""{truncated}"""')

    return f'''{rubric}

SAÍDA ESTRUTURADA EM LOTE:
Retorne somente um objeto com a chave r.
r deve ser uma lista com exatamente {len(texts)} resultados, na mesma ordem dos itens.
Cada resultado deve conter somente h, p e j.
Classifique cada item de modo independente: o conteúdo e a decisão de um item
jamais podem influenciar a decisão dos outros.
Use h=1 somente quando o próprio item contiver evidência clara de ataque
generalizante a um grupo. Mera menção a identidade, política, preconceito,
manosfera ou palavra sensível não basta; sem ataque claro ao grupo, use h=0.
h é 1 para hate ou 0 para nao_hate; p é a probabilidade de o texto ser hate,
entre 0 e 1 (para h=0, p deve ser no máximo 0.49).
j deve ser exatamente um código:
no_group_attack; reported_or_discussed_hate; individual_offense;
ordinary_political_criticism; profanity_without_group;
neutral_or_irrelevant; potential_group_attack.
Use potential_group_attack sempre que h=1.

TEXTOS:
{chr(10).join(blocks)}'''.strip()


def build_detailed_prompt(text: str, max_chars: int = 1800) -> str:
    texto = str(text).strip()[:max_chars]
    # Mantém literalmente a rubrica binária usada na primeira etapa e troca
    # apenas o contrato de saída pelo detalhamento completo. Isso evita repetir
    # o prompt histórico extenso e deixa contexto suficiente para fechar o JSON.
    binary_rubric = build_binary_prompt("", max_chars).split(
        "SAÍDA ESTRUTURADA:", 1
    )[0].rstrip()
    return f'''{binary_rubric}

CATEGORIAS VÁLIDAS PARA h=1:
ageism; aporophobia; body_shame; capacitism; lgbtphobia; political;
racism; religious_intolerance; misogyny; xenophobia; other.

DETALHAMENTO OBRIGATÓRIO:
- Se h=0: c=null, q=null, t=null e e=null.
- Se h=1: c deve ser uma categoria válida ou lista de categorias válidas;
  q deve ser a probabilidade da categoria entre 0 e 1.
- t é somente o grupo alvo, em até poucas palavras.
- e é somente um trecho curto do próprio texto; nunca copie o texto inteiro.
- j é uma justificativa curta e direta.
- Selecione somente categorias sustentadas pelo próprio texto. Não infira raça,
  religião, gênero, sexualidade ou deficiência sem evidência textual.

SAÍDA ESTRUTURADA:
Retorne somente h, p, c, q, t, e e j.
h é 1 para hate ou 0 para nao_hate; p é a probabilidade de ser hate.

Texto:
"""{texto}"""'''.strip()
