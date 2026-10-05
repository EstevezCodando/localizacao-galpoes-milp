# Pré-registro do teste fechado

*Registrado em 04/10/2026, antes de qualquer execução nas instâncias de teste, corredor, escala,
Holmberg e Olist com os métodos abaixo. O commit deste arquivo antecede o das execuções.*

## 1. O que está congelado

- **Código**: o commit em que este arquivo entra. Nenhum parâmetro muda depois dele.
- **Dados e modelos**: `data/frozen/LOCK.json` (o executor confere o hash antes de rodar).
- **Parâmetros**: os mesmos dos pilotos. CLNS com clusters de 15 clientes, limite de 0,5 s por
  subproblema (até 3× após falhas), partida pela solução da expansão; híbrido com α = 0,5;
  expansão em 20/40/60/100% com um quarto do tempo restante por estágio; Kernel Search com 25% do
  tempo restante para o núcleo.
- **Protocolo de tempo**: v3 (um núcleo físico por execução, uma thread, relógios de parede e de
  CPU, aquecimento fora do cronômetro).

## 2. Métodos

| Método | Componente aprendido | Sementes |
|---|---|---:|
| `completo` — SCIP 10 no modelo completo | nenhum | 1 |
| `kernel` — Kernel Search | nenhum | 1 |
| `adaptativa:pl` — expansão adaptativa ordenada pelo PL | nenhum | 1 |
| `adaptativa:gnn` — expansão adaptativa ordenada pela GNN | GNN | 1 |
| `hibridopl:rotacao` — expansão PL + CLNS com rotação | nenhum | 3 |
| `hibrido:rotacao` — expansão GNN + CLNS com rotação | GNN | 3 |

A memória de subproblemas e os seletores aprendido e guiado pela GNN ficam fora: nos pilotos não
se separaram da rotação, e incluí-los multiplicaria as comparações sem hipótese nova.

## 3. Conjuntos e orçamentos

| Conjunto | Instâncias | Tamanho | Orçamento | Referência |
|---|---:|---|---:|---|
| `teste` | 32 | 30 × 150 | 60 s | melhor conhecida |
| `gen_corredor` | 16 | 30 × 150 | 60 s | melhor conhecida (família nunca vista) |
| `gen_escala` | 16 | 50 × 200 | 120 s | melhor conhecida |
| `holmberg` | 71 | 10–30 × 50–200 | 30 s | ótimo publicado |
| `olist` | 4 | 30 × 150 a 150 × 850 | 120 s | melhor conhecida |

A GNN foi treinada só em 30 × 150 das famílias uniforme e clusters. Corredor, escala, Holmberg e
Olist medem generalização dela; o ranking do PL não depende de treino.

## 4. Hipóteses

Métrica primária: integral primal no orçamento. Secundária: desvio final. Unidade de análise: a
instância (sementes agregadas antes). Teste: Wilcoxon pareado bilateral. Correção de Holm sobre
as quatro hipóteses, separadamente para a métrica primária e para a secundária, no conjunto
agregado `teste` + `gen_corredor` + `gen_escala` (64 instâncias). Holmberg e Olist são relatados
à parte, de forma descritiva (o Olist tem só 4 instâncias).

- **H1 (restrição ajuda):** `adaptativa:pl` tem integral primal menor que `completo`.
- **H2 (aprendizado):** `adaptativa:gnn` difere de `adaptativa:pl`. Os pilotos sugerem que não
  difere, ou que o PL é melhor. Um resultado não significativo será relatado como "sem evidência
  de diferença", com o intervalo de confiança da diferença, e **não** como equivalência.
- **H3 (fase de CLNS):** `hibridopl:rotacao` difere de `adaptativa:pl`. No piloto 3 o efeito foi
  misto; a direção não é prevista.
- **H4 (clássico estabelecido):** `adaptativa:pl` difere de `kernel`.

## 5. O que será relatado em qualquer caso

- Tabela por conjunto: integral primal (média, IC 95% por bootstrap), desvio final mediano,
  parcela a até 1% e a até 0,1%, vitórias/empates/derrotas contra `completo`.
- Em Holmberg: desvio em relação ao ótimo publicado e número de instâncias em que o ótimo foi
  atingido no orçamento.
- Número de instâncias em que o certificado de custo reduzido provou o ótimo antes do último
  estágio.
- Qualidade da medição: distribuição da razão CPU ÷ parede, execuções marcadas e estouros.
- Todas as execuções entram na análise, inclusive as marcadas por contenção. Uma análise de
  sensibilidade sem as marcadas será relatada ao lado.

## 6. O que não será feito

- Nenhum ajuste de parâmetro depois de ver resultados de teste.
- Nenhuma troca de método principal em função do resultado: se a GNN vencer o PL no teste, isso
  é relatado como contradição dos pilotos.
- A ablação de α e as vizinhanças de maior alcance são trabalho posterior, em validação.

## 7. Desvios registrados durante a execução

- **04/10/2026, conjunto `teste`.** A máquina entrou em suspensão por ociosidade por cerca de
  4 h no meio da rodada. As três execuções que estavam em curso ficaram com tempo de parede de
  mais de 15.000 s e razão CPU ÷ parede de 0,002. Foram descartadas e refeitas. Critério,
  definido sem olhar para custos: estouro de orçamento acima de 60 s **e** razão CPU ÷ parede
  abaixo de 0,1, o que só ocorre com o processo parado. As linhas descartadas ficam em
  `results/pesquisa/fechado/descartadas_suspensao.csv`. As demais 165 execuções já gravadas
  terminaram antes da suspensão e foram mantidas. A partir daqui o executor é lançado por
  `scripts/rodar_acordado.py`, que impede a suspensão por ociosidade enquanto roda; o código
  dos métodos não mudou.
- **04/10/2026, conjunto `teste`, retomada.** Uma segunda suspensão, de cerca de 10 minutos,
  atingiu outras três execuções (tempo de parede de 646 a 654 s, razão CPU ÷ parede de 0,06 a
  0,07). O mesmo critério foi aplicado por `scripts/descartar_suspensas.py`, que passa a ser
  executado ao fim de cada conjunto, antes de qualquer análise. Três execuções com razão entre
  0,80 e 0,85 (contenção, sem suspensão) foram **mantidas** e ficam marcadas.

## 8. Emenda da análise, feita depois de ver os resultados (05/10/2026)

O parecer acadêmico de 04/10/2026 (`parecer-academico-previa-20261004.md`, seção 3.3) apontou
que o desvio final era calculado pelo objetivo devolvido ao fim da execução, e não por `g(T)`,
como o manuscrito define. Uma solução encontrada durante um estouro de orçamento entrava no
desvio final sem estar disponível em `T`. A correção foi aplicada **depois** de a primeira
análise ter sido vista, e por isso fica registrada como emenda, com as duas versões guardadas:

- `scripts/analisar_fechado.py` passa a reconstruir o desvio final pelo melhor incumbente com
  instante ≤ `T`; execução sem solução até `T` conta como desvio 1 e permanece no denominador.
- Efeito medido: 4 das 1.390 execuções tinham melhorado depois de `T` (1 em `teste`, 2 em
  `gen_corredor`, 1 em `gen_escala`); 2 execuções do Olist não têm solução até `T`.
- As tabelas das 64 instâncias sintéticas não mudam em duas casas decimais. Nos testes do desvio
  final, H3 passa de p (Holm) = 0,0496 para 0,0495 e H4 de 0,0499 para 0,0495. Integral primal,
  H1 e H2 não mudam. No Olist, o desvio médio da expansão passa a incluir a instância sem
  solução (desvio 1).
- Versão anterior: `results/pesquisa/fechado/analise_v1_objetivo_de_retorno.json` e
  `tabelas_v1_objetivo_de_retorno.md`.

Nenhuma hipótese, método, conjunto ou teste estatístico foi alterado.

## 9. Emenda: segunda rodada em hardware dedicado (registrada em 05/10/2026, antes de rodá-la)

**Motivo.** Na primeira rodada, 16,5% das execuções ficaram marcadas por contenção (máquina
ocupada durante `gen_corredor` e `gen_escala`) e houve duas suspensões. O motivo é de medição e
não depende do sentido dos resultados.

**O que será feito.** O teste fechado inteiro (os cinco conjuntos, os seis métodos, os mesmos
orçamentos e sementes) será repetido em uma instância de nuvem com núcleos dedicados:

- AWS EC2 `c7i.2xlarge`, Intel Xeon Platinum 8488C, 4 núcleos físicos com 1 thread por núcleo,
  16 GB, Ubuntu 24.04; 3 execuções simultâneas, um núcleo livre para o sistema, como no protocolo.
- Código: `src/alocacao_capacitada/pesquisa` idêntico byte a byte ao do commit do pré-registro
  (conferido pelos hashes de blob do git). Dados e modelos conferidos pela trava
  `data/frozen/LOCK.json` na instância.
- Única diferença de ambiente: `scripts/linux/corrigir_highs.sh` renomeia o SONAME da `libhighs`
  do pacote `highspy`, porque no Linux ela colide com a do `ortools`. Não altera código nem
  versões de biblioteca.
- Uma calibração de 24 execuções em 4 instâncias de **validação** foi feita antes para conferir
  que o ambiente funciona e que a razão CPU ÷ parede fica perto de 1. Nenhuma instância de teste
  foi executada na nuvem antes desta emenda.

**Como as duas rodadas serão lidas (fixado agora).**

- As duas rodadas são relatadas lado a lado, com a mesma análise (`scripts/analisar_fechado.py`,
  desvio final em T). Nenhuma é descartada.
- Tempos absolutos e integrais não são comparáveis entre as rodadas (hardware diferente); cada
  rodada tem a própria referência (BKS) calculada dentro dela.
- Uma hipótese é tratada como **confirmada** somente se for significativa (Holm, 5%) na rodada
  da nuvem **e** tiver o mesmo sinal na rodada local. Se as rodadas discordarem em sinal, ou se
  só uma for significativa, o resultado é relatado como **não replicado**.
- Os critérios de descarte por suspensão e a marca de contenção valem como antes.
