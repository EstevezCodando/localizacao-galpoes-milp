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
