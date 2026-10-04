# MOGRL — ficha de aquisição e leitura preliminar

Consulta: 03/10/2026. DOI: [10.1016/j.compenvurbsys.2026.102415](https://doi.org/10.1016/j.compenvurbsys.2026.102415).

**Estado: metadados coletados; texto integral e PDF não obtidos; reprodução não iniciada.** Não atribuir a esta ficha o status de revisão integral do artigo.

## Aquisição

Os arquivos em `data/reference/artigos/wang-cao-huang-2026-mogrl` contêm referência BibTeX, respostas de Crossref/OpenAlex/Semantic Scholar e registro das consultas com hashes SHA-256. São registros bibliográficos, não o artigo.

- [Editora](https://www.sciencedirect.com/science/article/pii/S0198971526000177): prévia pública, com acesso institucional/compra para o texto integral.
- [Página do coautor Kai Cao](https://kcao-ecnu.github.io/webpage/publication.html): referência direciona à editora.
- [OpenAlex](https://openalex.org/W7131622330): resposta da API classificou como fechado, sem localização aberta ou repositório com texto integral.
- [Semantic Scholar](https://www.semanticscholar.org/paper/4acc7dcfa26983248f46b2888ca14a77fe741316): campo de PDF aberto vazio.
- [ResearchGate](https://www.researchgate.net/publication/405540305_A_multi-objective_graph_reinforcement_learning_framework_for_urban_public_facility_location_problem): sem texto integral; oferece solicitação aos autores. Nenhuma mensagem foi enviada.
- Endpoint de texto da Elsevier indicado no Crossref: tentativa retornou HTTP 400. Isso não prova inexistência de acesso institucional; apenas registra a falha da consulta realizada.
- Não localizei código oficial ou preprint aberto nas buscas feitas. Isso não prova que não existam.

## Método confirmado na prévia

MOGRL combina soma ponderada dos objetivos, múltiplas políticas com transferência de parâmetros entre preferências vizinhas e GAT integrado a Transformer para seleção sequencial de instalações. A avaliação descrita inclui dados simulados/reais e comparação com NSGA-II e MOEA/D; o ambiente informado usa PyTorch, i9-13900K e RTX 4090. Esses elementos constam da [prévia oficial](https://www.sciencedirect.com/science/article/abs/pii/S0198971526000177); não foram reproduzidos.

## Interpretação para o nosso projeto — análise nossa

Há duas decomposições diferentes em discussão:

| Estratégia | O que é dividido | O que ainda precisa ser coordenado |
|---|---|---|
| Preferências multiobjetivo | Diferentes ponderações das metas | Comparabilidade de objetivos e diversidade das soluções |
| Partição espacial | Clientes/territórios/subproblemas | Capacidade, abertura compartilhada e atribuições globais |

O mecanismo de preferências confirmado na prévia não constitui evidência de implementação de clusters espaciais. Não sabemos se outra etapa do texto integral inclui agrupamento; a conclusão precisa ficar restrita ao material acessível.

Para ilustrar a diferença, considere nossa própria extensão hipotética, não uma equação transcrita do artigo:

`min λ * custo_normalizado(x) + (1 − λ) * desigualdade_normalizada(x)`.

Variações de λ mudam a preferência sobre soluções do mesmo território. Não reduzem automaticamente o número de clientes ou removem as restrições de capacidade. Já um agrupamento espacial altera quais decisões são tratadas juntas em cada subproblema.

Também não podemos transportar resultados de qualidade de uma fronteira de Pareto para uma conclusão de redução de gap no nosso SSCFLP. Primeiro é preciso verificar se as decisões, restrições e custos são equivalentes. Mesmo uma inferência neural rápida não garante uma atribuição capacitada viável.

Um exemplo matemático independente mostra por que soma ponderada não certifica toda fronteira discreta: para minimização, os pontos A=(0; 2), B=(1; 1,5) e C=(2; 0) são não dominados. Entretanto B nunca minimiza a soma ponderada: para λ≤0,5, C tem valor menor; para λ≥0,5, A tem valor menor. Essa observação é um critério para examinar as alegações de cobertura da fronteira, não uma acusação sobre os experimentos ainda não lidos.

## O que falta verificar no texto integral

1. Objetivos exatos, unidades, normalização e restrições; existência de capacidade, fonte única, orçamento e custos fixos.
2. Grafo: significado de nós/arestas, atributos, conectividade, coordenadas e distâncias viárias; eventual agrupamento e tratamento das fronteiras.
3. MDP: estado, ações, máscaras de viabilidade, transição, recompensa e término.
4. Treinamento: algoritmo de RL (não presumir PPO/REINFORCE), baseline, perdas, hiperparâmetros, sementes e estabilização.
5. Transferência: quais parâmetros são copiados, ordem dos vetores de preferência, camadas congeladas e quantidade de ajuste por subproblema.
6. Experimentos: cidades, bases, escalas, divisão treino/teste, métricas e configuração dos concorrentes. Não afirmar uso de hipervolume/IGD nem citar percentuais sem consultar tabelas e definições.
7. Tempo: treinamento versus inferência, hardware equivalente, custo de construir grafos e avaliação das soluções; custo de amortização por instância.
8. Reprodutibilidade: código, checkpoints, dados e ablações que separem efeito do grafo, transferência e múltiplas políticas.

## Decisão proposta

Manter este trabalho como referência potencial para uma extensão multiobjetivo. Não usá-lo, por enquanto, para justificar nossa decomposição geográfica ou escolher uma arquitetura de RL. A obtenção legítima do PDF por acesso institucional, cópia autorizada dos autores ou arquivo fornecido pelo usuário é a dependência restante para a leitura metodológica completa. Não houve compra, login institucional, contato com autores, treinamento ou execução de código desse artigo.
