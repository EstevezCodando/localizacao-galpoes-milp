# Auditoria incremental do CLNS — 03/10/2026

O workspace já continha CLNS, seletor aprendido e pilotos antes desta auditoria.
Esta rodada corrige a implementação existente; não representa reprodução do MOGRL.

## Correções

- O subproblema desconta validação e montagem do orçamento antes de chamar SCIP.
  Se não sobra tempo, preserva o incumbente. A montagem e chamadas nativas ainda
  não são preemptíveis; não há garantia de prazo rígido.
- Rejeita atribuições inviáveis e índices livres duplicados, fracionários ou fora
  dos limites, evitando modelos que contradizem o contrato da decomposição.
- Distâncias Manhattan usam `scipy.spatial.distance.cdist`, sem tensor intermediário
  centros × clientes × clientes. Para 50 × 600, um único tensor float64 anterior
  teria 144 MB, contra 2,88 MB da matriz de distâncias. Este é um cálculo de tamanho,
  não uma medição de pico de memória; ordenação e outras estruturas também ocupam RAM.
- Perfis iguais/custos zerados não geram divisão por zero no agrupamento e nos
  atributos. O número efetivo de grupos pode ser inferior ao solicitado.
- Corrigida documentação que anunciava reparo guloso inexistente. A falha do
  construtivo continua explícita e não é prova de inviabilidade do problema.

## Piloto diagnóstico

Script: `scripts/piloto_clns_auditoria.py`. Saída:
`results/pesquisa/auditoria_clns_20261003.json`.

Três instâncias sintéticas da família clusters: 30 × 150, 50 × 200 e 50 × 850;
razão de capacidade 1,5; semente dos dados 20261003; semente de busca 0.
Orçamento cooperativo de 5 segundos, execução sequencial e bibliotecas numéricas
limitadas a uma thread. Mede solução devolvida e tempo real, sem truncar trajetória
no horizonte: os resultados são diagnósticos, não métricas oficiais de benchmark.

Compara SCIP com dica, LNS convencional e CLNS por rotação com alvo de 1, 3 e 5
grupos. SCIP e CLNS recebem o mesmo construtivo, cobrado no orçamento. O LNS
convencional usa sua inicialização própria; também paga a construção de referência
usada na comparação, o que é uma pequena desvantagem deste piloto. CLNS sem PL
para isolar a busca. Agrupamento por perfis de custos, não por coordenadas.
Não testa decomposição independente paralela, GNN nem seletor aprendido.

Uma semente e três instâncias não sustentam generalização ou significância.
850 clientes sintéticos não equivalem à validação sobre as regiões do Olist.

## Pendências relevantes

1. Auditar o relógio externo da inferência GNN em `exp_clns` e recalcular métricas
   dos pilotos anteriores estritamente no horizonte antes de comparar métodos.
2. Tratar falhas do construtivo em instâncias de capacidade apertada; testar casos
   viáveis em que escolhas gulosas bloqueiam os clientes restantes.
3. Comparar vários tamanhos de vizinhança, múltiplas sementes e famílias, com 60/120 s,
   incluindo PL e inicialização em todos os orçamentos. Separar efeito do agrupamento
   do efeito de liberar quantidades diferentes de clientes.
4. Só então abrir a avaliação final congelada e o caso Olist. Não escolher o método
   usando os resultados do conjunto de teste.
