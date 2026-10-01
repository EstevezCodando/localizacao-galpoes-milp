# Extensão de roteirização e seleção de ferramentas

O projeto atual resolve localização capacitada e atribuição: decide quais centros abrir e qual centro atende cada região. Ele ainda não resolve o problema de rotas com veículos, horários e visitas. A tabela recebida é útil como mapa de evolução, mas não deve ser aplicada como se todas as ferramentas fossem equivalentes.

## Arquitetura recomendada

```text
previsão e dados territoriais
          ↓
localização capacitada (SSCFLP)  ← projeto atual
          ↓ centros e regiões atribuídas
roteirização diária (VRP/VRPTW)
          ↓ rotas, horários, equipes e veículos
execução e replanejamento
```

A fronteira entre os dois problemas é o conjunto de pedidos/visitas atribuídos a cada centro. Não é adequado colocar a distância percorrida por veículos diretamente no SSCFLP sem representar clientes, carga, sequência e jornada; isso muda o problema para uma formulação integrada de localização-roteamento e aumenta muito o custo computacional.

## Decisão sobre cada ferramenta

| Ferramenta | Entra agora? | Uso defensável |
|---|---|---|
| OR-Tools | Sim, já é dependência | Primeiro protótipo de VRP/VRPTW, com matriz de tempos, janelas, equipes, pausas, início às 6h e retorno até 17h |
| PyVRP | Depois, como comparação | Benchmark de qualidade em VRP rico; comparar mesma instância, matriz, restrições e orçamento |
| VROOM | Depois, para integração operacional | Motor de serviço/API quando o problema estiver definido e a matriz puder ser enviada em JSON; não é substituto da análise experimental |
| Pyomo + HiGHS | Opcional | Modelo auditável linear/misto e protótipos de variantes; não é obrigatório porque o projeto já tem OR-Tools e SCIP |
| Optuna | Só para calibração | Escolha de parâmetros de previsão/LNS com validação separada; nunca usar para “melhorar” o resultado no conjunto de teste |
| pymoo | Só se objetivos forem realmente conflitantes | Fronteira custo–tempo–cobertura–emissões, com regras de atendimento e comparação de soluções; não acrescenta valor a uma função escalar sem metas decisórias |
| PyPSA | Não pertence a esta extensão | Especializado em sistemas energéticos; só faria sentido se o escopo passasse a incluir energia, carregamento de veículos elétricos ou expansão elétrica |

Como OR-Tools já sustenta o MILP, o LP e o LNS do repositório, a escolha inicial deve ser reduzir o risco de modelagem: implementar uma instância pequena de VRPTW verificável por enumeração ou solução conhecida, antes de instalar cinco novas dependências.

## Formulação mínima da rota

Para cada veículo/equipe k e arco i→j, usar uma variável de visita. Restrições mínimas:

- cada visita obrigatória é atendida uma vez, ou recebe uma prioridade/penalidade explícita;
- capacidade de carga e, se necessário, volume/peso por veículo;
- início na base às 06:00 e retorno até 17:00;
- tempo de serviço por visita;
- janela de atendimento e espera;
- pausa de almoço e acessos indisponíveis;
- matriz de tempo viário real, com margem de segurança;
- compatibilidade entre equipe, veículo, região e tipo de serviço.

O objetivo inicial pode ser lexicográfico: primeiro maximizar atendimento/prioridade, depois minimizar atraso e horas extras, e por fim distância/custo. Somar tudo em uma única escala exige pesos justificados; sem isso, o algoritmo pode trocar uma visita importante por uma pequena economia de quilometragem.

## Dados que ainda faltam

Distância entre centroides é insuficiente para rota operacional. São necessários pontos de visita ou uma regra de desagregação, matriz de tempos por faixa horária, tempo de serviço, capacidade, equipe, jornada, calendário, restrições de acesso e definição do que significa “visita obrigatória”. A matriz deve ser congelada com data, perfil de veículo e origem do cálculo.

O resultado da localização deve ser avaliado em dois sentidos: custo do modelo de abertura/atribuição e custo de roteirizar as visitas atribuídas. Uma atribuição barata pode produzir muitas rotas fragmentadas; uma atribuição mais cara pode reduzir o número de veículos. Esse trade-off é precisamente o motivo para medir a segunda camada antes de afirmar que a rede é melhor.

## Protocolo de comparação

1. Criar instâncias sintéticas pequenas e uma instância real anonimizada.
2. Fixar visitas, matriz, jornada, pausas, capacidades, prioridades, seed e limite de tempo.
3. Validar viabilidade independentemente de cada solver.
4. Comparar custo, distância, atraso, visitas não atendidas, horas extras, número de veículos e tempo.
5. Usar OR-Tools como baseline funcional; PyVRP como comparação de qualidade; VROOM como teste de integração, não como prova de ótimo.
6. Reservar um conjunto de instâncias para teste e não calibrar parâmetros nele.

Não se deve afirmar “ótimo global” para OR-Tools, PyVRP ou VROOM em instâncias grandes apenas porque o solver terminou. A formulação pequena pode ser provada; na operação maior, registrar incumbente, limite inferior quando existir, status e tempo total.

## Ordem de implementação

**Fase 1 — contrato e exemplo.** Adicionar tipos para visita, veículo, jornada e matriz de tempo; construir um exemplo com 5–10 visitas e enumerar sequências quando possível.

**Fase 2 — OR-Tools.** Resolver VRPTW com capacidade, serviço, início/fim e prioridade. Exportar rotas e uma tabela de auditoria por visita.

**Fase 3 — comparação.** Adaptar a mesma instância para PyVRP e medir qualidade/tempo. Só então avaliar VROOM com um payload equivalente.

**Fase 4 — integração.** Passar a saída da localização para o VRP, recalcular o custo total da rede e testar se a abertura de centros muda quando o custo de roteamento entra.

**Fase 5 — multiobjetivo e tuning.** Usar Optuna apenas em validação interna; usar pymoo somente se o decisor aceitar uma fronteira de soluções. PyPSA permanece fora do escopo atual.
