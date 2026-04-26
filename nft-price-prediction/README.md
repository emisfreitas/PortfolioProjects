# 🐒 BAYC NFT: Análise de Dados e Modelo Preditivo

Este projeto utiliza Ciência de Dados e Machine Learning para explorar o mercado da coleção *Bored Ape Yacht Club* (BAYC). O foco principal é entender a dinâmica de preços e construir modelos capazes de prever o valor de venda (Price ETH) em cenários de alta volatilidade.

* **Pipeline de Dados:** Extração e limpeza de trades via API/Blockchain.
* **Análise de Séries Temporais:** Implementação de Médias Móveis (*Rolling Means*) e Lags para capturar o momento do mercado.
* **Machine Learning:** Modelo de regressão (v4) focado em predição de curto prazo.
* **Visualização Avançada:** Dashboard técnico com análise de resíduos e importância de variáveis (*Feature Importance*).

---
📈 Resultados Técnicos 

* Pontuação R²: Negativo (Devido ao Data Drift estrutural de 2025).

* RMSE: 1,79 ETH.

> _O modelo apresenta R² negativo no teste devido ao data drift característico do mercado de NFTs em 2025, onde os preços do BAYC caíram ~80% no período. O RMSE absoluto de 1.78 ETH representa ~18% do preço médio, e a análise de feature importance revela que indicadores de momento recente dominam as previsões._

---

📁 Estrutura de Arquivos

* collection.py: Script de coleta e processamento inicial.

* eda.py: Análise Exploratória de Dados.

* model.py: Treinamento do modelo de Machine Learning.

* trades_processed_v4.csv: Base de dados final tratada.

   <!-- LICENSE -->
## License

MIT License

Copyright (c) 2026 Emily Freitas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
