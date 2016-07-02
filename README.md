# mbolado-learning
## Regressão Linear

### Objetivo:
Ajustar uma 'reta' a um conjunto de pontos

### Exemplo Básico:
```
X |  Y
1 | 2.1
2 | 3.9
3 | 5.9
4 | 8.1
```
Queremos encontrar o par `(m,b)` tal que a função `h(x) = m*x + b` melhor se ajuste aos pontos `(x,y)`

### Abordagem Inicial:
Defina uma função de erro (tambem chamada de função de custo) que mede o quão "boa" uma dada linha é. Essa função recebe o par `(m,b)` como parâmetro e retorna um valor de erro baseado em quão bem a nossa linha se ajusta aos dados.
Para medir este erro, itere sobre todos os pontos `(x,y)` do conjunto de dados e some a distancia da função `h(x) = m*x + b` ao `y` correspondente. Convencionalmente, usa-se a distância ao quadrado para que os valores sejam sempre positivos e a função de erro seja diferenciável.
```
J(m,b) = (1/N) * SUM((y[i] - (m*x[i] + b))**2) = (1/N) * SUM((y[i] - h(x[i]))**2)
```
Onde N é o numero de amostras.

A reta que melhor se ajusta aos pontos irá ter um erro menor. Tem-se, então, que minimizar esta função para que ela se aproxime o melhor possível do conjunto de dados.

A função de erro, por receber dois parâmetros, pode ser visualizada como uma superfície. Da maneira com que ela foi definida, é possível compará-la a um parabolóide elíptico, tendo apenas um ponto de mínimo.

### Método do Gradiente (Gradient Descent Search)
O Método do Gradiente é um método utilizado para encontrar um mínimo local de uma função de maneira iterativa. Em cada passo, toma-se a direção (negativa) do gradiente, onde o declive será máximo. Escolha o tamanho do passo `alpha` em direção ao declive pequeno o suficiente de tal forma que `(new_m,new_b) = (m,b) - alpha*GRADIENT(J(m,b))`, onde `J(new_m,new_b) <= J(m,b)`
Teremos então o seguinte sistema de equações:
```
new_m = m + alpha*(2/N)*SUM(-x[i]*(y[i] - h(x[i])))
new_b = b + alpha*(2/N)*SUM(y[i] - h(x[i]))
```
Se tomarmos um `alpha` grande demais, corremos o risco de ultrapassar o ponto de mínimo.

### Conjunto de dados bidimensionais
```
X1 | X2 | Y
 1 |  2 | 3
 2 |  4 | 6
-1 |  8 | 7
```
Nesse caso, queremos encontrar a tripla `(m1, m2, b)` tal que a função `h(x) = m1*x1 + m2*x2 + b` melhor se ajuste aos pontos `(x1, x2, y)`

É possível mostrar que todo cálculo feito na seção anterior é valido para qualquer número N de *features*




### Sources:
[An Introduction to Gradient Descent and Linear Regression - Matt Nedrich](https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/)

[Gradient Descent Article - Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent)
