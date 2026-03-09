# HMM-From-Scratch

**Autor:** Edson Ricardo Pedroza Olivera  
**Versión:** 1.0  
**Repositorio:** [https://github.com/epedrozao/HMM-From-Scratch](https://github.com/epedrozao/HMM-From-Scratch)

---

## Descripción

Esta librería proporciona una implementación completa y educativa de **Modelos Ocultos de Markov (HMM)** con observaciones discretas, escrita en Python puro con soporte de NumPy. Incluye los algoritmos fundamentales:

- **Viterbi** (decodificación)
- **Forward** (evaluación)
- **Backward** (probabilidad hacia atrás)
- **Forward-Backward** (suavizado)
- **Baum-Welch** (entrenamiento EM)

Cada algoritmo está disponible en dos versiones:

- **Optimizada**: utiliza logaritmos y técnicas numéricamente estables para trabajar con secuencias largas sin underflow.
- **Teórica**: sigue fielmente las expresiones matemáticas de libro, ideal para el aprendizaje y la comprensión del algoritmo.

La librería está pensada tanto para fines educativos como para su uso en proyectos reales, y sirve como base para aplicaciones más especializadas en **procesamiento de lenguaje natural**, **finanzas cuantitativas**, **bioinformática** y otras áreas.

---

## Características

- Implementación limpia y comentada, con énfasis en la legibilidad y la eficiencia.
- Validación exhaustiva de parámetros y manejo de casos extremos (secuencias vacías, estados no visitados, etc.).
- Documentación completa en formato docstring, con ejemplos ejecutables (doctests).
- Versiones optimizadas y teóricas de todos los algoritmos.
- Compatible con Python 3.8+ y NumPy.

---

## Instalación

Clona el repositorio e instala con pip:

```bash
git clone https://github.com/epedrozao/HMM-From-Scratch.git
cd HMM-From-Scratch
pip install .
```

Luego importa la clase en tu proyecto:

```python
from hmm import HMM
```

> **Nota:** NumPy es la única dependencia y se instalará automáticamente.

---

## Uso rápido

```python
import numpy as np
from hmm import HMM

# Definir un modelo simple (2 estados, 2 observaciones)
A  = np.array([[0.7, 0.3],
               [0.4, 0.6]])
B  = np.array([[0.8, 0.2],
               [0.4, 0.6]])
pi = np.array([0.8, 0.2])

hmm = HMM(2, 2, A=A, B=B, pi=pi)

# Secuencia de observaciones (0: feliz, 1: gruñón)
obs = [0, 0, 1, 0]

# Decodificación con Viterbi
path, prob = hmm.viterbi(obs)
print("Estados más probables:", path)

# Log-verosimilitud con forward
_, log_lik = hmm.forward(obs, return_log=True)
print("Log-verosimilitud:", log_lik)

# Suavizado con forward-backward
gamma = hmm.forward_backward(obs)
print("Probabilidades suavizadas:\n", gamma)

# Estimación de parámetros con Baum-Welch (secuencia única)
obs_train = [0, 0, 1, 0, 0, 1, 0, 0]
log_lik_final = hmm.baum_welch(obs_train, max_iter=50)
print("Log-verosimilitud final tras el entrenamiento:", log_lik_final)

# Estimación de parámetros con Baum-Welch (múltiples secuencias)
obs_multi  = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0])
lengths    = [4, 3, 3]
log_lik_multi = hmm.baum_welch_mult(obs_multi, lengths, max_iter=50)
print("Log-verosimilitud final (multi-secuencia):", log_lik_multi)
```

---

## Documentación de métodos

### Constructor

```python
HMM(n_estados, n_obs, A=None, B=None, pi=None)
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `n_estados` | `int` | Número de estados ocultos N |
| `n_obs` | `int` | Número de posibles símbolos observables M |
| `A` | `ndarray`, opcional | Matriz de transición de shape (N, N). Cada fila debe sumar 1. |
| `B` | `ndarray`, opcional | Matriz de emisión de shape (N, M). Cada fila debe sumar 1. |
| `pi` | `ndarray`, opcional | Vector de probabilidades iniciales de shape (N,). Debe sumar 1. |

Si no se proporcionan, `A`, `B` y `pi` se inicializan aleatoriamente y se normalizan para que sean estocásticos.

---

### Métodos optimizados (numéricamente estables)

| Método | Descripción |
|--------|-------------|
| `viterbi(obs, return_log=False)` | Secuencia de estados más probable (decodificación Viterbi) |
| `forward(obs, return_log=False)` | Matriz forward α y verosimilitud total P(O\|λ) |
| `backward(obs, return_log=False)` | Matriz backward β |
| `forward_backward(obs, return_log=False)` | Probabilidades suavizadas γ |
| `baum_welch(obs, max_iter, tol, return_log_history)` | Estimación EM de parámetros — secuencia única |
| `baum_welch_mult(obs_concatenated, lengths, max_iter, tol, return_log_history)` | Estimación EM de parámetros — múltiples secuencias |

---

#### `viterbi(obs, return_log=False)`

Encuentra la secuencia de estados ocultos más probable para una secuencia de observaciones, usando log-probabilidades para evitar underflow numérico.

**Retorno:** `path` (list of int), `best_log_prob` o `best_prob` (float)

---

#### `forward(obs, return_log=False)`

Calcula la matriz forward α_t(j) = P(o_1, …, o_t, q_t = j | λ) y la verosimilitud total P(O | λ).

**Retorno:**
- Si `return_log=True`: `log_alpha` (ndarray, shape T×N), `log_likelihood` (float)
- Si `return_log=False`: `alpha` (ndarray), `likelihood` (float)

---

#### `backward(obs, return_log=False)`

Calcula la matriz backward β_t(j) = P(o_{t+1}, …, o_T | q_t = j, λ).

**Retorno:**
- Si `return_log=True`: `log_beta` (ndarray, shape T×N)
- Si `return_log=False`: `beta` (ndarray)

---

#### `forward_backward(obs, return_log=False)`

Calcula las probabilidades suavizadas γ_t(j) = P(q_t = j | O, λ).

**Retorno:**
- Si `return_log=True`: `log_gamma` (ndarray, shape T×N)
- Si `return_log=False`: `gamma` (ndarray, filas suman 1)

---

#### `baum_welch(obs, max_iter=100, tol=1e-6, return_log_history=False)`

Algoritmo de Expectation-Maximization (EM) para estimar los parámetros del modelo (π, A, B) a partir de una sola secuencia de observaciones. Itera hasta que el cambio relativo en la log-verosimilitud sea menor que `tol` o se alcance `max_iter`.

**Retorno:**
- Si `return_log_history=False`: `log_likelihood` (float)
- Si `return_log_history=True`: `log_likelihood` (float), `log_lik_history` (list of float)

---

#### `baum_welch_mult(obs_concatenated, lengths, max_iter=100, tol=1e-6, return_log_history=False)`

Extensión multi-secuencia de Baum-Welch. Acepta varias secuencias de observaciones independientes concatenadas en un único array 1-D, con sus longitudes individuales especificadas por separado. Las estadísticas suficientes (γ, ξ) se acumulan sobre todas las secuencias en el E-step y los parámetros se reestiman conjuntamente en el M-step, lo que equivale a maximizar la log-verosimilitud total Σ_s log P(obs_s | modelo).

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `obs_concatenated` | `array-like of int` | Todas las secuencias concatenadas en un array 1-D. |
| `lengths` | `array-like of int` | Longitud de cada secuencia individual. La suma debe coincidir con `len(obs_concatenated)`. |
| `max_iter` | `int` | Número máximo de iteraciones EM. |
| `tol` | `float` | Tolerancia de convergencia sobre el cambio relativo en la log-verosimilitud total. |
| `return_log_history` | `bool` | Si `True`, devuelve también la lista de log-verosimilitudes por iteración. |

**Retorno:**
- Si `return_log_history=False`: `log_likelihood` (float) — log-verosimilitud total final Σ_s log P(obs_s | modelo)
- Si `return_log_history=True`: `log_likelihood` (float), `log_lik_history` (list of float)

---

### Métodos teóricos (sin logaritmos, para aprendizaje)

Estos métodos siguen exactamente las fórmulas de libro y son ideales para entender el funcionamiento de los algoritmos.

> **Advertencia:** estas versiones trabajan en espacio lineal y pueden sufrir underflow numérico para secuencias largas. Usa las versiones optimizadas en producción.

| Método | Retorno |
|--------|---------|
| `viterbi_teorico(obs)` | `path` (list of int), `prob` (float) |
| `forward_teorico(obs)` | `alpha` (ndarray, T×N), `prob` (float) |
| `backward_teorico(obs)` | `beta` (ndarray, T×N) |
| `forward_backward_teorico(obs)` | `gamma` (ndarray, T×N) |
| `baum_welch_teorico(obs, max_iter, tol, return_log_history)` | igual que `baum_welch` |

---

## Estrategia Open Core

La librería base (`hmm.py`) se publica bajo la licencia MIT para fomentar su uso, estudio y contribución por parte de la comunidad. Esto permite:

- Demostrar capacidad técnica a reclutadores y clientes en plataformas como Toptal o Gun.io.
- Construir una reputación como desarrollador de herramientas de calidad.
- Servir como base para proyectos más complejos.

Sobre este núcleo abierto, se han desarrollado aplicaciones especializadas en dos áreas de alto valor comercial:

- **Procesamiento de lenguaje natural (NLP):** etiquetado POS, reconocimiento de entidades, generación de texto basada en HMM.
- **Modelos financieros:** detección de regímenes de mercado, predicción de series temporales, análisis de riesgos.

Estas aplicaciones se distribuyen bajo licencia comercial y están disponibles para empresas y profesionales que deseen integrarlas en sus productos o servicios. Si estás interesado, contacta al autor para discutir los términos.

Este modelo Open Core garantiza que la herramienta fundamental permanezca libre y accesible, mientras que las soluciones específicas de negocio se comercializan de manera justa, protegiendo la propiedad intelectual y permitiendo el desarrollo continuo.

---

## Licencia

- **Núcleo (`hmm.py`):** MIT License. Puedes usar, copiar, modificar y distribuir este código libremente, siempre que se mantenga el aviso de copyright.
- **Aplicaciones especializadas (NLP, finanzas):** Licencia comercial. Para más información, contacta al autor.

---

## Contacto

- **Email:** epedroz2@itam.mx · edsonpedroza99@gmail.com
- **LinkedIn:** [Edson Ricardo Pedroza Olivera](https://www.linkedin.com/in/edson-pedroza-b0108232a)
- **GitHub:** [@epedrozao](https://github.com/epedrozao)

Para preguntas, sugerencias o interés en las aplicaciones comerciales, no dudes en escribir.

---

## Referencias

- Rabiner, L. R. (1989). *A tutorial on hidden Markov models and selected applications in speech recognition.* Proceedings of the IEEE.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.* Springer.
- Wikipedia: [Hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model)

---

*¡Disfruta explorando el mundo de los HMM!*
