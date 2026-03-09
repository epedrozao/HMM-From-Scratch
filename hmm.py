"""

Implementación de Modelos Ocultos de Markov (HMM).

Algoritmos: Viterbi, Forward-Backward, Baum-Welch.

Autor: Edson Ricardo Pedroza Olivera.

"""

import numpy as np


class HMM:

    """

    Clase para Modelos Ocultos de Markov con observaciones discretas.

    Parámetros:
        n_estados (int): Número de estados ocultos.
        n_obs     (int): Número de posibles observaciones distintas.
        A  (ndarray): Matriz de transición de shape (n_estados x n_estados).
                      Cada fila debe sumar 1 (estocástica por filas).
        B  (ndarray): Matriz de emisión de shape (n_estados x n_obs).
                      Cada fila debe sumar 1 (estocástica por filas).
        pi (ndarray): Vector de probabilidades iniciales de shape (n_estados,).
                      Debe sumar 1.

    Si A, B o pi no se proporcionan, se inicializan aleatoriamente
    con distribuciones uniformes normalizadas.

    """
    # ------------------------------------------------------------------
    # Métodos estáticos para control de excepciones.
    # ------------------------------------------------------------------

    @staticmethod
    def _validar_estocastica(matriz, nombre, axis=1, tol=1e-6):
        """
        Verifica que las filas (o el vector) de una matriz sumen 1.
        
        Args:
            matriz  (ndarray): Matriz o vector a validar.
            nombre  (str):     Nombre del parámetro (para el mensaje de error).
            axis    (int):     Eje sobre el que se suman (1 para matrices, 0 para vectores).
            tol     (float):   Tolerancia numérica.
        
        Raises:
            ValueError: Si alguna fila (o el vector) no suma 1 dentro de la tolerancia.
        """
        sumas = matriz.sum(axis=axis)
        if not np.allclose(sumas, 1.0, atol=tol):
            raise ValueError(
                f"{nombre} debe ser estocástica (filas suman 1). "
                f"Sumas obtenidas: {sumas}.")
        
    @staticmethod
    def _log_sum_exp(x, axis=0):       
        """

        Calcula log(sum(exp(x))) de forma numéricamente estable,
        sustrayendo el máximo antes de exponenciar.

        Útil para combinar log-probabilidades sin underflow/overflow.

        Args:
            x    (ndarray): Array de log-probabilidades.
            axis (int):     Eje sobre el que se reduce (0 para columnas, 1 para filas).

        Returns:
            result (ndarray): Resultado de log(sum(exp(x))) con shape reducido en `axis`.

        """
        x_max = np.max(x, axis=axis, keepdims=True)
        return np.log(np.sum(np.exp(x - x_max), axis=axis)) + np.squeeze(x_max, axis=axis)
    
    def __init__(self, n_estados: int, n_obs: int,
                 A=None, B=None, pi=None):

        if n_estados <= 0 or n_obs <= 0:
            raise ValueError("n_estados y n_obs deben ser enteros positivos.")

        self.n_estados = n_estados
        self.n_obs = n_obs

        # --- Matriz de transición A ---

        if A is None:
            self.A = np.random.rand(n_estados, n_estados)
            self.A /= self.A.sum(axis=1, keepdims=True)  # Normalizar filas → estocástica.
        else:
            A = np.asarray(A, dtype=float)
            if A.shape != (n_estados, n_estados):
                raise ValueError(f"A debe tener shape ({n_estados}, {n_estados}), "
                                 f"se recibió {A.shape}.")
            self._validar_estocastica(A, "A", axis=1)
            self.A = A

        # --- Matriz de emisión B ---

        if B is None:
            self.B = np.random.rand(n_estados, n_obs)
            self.B /= self.B.sum(axis=1, keepdims=True)  # Normalizar filas → estocástica.
        else:
            B = np.asarray(B, dtype=float)
            if B.shape != (n_estados, n_obs):
                raise ValueError(f"B debe tener shape ({n_estados}, {n_obs}), "
                                 f"se recibió {B.shape}.")
            self._validar_estocastica(B, "B", axis=1)
            self.B = B

        # --- Vector de probabilidades iniciales pi ---

        if pi is None:
            self.pi = np.random.rand(n_estados)
            self.pi /= self.pi.sum()  # Normalizar → distribución de probabilidad.
        else:
            pi = np.asarray(pi, dtype=float)
            if pi.shape != (n_estados,):
                raise ValueError(f"pi debe tener shape ({n_estados},), "
                                 f"se recibió {pi.shape}.")
            self._validar_estocastica(pi, "pi", axis=0)
            self.pi = pi

        # Precalcular logaritmos para eficiencia en inferencia.

        self._EPS = 1e-300
        self._log_A  = np.log(self.A  + self._EPS)
        self._log_B  = np.log(self.B  + self._EPS)
        self._log_pi = np.log(self.pi + self._EPS)

    # ------------------------------------------------------------------
    # Métodos de inferencia
    # ------------------------------------------------------------------

    def viterbi(self, obs, return_log=False):
        """
        Algoritmo de Viterbi: encuentra la secuencia de estados ocultos
        más probable dada una secuencia de observaciones.

        Usa log-probabilidades para evitar underflow numérico al trabajar
        con secuencias largas.

        Args:
            obs (array-like of int): Secuencia de T observaciones.
                Cada elemento debe ser un índice entero en [0, n_obs).
            return_log (bool): Si True, devuelve log-probabilidad.
                Si False (por defecto), devuelve probabilidad lineal.
        Returns:
            Si return_log=True:
                path (list of int): Secuencia de estados más probable.
                best_log_prob (float): Log-probabilidad de dicha secuencia.
            Si return_log=False:
                path (list of int): Secuencia de estados más probable.
                best_prob (float): Probabilidad lineal de dicha secuencia.

        Raises:
            TypeError:  Si obs no es un array 1-D de enteros.
            ValueError: Si alguna observación está fuera del rango [0, n_obs).

        Ejemplo:
            >>> import numpy as np
            >>> A  = np.array([[0.7, 0.3], [0.4, 0.6]])
            >>> B  = np.array([[0.8, 0.2], [0.4, 0.6]])
            >>> pi = np.array([0.8, 0.2])
            >>> hmm = HMM(2, 2, A=A, B=B, pi=pi)
            >>> path, p = hmm.viterbi([0, 0, 1, 0])
            >>> print(path)
            [0, 0, 0, 0]

        """
        obs = np.asarray(obs, dtype=int)

        if obs.ndim != 1:
            raise TypeError("obs debe ser un array 1-D de índices enteros.")
        if obs.size > 0 and (obs.min() < 0 or obs.max() >= self.n_obs):
            raise ValueError(f"Todas las observaciones deben estar en [0, {self.n_obs}). "
                             f"Se encontró rango [{obs.min()}, {obs.max()}].")

        T = len(obs)

        if T == 0:
            return [], 0.0

        J = self.n_estados

        # Precomputar log-emisiones para toda la secuencia: shape (T, J)
        # Evita indexar self._log_B en cada paso del loop.

        log_emit = self._log_B[:, obs].T  # (T, J)

        # --- Inicialización (t = 0) ---
        # V[t, j]: log-probabilidad máxima de cualquier camino que termine
        #          en el estado j en el instante t. 
        #          Trabajar con Log-probabilidad permite usar sumas de matrices en vez de productos 
        #          y es más eficiente en términos numéricos (evita underflow numérico).
        # psi[t, j]: estado anterior que alcanza ese máximo.

        V   = np.empty((T, J))
        psi = np.empty((T, J), dtype=int)

        V[0] = self._log_pi + log_emit[0]

        # --- Recursión (t = 1 … T-1) ---
        # Para cada estado j, buscamos el estado previo i que maximiza:
        #   V[t-1, i] + log A[i, j] + log B[j, obs[t]]
        # La suma V[t-1, :] + log_A[:, j] se vectoriza sobre todos los i a la vez.
        
        for t in range(1, T):

            # trans[j] = V[t-1, :] + log_A[:, j]  → shape (J, J)
            trans = V[t - 1, :, np.newaxis] + self._log_A  # shape (J, J)
            psi[t]  = np.argmax(trans, axis=0) # psi[t] guarda el mejor estado previo para cada destino j; # V[t] obtiene el valor correspondiente.
            V[t]    = trans[psi[t], np.arange(J)] + log_emit[t] # Selecciona, para cada columna j, el valor de la fila que maximiza (índice en psi[t, j])

        # --- Terminación ---
        best_last_state = int(np.argmax(V[T - 1]))
        best_log_prob   = float(V[T - 1, best_last_state])

        # --- Backtracking ---
        # Reconstruimos el camino óptimo hacia atrás usando psi.
        path = [best_last_state]
        for t in range(T - 1, 0, -1):
            path.append(int(psi[t, path[-1]]))

        path.reverse() # La reconstrucción se hizo desde el final, invertimos para obtener orden temporal.

        if return_log:
            return path, best_log_prob
        else:
            best_prob = np.exp(best_log_prob)
            return path, best_prob  

    # ------------------------------------------------------------------

    def viterbi_teorico(self, obs):
        """
        Versión teórica del algoritmo de Viterbi (sin logaritmos). 
        Código útil para la comprensión teórica del algoritmo.
        Sigue fielmente las expresiones matemáticas:
            V_1(j) = pi_j · b_j(o_1)
            V_t(j) = max_i [ V_{t-1}(i) · a_{ij} ] · b_j(o_t)
            psi_t(j) = argmax_i [ V_{t-1}(i) · a_{ij} ]

        Args:
            obs (array-like of int): Secuencia de T observaciones.
                Cada elemento debe ser un índice entero en [0, n_obs).

        Returns:
            path (list of int): Secuencia de T estados más probable.
            prob (float):       Probabilidad lineal de dicha secuencia.

        Raises:
            TypeError:  Si obs no es un array 1-D de enteros.
            ValueError: Si alguna observación está fuera del rango [0, n_obs).    
        
        Advertencia:
            Para secuencias largas, los valores pueden sufrir underflow.
            En producción se recomienda usar la versión logarítmica `viterbi`.
        
        Ejemplo:
            >>> import numpy as np
            >>> A  = np.array([[0.7, 0.3], [0.4, 0.6]])
            >>> B  = np.array([[0.8, 0.2], [0.4, 0.6]])
            >>> pi = np.array([0.8, 0.2])
            >>> hmm = HMM(2, 2, A=A, B=B, pi=pi)
            >>> path, prob = hmm.viterbi_teorico([0, 0, 1, 0])
            >>> print(path)
            [0, 0, 0, 0]

        """
        obs = np.asarray(obs, dtype=int)

        # Validaciones (idénticas a las del método viterbi optimizado)

        if obs.ndim != 1:
            raise TypeError("obs debe ser un array 1-D de índices enteros.")
        if obs.size > 0 and (obs.min() < 0 or obs.max() >= self.n_obs):
            raise ValueError(f"Todas las observaciones deben estar en [0, {self.n_obs}). "
                            f"Se encontró rango [{obs.min()}, {obs.max()}].")

        T = len(obs)
        if T == 0:
            return [], 1.0

        J = self.n_estados

        # delta[t, j]: probabilidad del mejor camino hasta el instante t que termina en el estado j
        delta = np.zeros((T, J))
        # psi[t, j]: estado anterior que maximiza dicho camino (backpointer)
        psi = np.zeros((T, J), dtype=int)

        # --- Inicialización (t = 0) ---
        delta[0] = self.pi * self.B[:, obs[0]]

        # --- Recursión (t = 1 … T-1) ---
        for t in range(1, T):
            for j in range(J):
                # Calculamos max_i [ delta[t-1, i] * self.A[i, j] ]
                max_val = -1.0
                max_i = -1
                for i in range(J):
                    valor = delta[t-1, i] * self.A[i, j]
                    if valor > max_val:
                        max_val = valor
                        max_i = i
                # Multiplicamos por la emisión b_j(obs[t])
                delta[t, j] = max_val * self.B[j, obs[t]]
                psi[t, j] = max_i

        # --- Terminación ---
        best_last_state = int(np.argmax(delta[T-1]))
        prob = delta[T-1, best_last_state]

        # --- Backtracking (reconstrucción del camino) ---
        path = [best_last_state]
        for t in range(T-1, 0, -1):
            path.append(int(psi[t, path[-1]]))
        path.reverse()

        return path, prob

    # ------------------------------------------------------------------

    def forward(self, obs, return_log=False):
        """
        Algoritmo Forward: calcula alpha[t, j], la probabilidad conjunta de
        observar la secuencia parcial obs[0..t] y estar en el estado j en t,
        dado el modelo (A, B, pi).

        Usa log-sum-exp en cada paso para evitar underflow numérico.

        Args:
            obs (array-like of int): Secuencia de T observaciones.
                Cada elemento debe ser un índice entero en [0, n_obs).
            return_log (bool): Si True, devuelve log-probabilidades.       
                Si False (por defecto), devuelve probabilidades lineales

        Returns:
            Si return_log=True:
                log_alpha (ndarray): shape (T, n_estados) con las log-probabilidades forward.
                log_likelihood (float): Log-verosimilitud de la secuencia, log P(obs | modelo).
            Si return_log=False:
                alpha (ndarray): shape (T, n_estados) con las probabilidades forward lineales.
                likelihood (float): Verosimilitud de la secuencia, P(obs | modelo).

        Raises:
            TypeError:  Si obs no es un array 1-D de enteros.
            ValueError: Si alguna observación está fuera de [0, n_obs).

        Ejemplo:
            >>> import numpy as np
            >>> A  = np.array([[0.7, 0.3], [0.4, 0.6]])
            >>> B  = np.array([[0.9, 0.1], [0.2, 0.8]])
            >>> pi = np.array([0.6, 0.4])
            >>> hmm = HMM(2, 2, A=A, B=B, pi=pi)
            >>> log_alpha, log_lik = hmm.forward([0, 0, 1, 0], return_log=True)
            >>> print(log_alpha.shape)
            (4, 2)
            >>> print(round(log_lik, 4))
            -2.6427

        """
        obs = np.asarray(obs, dtype=int)

        if obs.ndim != 1:
            raise TypeError("obs debe ser un array 1-D de índices enteros.")
        if obs.size > 0 and (obs.min() < 0 or obs.max() >= self.n_obs):
            raise ValueError(
                f"Todas las observaciones deben estar en [0, {self.n_obs}). "
                f"Se encontró rango [{obs.min()}, {obs.max()}].")

        T = len(obs)
        if T == 0:
            return np.empty((0, self.n_estados)), 0.0

        J = self.n_estados

        # Precomputar log-emisiones para toda la secuencia: shape (T, J)
        # Evita indexar self._log_B dentro del loop.
        log_emit = self._log_B[:, obs].T  # (T, J)

        log_alpha = np.empty((T, J))

        # --- Inicialización (t = 0) ---
        # log alpha_0(j) = log pi_j + log b_j(obs[0])
        log_alpha[0] = self._log_pi + log_emit[0]

        # --- Recursión (t = 1 … T-1) ---
        # log alpha_t(j) = log b_j(obs[t]) + log  sum_i [ alpha_{t-1}(i) * A[i,j] ]
        #                = log b_j(obs[t]) + log_sum_exp( log_alpha[t-1] + log_A[:, j] )
        #
        # log_sum_exp se aplica sobre los estados anteriores i (axis=0),
        # sumando para cada estado destino j.
        for t in range(1, T):
            # scores[i, j] = log_alpha[t-1, i] + log_A[i, j]  → shape (J, J)
            scores = log_alpha[t - 1, :, np.newaxis] + self._log_A  # (J, J)

            # log_sum_exp sobre i (axis=0) → shape (J,), uno por estado destino j.
            log_alpha[t] = log_emit[t] + self._log_sum_exp(scores, axis=0)

        # --- Verosimilitud ---
        # P(obs | modelo) = sum_j alpha_T(j)  →  en log: log_sum_exp sobre j.
        log_likelihood = float(self._log_sum_exp(log_alpha[T - 1], axis=0))

        if return_log:
            return log_alpha, log_likelihood
        else:
            alpha = np.exp(log_alpha)
            likelihood = np.exp(log_likelihood)    
            return alpha, likelihood
        
    # ------------------------------------------------------------------

    def forward_teorico(self, obs):
        """
        Versión teórica del algoritmo Forward (sin logaritmos). 
        Código útil para la comprensión teórica del algoritmo.
        Sigue fielmente las expresiones matemáticas:

            alpha_1(j) = pi_j · b_j(o_1)
            alpha_t(j) = sum[i in I](alpha_t-1(i) · a_{ij}) · b_j(o_t)
            P(O | lambda) = sum[j in J](alpha_T(j))

        Args:
            obs (array-like of int): Secuencia de T observaciones.
                Cada elemento debe ser un índice entero en [0, n_obs).

        Returns:
            alpha (ndarray): shape (T, n_estados). Probabilidades forward αₜ(j).
            prob  (float):   Verosimilitud lineal P(obs | modelo).

        Raises:
            TypeError:  Si obs no es un array 1-D de enteros.
            ValueError: Si alguna observación está fuera de [0, n_obs).

        Advertencia:
            Para secuencias largas, los valores pueden sufrir underflow.
            En producción se recomienda usar la versión logarítmica `forward`.

        Ejemplo:
            >>> import numpy as np
            >>> A  = np.array([[0.7, 0.3], [0.4, 0.6]])
            >>> B  = np.array([[0.9, 0.1], [0.2, 0.8]])
            >>> pi = np.array([0.6, 0.4])
            >>> hmm = HMM(2, 2, A=A, B=B, pi=pi)
            >>> alpha, prob = hmm.forward_teorico([0, 0, 1, 0])
            >>> print(alpha.shape)
            (4, 2)
            >>> print(round(prob, 6))
            0.071168
        """
        obs = np.asarray(obs, dtype=int)

        # Validaciones (idénticas a las del método forward optimizado)
        if obs.ndim != 1:
            raise TypeError("obs debe ser un array 1-D de índices enteros.")
        if obs.size > 0 and (obs.min() < 0 or obs.max() >= self.n_obs):
            raise ValueError(
                f"Todas las observaciones deben estar en [0, {self.n_obs}). "
                f"Se encontró rango [{obs.min()}, {obs.max()}].")

        T = len(obs)
        if T == 0:
            return np.empty((0, self.n_estados)), 1.0

        J = self.n_estados

        # Matriz forward: alpha[t, j] = αₜ(j)
        alpha = np.zeros((T, J))

        # --- Inicialización (t = 0) ---
        # α₀(j) = πⱼ · bⱼ(obs[0])
        alpha[0] = self.pi * self.B[:, obs[0]]

        # --- Recursión (t = 1 … T-1) ---
        for t in range(1, T):
            for j in range(J):
                # Suma sobre estados anteriores i: ∑ᵢ αₜ₋₁(i) · aᵢⱼ
                suma = np.sum(alpha[t-1, :] * self.A[:, j])
                # Multiplicar por la emisión actual bⱼ(obs[t])
                alpha[t, j] = suma * self.B[j, obs[t]]

        # --- Terminación ---
        prob = float(np.sum(alpha[T-1, :]))

        return alpha, prob
    
    # ------------------------------------------------------------------

    def backward(self, obs, return_log=False):
        """
        Algoritmo Backward: calcula beta[t, j], la probabilidad de observar
        la secuencia parcial obs[t+1..T-1] dado que se está en el estado j
        en el instante t, según el modelo (A, B, pi).

        Usa log-sum-exp en cada paso para evitar underflow numérico.

        Args:
            obs (array-like of int): Secuencia de T observaciones. Cada elemento debe ser un índice entero en [0, n_obs).
            return_log (bool): Si True, devuelve log-probabilidades. Si False (por defecto), devuelve probabilidades lineales.

        Returns:
            Si return_log=True:
                log_beta (ndarray): shape (T, n_estados) con log-probabilidades backward.
            Si return_log=False:
                beta (ndarray): shape (T, n_estados) con probabilidades backward lineales.

        Raises:
            TypeError:  Si obs no es un array 1-D de enteros.
            ValueError: Si alguna observación está fuera de [0, n_obs).

        Nota:
            La log-verosimilitud de la secuencia completa, log P(obs | modelo),
            puede obtenerse a partir de las probabilidades forward como
            log P(obs) = log_sum_exp(log_alpha[T-1, :]) (ver método forward).
            No se retorna directamente en backward para evitar redundancia con 
            el método forward, ya que el cálculo sería equivalente y requeriría 
            combinar log_beta[0, :] con self._log_pi y log_emit[0].

        Ejemplo:
            >>> import numpy as np
            >>> A  = np.array([[0.7, 0.3], [0.4, 0.6]])
            >>> B  = np.array([[0.9, 0.1], [0.2, 0.8]])
            >>> pi = np.array([0.6, 0.4])
            >>> hmm = HMM(2, 2, A=A, B=B, pi=pi)
            >>> log_beta = hmm.backward([0, 0, 1, 0], return_log=True)
            >>> print(log_beta.shape)
            (4, 2)

        """
        obs = np.asarray(obs, dtype=int)

        if obs.ndim != 1:
            raise TypeError("obs debe ser un array 1-D de índices enteros.")
        if obs.size > 0 and (obs.min() < 0 or obs.max() >= self.n_obs):
            raise ValueError(
                f"Todas las observaciones deben estar en [0, {self.n_obs}). "
                f"Se encontró rango [{obs.min()}, {obs.max()}].")

        T = len(obs)
        if T == 0:
            return np.empty((0, self.n_estados))

        J = self.n_estados

        # Precomputar log-emisiones para toda la secuencia: shape (T, J)
        # Evita indexar self._log_B dentro del loop.
        log_emit = self._log_B[:, obs].T  # (T, J)

        log_beta = np.empty((T, J))

        # --- Inicialización (t = T-1) ---
        # beta_{T-1}(j) = 1 para todo j  →  log beta_{T-1}(j) = 0
        log_beta[T - 1] = 0.0

        # --- Recursión (t = T-2 … 0) ---
        # beta_t(j) = sum_k [ A[j,k] · b_k(obs[t+1]) · beta_{t+1}(k) ]
        #           en log:
        # log_beta_t(j) = log_sum_exp( log_A[j,:] + log_emit[t+1] + log_beta[t+1] )
        #
        # log_sum_exp se aplica sobre los estados destino k (axis=1),
        # acumulando para cada estado origen j.
        for t in range(T - 2, -1, -1):
            # scores[j, k] = log_A[j,k] + log_emit[t+1, k] + log_beta[t+1, k]  → shape (J, J)
            scores = self._log_A + log_emit[t + 1] + log_beta[t + 1]  # (J, J)

            # log_sum_exp sobre k (axis=1) → shape (J,), uno por estado origen j.
            log_beta[t] = self._log_sum_exp(scores, axis=1)

        if return_log:
            return log_beta
        else:
            beta = np.exp(log_beta)
            return beta
        
    # ------------------------------------------------------------------

    def backward_teorico(self, obs):
        """
        Versión teórica del algoritmo Backward (sin logaritmos).
        Código útil para la comprensión teórica del algoritmo.
        Sigue fielmente las expresiones matemáticas:

            beta_{T-1}(j) = 1  para todo j
            beta_t(j) = sum_k [ a_{jk} · b_k(o_{t+1}) · beta_{t+1}(k) ]  para t = T-2, …, 0

        donde T es la longitud de la secuencia de observaciones y los índices t están
        basados en cero (t=0 corresponde al primer instante).

        Args:
            obs (array-like of int): Secuencia de T observaciones.
                Cada elemento debe ser un índice entero en [0, n_obs).

        Returns:
            beta (ndarray): shape (T, n_estados). Probabilidades backward beta_t(j).

        Raises:
            TypeError:  Si obs no es un array 1-D de enteros.
            ValueError: Si alguna observación está fuera de [0, n_obs).

        Advertencia:
            Para secuencias largas, los valores pueden sufrir underflow.
            En producción se recomienda usar la versión logarítmica `backward`.

        Nota:
            La verosimilitud P(obs | modelo) puede obtenerse como
            sum_j pi_j * b_j(obs[0]) * beta[0, j], pero este cálculo es
            redundante con el método forward_teorico, por lo que no se retorna.
            

        Ejemplo:
            >>> import numpy as np
            >>> A  = np.array([[0.7, 0.3], [0.4, 0.6]])
            >>> B  = np.array([[0.9, 0.1], [0.2, 0.8]])
            >>> pi = np.array([0.6, 0.4])
            >>> hmm = HMM(2, 2, A=A, B=B, pi=pi)
            >>> beta = hmm.backward_teorico([0, 0, 1, 0])
            >>> print(beta.shape)
            (4, 2)
        """
        obs = np.asarray(obs, dtype=int)

        # Validaciones (idénticas a las del método backward optimizado)
        if obs.ndim != 1:
            raise TypeError("obs debe ser un array 1-D de índices enteros.")
        if obs.size > 0 and (obs.min() < 0 or obs.max() >= self.n_obs):
            raise ValueError(
                f"Todas las observaciones deben estar en [0, {self.n_obs}). "
                f"Se encontró rango [{obs.min()}, {obs.max()}].")

        T = len(obs)
        if T == 0:
            return np.empty((0, self.n_estados))

        J = self.n_estados

        # Matriz backward: beta[t, j] = beta_t(j)  (t en 0-based, j en 0-based)
        beta = np.zeros((T, J))

        # --- Inicialización (t = T-1) ---
        # beta_{T-1}(j) = 1 para todo j (último instante)
        beta[T - 1, :] = 1.0

        # --- Recursión hacia atrás (t = T-2 … 0) ---
        for t in range(T - 2, -1, -1):
            for j in range(J):
                # Suma sobre todos los estados futuros k:
                # beta_t(j) = sum_k a_{jk} * b_k(obs[t+1]) * beta_{t+1}(k)
                suma = 0.0
                for k in range(J):
                    suma += self.A[j, k] * self.B[k, obs[t + 1]] * beta[t + 1, k]
                beta[t, j] = suma

        return beta
    
    # ------------------------------------------------------------------

    def forward_backward(self, obs, return_log=False):
        """
        Calcula las probabilidades suavizadas gamma[t, j] = P(estado_t = j | obs, modelo).

        Utiliza los algoritmos forward y backward para obtener las distribuciones
        a posteriori de los estados en cada instante, combinando alpha y beta.

        Args:
            obs (array-like of int): Secuencia de T observaciones.
                Cada elemento debe ser un índice entero en [0, n_obs).
            return_log (bool): Si True, devuelve log-probabilidades.
                Si False (por defecto), devuelve probabilidades lineales.

        Returns:
            Si return_log=True:
                log_gamma (ndarray): shape (T, n_estados) con log-probabilidades suavizadas.
            Si return_log=False:
                gamma (ndarray): shape (T, n_estados) con probabilidades suavizadas lineales.

        Raises:
            TypeError:  Si obs no es un array 1-D de enteros.
            ValueError: Si alguna observación está fuera de [0, n_obs).

        Nota:
            Las probabilidades gamma se calculan como:
                gamma[t, j] = alpha[t, j] * beta[t, j] / P(obs)
            donde alpha y beta se obtienen de los métodos forward y backward,
            y P(obs) es la verosimilitud. Para evitar underflow, internamente se
            trabaja en escala logarítmica. Si return_log=False, se exponencia y
            se renormaliza para garantizar suma 1 por fila.

        Ejemplo:
            >>> import numpy as np
            >>> A  = np.array([[0.7, 0.3], [0.4, 0.6]])
            >>> B  = np.array([[0.9, 0.1], [0.2, 0.8]])
            >>> pi = np.array([0.6, 0.4])
            >>> hmm = HMM(2, 2, A=A, B=B, pi=pi)
            >>> gamma = hmm.forward_backward([0, 0, 1, 0])
            >>> print(gamma.shape)
            (4, 2)
            >>> # Las filas deben sumar 1
            >>> print(np.allclose(gamma.sum(axis=1), 1.0))
            True
        """
        obs = np.asarray(obs, dtype=int)

        if obs.ndim != 1:
            raise TypeError("obs debe ser un array 1-D de índices enteros.")
        if obs.size > 0 and (obs.min() < 0 or obs.max() >= self.n_obs):
            raise ValueError(
                f"Todas las observaciones deben estar en [0, {self.n_obs}). "
                f"Se encontró rango [{obs.min()}, {obs.max()}].")

        T = len(obs)
        if T == 0:
            return np.empty((0, self.n_estados))

        # Obtener versiones logarítmicas (numéricamente estables)
        log_alpha, log_lik = self.forward(obs, return_log=True)
        log_beta = self.backward(obs, return_log=True)

        # Calcular log_gamma: log(alpha * beta) - log_lik
        log_gamma = log_alpha + log_beta - log_lik

        if return_log:
            return log_gamma
        else:
            # Exponenciar y normalizar para evitar pequeños errores numéricos
            gamma = np.exp(log_gamma)
            gamma /= gamma.sum(axis=1, keepdims=True)
            return gamma
    
    # ------------------------------------------------------------------

    def forward_backward_teorico(self, obs):
        """
        Versión teórica del algoritmo Forward-Backward (sin logaritmos).
        Calcula las probabilidades suavizadas gamma[t, j] = P(estado_t = j | todas las observaciones, modelo).

        Utiliza las versiones teóricas de forward y backward (con bucles explícitos)
        para obtener alpha y beta, y luego combina:
            gamma[t, j] = alpha[t, j] * beta[t, j] / P(obs)
        donde P(obs) es la verosimilitud obtenida de forward.

        Args:
            obs (array-like of int): Secuencia de T observaciones.
                Cada elemento debe ser un índice entero en [0, n_obs).

        Returns:
            gamma (ndarray): shape (T, n_estados). Probabilidades suavizadas gamma_t(j).

        Raises:
            TypeError:  Si obs no es un array 1-D de enteros.
            ValueError: Si alguna observación está fuera de [0, n_obs).

        Advertencia:
            Para secuencias largas, los valores pueden sufrir underflow.
            En producción se recomienda usar la versión logarítmica `forward_backward`.

        Ejemplo:
            >>> import numpy as np
            >>> A  = np.array([[0.7, 0.3], [0.4, 0.6]])
            >>> B  = np.array([[0.9, 0.1], [0.2, 0.8]])
            >>> pi = np.array([0.6, 0.4])
            >>> hmm = HMM(2, 2, A=A, B=B, pi=pi)
            >>> gamma = hmm.forward_backward_teorico([0, 0, 1, 0])
            >>> print(gamma.shape)
            (4, 2)
            >>> # Las filas deben sumar 1 (aproximadamente)
            >>> print(np.allclose(gamma.sum(axis=1), 1.0))
            True
        """
        obs = np.asarray(obs, dtype=int)

        if obs.ndim != 1:
            raise TypeError("obs debe ser un array 1-D de índices enteros.")
        if obs.size > 0 and (obs.min() < 0 or obs.max() >= self.n_obs):
            raise ValueError(
                f"Todas las observaciones deben estar en [0, {self.n_obs}). "
                f"Se encontró rango [{obs.min()}, {obs.max()}].")

        T = len(obs)
        if T == 0:
            return np.empty((0, self.n_estados))

        # Obtener alpha y verosimilitud usando forward_teorico
        alpha, prob = self.forward_teorico(obs)

        # Obtener beta usando backward_teorico
        beta = self.backward_teorico(obs)

        # Calcular gamma
        gamma = alpha * beta / prob  # prob es un escalar

        return gamma

    # ------------------------------------------------------------------

    def baum_welch(self, obs, max_iter=100, tol=1e-6, return_log_history=False):
        """
        Algoritmo Baum-Welch (EM) para estimar los parámetros del modelo (A, B, pi)
        a partir de una secuencia de observaciones.

        Utiliza las versiones logarítmicas de forward y backward para mantener
        estabilidad numérica. En cada iteración se calculan las probabilidades
        suavizadas gamma (P(estado_t = i | obs)) y xi (P(estado_t = i, estado_{t+1} = j | obs)),
        y se reestiman los parámetros mediante las fórmulas:

            new_pi[i]   = gamma[0, i]
            new_A[i, j] = sum_t xi[t, i, j]  /  sum_t gamma[t, i]
            new_B[i, k] = sum_t gamma[t, i] * I(obs[t] == k)  /  sum_t gamma[t, i]

        El algoritmo itera hasta que el cambio relativo en la log-verosimilitud
        es menor que `tol` o se alcanza `max_iter`.

        Args:
            obs                (array-like of int): Secuencia de T observaciones.
                Cada elemento debe ser un índice entero en [0, n_obs).
            max_iter           (int):   Número máximo de iteraciones EM.
            tol                (float): Tolerancia para convergencia (cambio relativo en log-verosimilitud).
            return_log_history (bool):  Si True, devuelve además la lista de log-verosimilitudes por iteración.

        Returns:
            Si return_log_history=False:
                log_likelihood (float): Log-verosimilitud final del modelo.
            Si return_log_history=True:
                log_likelihood   (float):        Log-verosimilitud final.
                log_lik_history  (list of float): Log-verosimilitud de cada iteración.

        Raises:
            TypeError:  Si obs no es un array 1-D de enteros.
            ValueError: Si alguna observación está fuera de [0, n_obs).
            ValueError: Si la secuencia de observaciones está vacía.

        Ejemplo:
            >>> import numpy as np
            >>> A  = np.array([[0.7, 0.3], [0.4, 0.6]])
            >>> B  = np.array([[0.9, 0.1], [0.2, 0.8]])
            >>> pi = np.array([0.6, 0.4])
            >>> hmm = HMM(2, 2, A=A, B=B, pi=pi)
            >>> obs = [0, 0, 1, 0, 0, 1, 0, 0]
            >>> logL_inicial, _ = hmm.forward(obs, return_log=True)
            >>> logL_final = hmm.baum_welch(obs, max_iter=50)
            >>> print(logL_final > logL_inicial)  # el modelo debe mejorar
            True
        """
        obs = np.asarray(obs, dtype=int)

        if obs.ndim != 1:
            raise TypeError("obs debe ser un array 1-D de índices enteros.")
        if obs.size > 0 and (obs.min() < 0 or obs.max() >= self.n_obs):
            raise ValueError(
                f"Todas las observaciones deben estar en [0, {self.n_obs}). "
                f"Se encontró rango [{obs.min()}, {obs.max()}].")
        if obs.size == 0:
            raise ValueError("La secuencia de observaciones no puede estar vacía.")

        T = len(obs)
        J = self.n_estados
        K = self.n_obs

        # Matriz indicadora: obs_mask[t, k] = 1 si obs[t] == k, 0 en otro caso.  shape (T, K)
        # Precomputada fuera del loop porque obs no cambia entre iteraciones.
        obs_mask = (obs[:, np.newaxis] == np.arange(K))  # (T, K)

        log_lik_history = []

        for _ in range(max_iter):

            # ----------------------------------------------------------
            # E-step: calcular log_alpha, log_beta, log_gamma y log_xi
            # ----------------------------------------------------------

            log_alpha, log_lik = self.forward(obs, return_log=True)   # (T, J)
            log_beta            = self.backward(obs, return_log=True)  # (T, J)

            # log_gamma[t, j] = log P(estado_t = j | obs, modelo)
            # = log_alpha[t, j] + log_beta[t, j] - log P(obs)
            log_gamma = log_alpha + log_beta - log_lik  # (T, J)

            # log_xi[t, i, j] = log P(estado_t = i, estado_{t+1} = j | obs, modelo)
            # = log_alpha[t, i] + log_A[i, j] + log_emit[t+1, j] + log_beta[t+1, j] - log_lik
            #
            # Broadcasting sobre (T-1, J, J):
            #   log_alpha[:-1]  → (T-1, J, 1)
            #   self._log_A     → (J, J)
            #   log_emit[1:]    → (T-1, 1, J)   precomputado desde self._log_B actualizado
            #   log_beta[1:]    → (T-1, 1, J)
            log_emit = self._log_B[:, obs].T  # (T, J) — refleja self._log_B de la iteración actual
            log_xi = (log_alpha[:-1, :, np.newaxis] +
                      self._log_A +
                      log_emit[1:, np.newaxis, :] +
                      log_beta[1:, np.newaxis, :] - log_lik)  # (T-1, J, J)

            # ----------------------------------------------------------
            # M-step: reestimar parámetros en escala lineal
            # ----------------------------------------------------------

            # Exponenciar es seguro aquí: log_lik normaliza log_gamma y log_xi,
            # manteniéndolas en rango numérico estable.
            gamma = np.exp(log_gamma)  # (T, J)
            xi    = np.exp(log_xi)     # (T-1, J, J)

            # Reestimar pi: distribución inicial = gamma en t=0.
            new_pi = gamma[0].copy()
            new_pi /= new_pi.sum()  # Normalizar para absorber errores de redondeo.

            # ----------------------------------------------------------------------
            # Reestimación de la matriz de transición A
            # ----------------------------------------------------------------------
            # new_A[i, j] = sum_t xi[t, i, j] / sum_t gamma[t, i]   (t = 0..T-2)
            #
            # Si un estado i nunca es visitado en los primeros T-1 pasos,
            # el denominador es cero y la expresión queda indeterminada.
            # En ese caso, no hay información en los datos para estimar las
            # transiciones desde i. Para mantener la estocasticidad del modelo
            # y permitir que el estado pueda ser utilizado en futuras iteraciones
            # o en inferencia, se asigna una distribución uniforme a toda la fila.
            #
            # Esta elección es neutral (no informativa) y evita sesgos;
            # es equivalente a un suavizado de Laplace con pseudocuento 1/J.
            # En el contexto de NLP, donde los estados representan etiquetas
            # lingüísticas, esta práctica es común y robusta, ya que permite
            # que el modelo generalice a secuencias que contengan etiquetas
            # no vistas durante el entrenamiento.
            # ----------------------------------------------------------------------

            # Sumar xi a lo largo del tiempo → numerador (J, J)
            new_A = xi.sum(axis=0).copy()

            # Sumar gamma para t=0..T-2 → denominador por estado (J,)
            denom = gamma[:-1].sum(axis=0)

            # Dividir solo donde denom > 0
            mask = denom > 0
            new_A[mask] /= denom[mask, np.newaxis]

            # Para estados no visitados (denom == 0), asignar distribución uniforme
            new_A[~mask] = 1.0 / J

            # Normalizar filas por si acaso (aunque ya deberían sumar 1, excepto errores de redondeo)
            new_A /= new_A.sum(axis=1, keepdims=True)

            # ----------------------------------------------------------------------
            # Reestimación de la matriz de emisión B
            # ----------------------------------------------------------------------
            # new_B[i, k] = sum_t gamma[t, i] * I(obs[t] == k) / sum_t gamma[t, i]
            #
            # Si un estado i nunca es visitado (sum_t gamma[t, i] = 0),
            # el denominador es cero y no hay información para estimar sus emisiones.
            # Se asigna una distribución uniforme sobre las observaciones para
            # mantener la estocasticidad y permitir que el estado pueda ser utilizado
            # en el futuro. Esto es análogo al suavizado de Laplace con pseudocuento 1/K.
            # ----------------------------------------------------------------------

            new_B = (gamma.T @ obs_mask).copy()                 # numerador (J, K)
            denom_B = gamma.sum(axis=0)                # denominador por estado (J,)

            # Máscara para estados con denominador positivo
            mask_B = denom_B > 0

            # Dividir solo donde hay datos
            new_B[mask_B] /= denom_B[mask_B, np.newaxis]

            # Estados no visitados reciben distribución uniforme
            new_B[~mask_B] = 1.0 / K

            # Normalizar filas por si acaso (elimina errores de redondeo)
            new_B /= new_B.sum(axis=1, keepdims=True)

            # Actualizar parámetros y sus logaritmos precomputados.
            self.A   = new_A
            self.B   = new_B
            self.pi  = new_pi
            self._log_A  = np.log(self.A  + self._EPS)
            self._log_B  = np.log(self.B  + self._EPS)
            self._log_pi = np.log(self.pi + self._EPS)

            log_lik_history.append(log_lik)

            # ----------------------------------------------------------
            # Verificar convergencia: cambio relativo en log-verosimilitud.
            # ----------------------------------------------------------
            if len(log_lik_history) > 1:
                delta = abs(log_lik - log_lik_history[-2])
                if delta < tol * abs(log_lik_history[-2]):
                    break

        if return_log_history:
            return log_lik_history[-1], log_lik_history
        else:
            return log_lik_history[-1]
        
    # ------------------------------------------------------------------

    def baum_welch_mult(self, obs_concatenated, lengths, max_iter=100, tol=1e-6, return_log_history=False):
        """
        Algoritmo Baum-Welch (EM) para múltiples secuencias de observaciones.

        Extiende `baum_welch` al caso de varias secuencias independientes.
        Las estadísticas suficientes (gamma, xi) se acumulan sobre todas las
        secuencias en el E-step, y los parámetros se reestiman conjuntamente
        en el M-step. Esto es equivalente a maximizar la log-verosimilitud
        total sum_s log P(obs_s | modelo).

        Las fórmulas de reestimación son:

            new_pi[i]   = sum_s gamma_s[0, i]  /  S
            new_A[i, j] = sum_s sum_t xi_s[t, i, j]  /  sum_s sum_t gamma_s[t, i]
            new_B[i, k] = sum_s sum_t gamma_s[t, i] * I(obs_s[t] == k)  /  sum_s sum_t gamma_s[t, i]

        donde S es el número de secuencias y el índice t recorre cada secuencia s.

        Args:
            obs_concatenated (array-like of int): Array 1-D con todas las observaciones
                concatenadas. Cada elemento debe ser un índice entero en [0, n_obs).
            lengths          (array-like of int): Longitudes de cada secuencia individual.
                La suma debe coincidir con len(obs_concatenated).
            max_iter         (int):   Número máximo de iteraciones EM.
            tol              (float): Tolerancia para convergencia (cambio relativo en
                log-verosimilitud total).
            return_log_history (bool): Si True, devuelve además la lista de
                log-verosimilitudes totales por iteración.

        Returns:
            Si return_log_history=False:
                log_likelihood (float): Log-verosimilitud total final, sum_s log P(obs_s | modelo).
            Si return_log_history=True:
                log_likelihood   (float):        Log-verosimilitud total final.
                log_lik_history  (list of float): Log-verosimilitud total de cada iteración.

        Raises:
            TypeError:  Si obs_concatenated no es un array 1-D de enteros.
            ValueError: Si alguna observación está fuera de [0, n_obs).
            ValueError: Si obs_concatenated está vacío o la suma de lengths no coincide con su longitud.
            ValueError: Si alguna longitud en lengths es menor o igual a cero.

        Ejemplo:
            >>> import numpy as np
            >>> A  = np.array([[0.7, 0.3], [0.4, 0.6]])
            >>> B  = np.array([[0.9, 0.1], [0.2, 0.8]])
            >>> pi = np.array([0.6, 0.4])
            >>> hmm = HMM(2, 2, A=A, B=B, pi=pi)
            >>> obs = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0])
            >>> lengths = [4, 3, 3]
            >>> logL_inicial = sum(hmm.forward(obs[s:s+l], return_log=True)[1]
            ...                   for s, l in zip(np.cumsum([0]+lengths[:-1]), lengths))
            >>> logL_final = hmm.baum_welch_mult(obs, lengths, max_iter=50)
            >>> print(logL_final > logL_inicial)  # el modelo debe mejorar
            True

        """
        obs = np.asarray(obs_concatenated, dtype=int)
        lengths = np.asarray(lengths, dtype=int)

        if obs.ndim != 1:
            raise TypeError("obs_concatenated debe ser un array 1-D de índices enteros.")
        if obs.size == 0:
            raise ValueError("La secuencia de observaciones no puede estar vacía.")
        if obs.size > 0 and (obs.min() < 0 or obs.max() >= self.n_obs):
            raise ValueError(
                f"Todas las observaciones deben estar en [0, {self.n_obs}). "
                f"Se encontró rango [{obs.min()}, {obs.max()}].")
        if lengths.sum() != obs.size:
            raise ValueError(
                f"La suma de lengths ({lengths.sum()}) debe coincidir con "
                f"la longitud de obs_concatenated ({obs.size}).")
        if (lengths <= 0).any():
            raise ValueError("Todas las longitudes deben ser positivas.")
        
        J = self.n_estados
        K = self.n_obs
        S = len(lengths)  # número de secuencias

        # Matriz indicadora global: obs_mask[t, k] = 1 si obs[t] == k.  shape (total_T, K)
        # Precomputada fuera del loop porque obs no cambia entre iteraciones.
        obs_mask = (obs[:, np.newaxis] == np.arange(K))  # (total_T, K)

        # Índices de inicio de cada secuencia (precomputados fuera del loop).
        starts = np.concatenate([[0], np.cumsum(lengths[:-1])])

        log_lik_history = []

        for _ in range(max_iter):

            # ----------------------------------------------------------
            # E-step: acumular estadísticas sobre todas las secuencias.
            # ----------------------------------------------------------

            # Acumuladores del M-step: se suman las contribuciones de cada secuencia.
            A_num        = np.zeros((J, J))  # numerador de new_A
            A_den        = np.zeros(J)       # denominador de new_A
            B_num        = np.zeros((J, K))  # numerador de new_B
            B_den        = np.zeros(J)       # denominador de new_B
            pi_num       = np.zeros(J)       # acumulador para new_pi
            total_log_lik = 0.0

            for start, length in zip(starts, lengths):
                end = start + length
                seq = obs[start:end]  # subsecuencia de esta secuencia
                T   = length

                log_alpha, log_lik_seq = self.forward(seq, return_log=True)   # (T, J)
                log_beta               = self.backward(seq, return_log=True)  # (T, J)
                total_log_lik         += log_lik_seq

                # log_gamma[t, j] = log P(estado_t = j | seq, modelo)
                log_gamma = log_alpha + log_beta - log_lik_seq  # (T, J)
                gamma     = np.exp(log_gamma)                   # (T, J)

                # log_xi solo si la secuencia tiene al menos 2 pasos.
                if T >= 2:
                    log_emit_seq = self._log_B[:, seq].T  # (T, J) — usa _log_B actualizado
                    log_xi = (log_alpha[:-1, :, np.newaxis] +
                            self._log_A +
                            log_emit_seq[1:, np.newaxis, :] +
                            log_beta[1:, np.newaxis, :] - log_lik_seq)  # (T-1, J, J)
                    xi = np.exp(log_xi)  # (T-1, J, J)

                    A_num += xi.sum(axis=0)          # acumular numerador de A:   (J, J)
                    A_den += gamma[:-1].sum(axis=0)  # acumular denominador de A: (J,)

                pi_num += gamma[0]                           # estado inicial de esta secuencia
                B_num  += gamma.T @ obs_mask[start:end]     # acumular numerador de B:   (J, K)
                B_den  += gamma.sum(axis=0)                 # acumular denominador de B: (J,)

            # ----------------------------------------------------------
            # M-step: reestimar parámetros con estadísticas acumuladas.
            # ----------------------------------------------------------

            # Exponenciar es seguro aquí: log_lik_seq normaliza log_gamma y log_xi
            # en cada secuencia, manteniéndolas en rango numérico estable.

            # Reestimar pi: promedio de los estados iniciales sobre las S secuencias.
            new_pi = pi_num / S
            new_pi /= new_pi.sum()  # Normalizar para absorber errores de redondeo.

            # Reestimar A: cociente de acumuladores — estados no visitados → uniforme.
            new_A  = A_num.copy()
            mask_A = A_den > 0
            new_A[ mask_A] /= A_den[mask_A, np.newaxis]
            new_A[~mask_A]  = 1.0 / J
            new_A /= new_A.sum(axis=1, keepdims=True)  # Normalizar filas.

            # Reestimar B: cociente de acumuladores — estados no visitados → uniforme.
            new_B  = B_num.copy()
            mask_B = B_den > 0
            new_B[ mask_B] /= B_den[mask_B, np.newaxis]
            new_B[~mask_B]  = 1.0 / K
            new_B /= new_B.sum(axis=1, keepdims=True)  # Normalizar filas.

            # Actualizar parámetros y sus logaritmos precomputados.
            self.A   = new_A
            self.B   = new_B
            self.pi  = new_pi
            self._log_A  = np.log(self.A  + self._EPS)
            self._log_B  = np.log(self.B  + self._EPS)
            self._log_pi = np.log(self.pi + self._EPS)

            log_lik_history.append(total_log_lik)

            # ----------------------------------------------------------
            # Verificar convergencia: cambio relativo en log-verosimilitud total.
            # ----------------------------------------------------------
            if len(log_lik_history) > 1:
                delta = abs(total_log_lik - log_lik_history[-2])
                if delta < tol * abs(log_lik_history[-2]):
                    break

        if return_log_history:
            return log_lik_history[-1], log_lik_history
        else:
            return log_lik_history[-1]

    # ------------------------------------------------------------------

    def baum_welch_teorico(self, obs, max_iter=100, tol=1e-6, return_log_history=False):
        """
        Versión teórica del algoritmo Baum-Welch (sin logaritmos).
        Código útil para la comprensión teórica del algoritmo EM en HMM.
        Sigue fielmente las expresiones matemáticas de las ecuaciones de reestimación.

        Dada una secuencia de observaciones, estima los parámetros del modelo
        (A, B, pi) mediante el algoritmo de Expectation-Maximization (EM).
        En cada iteración se calculan las probabilidades suavizadas
        gamma (P(estado_t = i | obs)) y xi (P(estado_t = i, estado_{t+1} = j | obs))
        utilizando las versiones teóricas de forward y backward, y luego se
        reestiman los parámetros con las fórmulas:

            new_pi[i]   = gamma[0, i]
            new_A[i, j] = sum_t xi[t, i, j]  /  sum_t gamma[t, i]   (t = 0..T-2)
            new_B[i, k] = sum_t gamma[t, i] * I(obs[t] == k)  /  sum_t gamma[t, i]

        El algoritmo itera hasta que el cambio relativo en la log-verosimilitud
        es menor que `tol` o se alcanza `max_iter`.

        Args:
            obs                (array-like of int): Secuencia de T observaciones.
                Cada elemento debe ser un índice entero en [0, n_obs).
            max_iter           (int):   Número máximo de iteraciones EM.
            tol                (float): Tolerancia para convergencia (cambio relativo en log-verosimilitud).
            return_log_history (bool):  Si True, devuelve además la lista de log-verosimilitudes por iteración.

        Returns:
            Si return_log_history=False:
                log_likelihood (float): Log-verosimilitud final del modelo.
            Si return_log_history=True:
                log_likelihood   (float):        Log-verosimilitud final.
                log_lik_history  (list of float): Log-verosimilitud de cada iteración.

        Raises:
            TypeError:  Si obs no es un array 1-D de enteros.
            ValueError: Si alguna observación está fuera de [0, n_obs).
            ValueError: Si la secuencia de observaciones está vacía.

        Advertencia:
            Esta versión trabaja en espacio lineal y puede sufrir underflow
            numérico para secuencias largas o modelos con muchas iteraciones.
            En producción se recomienda usar la versión logarítmica `baum_welch`.

        Ejemplo:
            >>> import numpy as np
            >>> A  = np.array([[0.7, 0.3], [0.4, 0.6]])
            >>> B  = np.array([[0.9, 0.1], [0.2, 0.8]])
            >>> pi = np.array([0.6, 0.4])
            >>> hmm = HMM(2, 2, A=A, B=B, pi=pi)
            >>> obs = [0, 0, 1, 0, 0, 1, 0, 0]
            >>> logL_inicial, _ = hmm.forward(obs, return_log=True)
            >>> logL_final = hmm.baum_welch_teorico(obs, max_iter=20)
            >>> print(logL_final > logL_inicial)
            True
        """
        obs = np.asarray(obs, dtype=int)

        if obs.ndim != 1:
            raise TypeError("obs debe ser un array 1-D de índices enteros.")
        if obs.size > 0 and (obs.min() < 0 or obs.max() >= self.n_obs):
            raise ValueError(
                f"Todas las observaciones deben estar en [0, {self.n_obs}). "
                f"Se encontró rango [{obs.min()}, {obs.max()}].")
        if obs.size == 0:
            raise ValueError("La secuencia de observaciones no puede estar vacía.")

        T = len(obs)
        J = self.n_estados
        K = self.n_obs

        # Matriz indicadora para facilitar los cálculos de new_B
        obs_mask = (obs[:, np.newaxis] == np.arange(K))  # (T, K)

        log_lik_history = []

        for _ in range(max_iter):

            # ---------- E-step (teórico) ----------
            alpha, prob = self.forward_teorico(obs)   # alpha: (T, J), prob: escalar
            beta        = self.backward_teorico(obs)  # beta:  (T, J)

            # gamma[t, j] = P(estado_t = j | obs)
            gamma = alpha * beta / prob  # (T, J)

            # xi[t, i, j] = P(estado_t = i, estado_{t+1} = j | obs)  para t = 0..T-2
            xi = np.zeros((T-1, J, J)) if T > 1 else np.zeros((0, J, J))
            for t in range(T-1):
                for i in range(J):
                    for j in range(J):
                        xi[t, i, j] = (alpha[t, i] * self.A[i, j] *
                                        self.B[j, obs[t+1]] * beta[t+1, j]) / prob

            # ---------- M-step (teórico) ----------
            # Reestimar pi
            new_pi = gamma[0].copy()
            new_pi /= new_pi.sum()   # normalizar por si acaso

            # Reestimar A
            new_A = np.zeros((J, J))
            # Denominador por estado i: suma de gamma[t, i] para t=0..T-2
            denom_A = gamma[:-1].sum(axis=0) if T > 1 else np.zeros(J)
            for i in range(J):
                if denom_A[i] > 0:
                    for j in range(J):
                        numer = xi[:, i, j].sum() if T > 1 else 0.0
                        new_A[i, j] = numer / denom_A[i]
                else:
                    # Estado i no visitado: distribución uniforme
                    new_A[i, :] = 1.0 / J
            # Normalizar filas por si acaso (aunque la fórmula ya las normaliza)
            new_A /= new_A.sum(axis=1, keepdims=True)

            # Reestimar B
            new_B = np.zeros((J, K))
            denom_B = gamma.sum(axis=0)  # (J,)
            for i in range(J):
                if denom_B[i] > 0:
                    for k in range(K):
                        numer = np.sum(gamma[:, i] * obs_mask[:, k])
                        new_B[i, k] = numer / denom_B[i]
                else:
                    new_B[i, :] = 1.0 / K
            new_B /= new_B.sum(axis=1, keepdims=True)

            # Actualizar parámetros del modelo
            self.A = new_A
            self.B = new_B
            self.pi = new_pi

            self._log_A  = np.log(self.A  + self._EPS)
            self._log_B  = np.log(self.B  + self._EPS)
            self._log_pi = np.log(self.pi + self._EPS)

            # Se usa log-verosimilitud (monótona) para el criterio de convergencia (aunque el algorítmo sea teórico),
            # evitando underflow de la probabilidad lineal en secuencias largas.
            # (volvemos a calcular forward para obtener la nueva prob)
            _, prob_new = self.forward_teorico(obs)
            log_lik = np.log(prob_new + self._EPS)   # proteger log(0)
            log_lik_history.append(log_lik)

            # Verificar convergencia
            if len(log_lik_history) > 1:
                delta = abs(log_lik - log_lik_history[-2])
                if delta < tol * abs(log_lik_history[-2]):
                    break

        if return_log_history:
            return log_lik_history[-1], log_lik_history
        else:
            return log_lik_history[-1]
    
    # ------------------------------------------------------------------
    