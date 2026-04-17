# TinyGPT

Implementación didáctica de un modelo GPT desde cero en PyTorch, desarrollada para el curso **NLP-II** de la [Especialización en Inteligencia Artificial (CEIA)](https://github.com/FIUBA-Posgrado-Inteligencia-Artificial/CEIA-LLMIAG) de FIUBA (Universidad de Buenos Aires).

El proyecto construye un Transformer decoder-only completo sobre el corpus de *tinyshakespeare*, implementa distintas estrategias de decodificación para inferencia, extiende la arquitectura con una capa **Mixture of Experts (MoE)**, y finalmente incorpora un tokenizador **BPE (Byte Pair Encoding)** como mejora sobre el tokenizador por carácter original.

---

## Contenido

- `TinyGPT.ipynb` — notebook principal con la implementación del modelo, las tareas del TP (decodificación + MoE) y visualización de atención.

---

## Arquitectura

TinyGPT es un Transformer decoder-only con las siguientes características:

- **Token embeddings** + **positional embeddings aprendidos** (sumados).
- `n_layer` bloques Transformer apilados, cada uno con:
  - LayerNorm **pre-norm**.
  - Multi-Head Self-Attention causal (con máscara triangular).
  - LayerNorm pre-norm.
  - FeedForward con expansión 4x (o MoE, según configuración).
  - Conexiones residuales en ambos submódulos.
- LayerNorm final + proyección lineal a logits del vocabulario.
- Soporte para **KV-cache** durante generación autoregresiva.

**Configuración por defecto:**

| Parámetro | Valor |
|---|---|
| `block_size` | 32 |
| `batch_size` | 8 |
| `n_embd` | 64 |
| `n_head` | 4 |
| `n_layer` | 2 |
| `dropout` | 0.1 |
| `vocab_size` | ~65 (char-level) |

---

## Tareas del TP

### Task I — Estrategias de decodificación

Se implementa la función `generateV2` que unifica tres algoritmos de decodificación controlados por parámetros:

- **Greedy decoding** (`temperature=0`): toma siempre el token con mayor logit. Determinístico pero propenso a loops repetitivos.
- **Temperature sampling**: reescala los logits antes del softmax. Valores bajos → texto conservador; valores altos → texto más diverso.
- **Top-k filtering**: restringe el sampling a los `k` tokens más probables.
- **Top-p (nucleus) filtering**: restringe el sampling al conjunto mínimo de tokens cuya probabilidad acumulada supere `p`. Adaptativo según la confianza del modelo en cada paso.

Los tres se pueden combinar (por ejemplo top-k aplicado primero y top-p como filtro adicional).

### Task II — Mixture of Experts (MoE)

Se extiende la arquitectura reemplazando el FeedForward vanilla por una capa MoE. Componentes implementados:

- **`Expert`**: un FFN completo (dos capas lineales + ReLU + dropout).
- **`Gate`**: red lineal que asigna un score por experto para cada token.
- **`MoELayer`**: implementa el routing top-k, donde cada token se procesa solo por sus `num_experts_per_token` expertos con mayor score. Las salidas se combinan con pesos softmax normalizados.

Este es el mismo principio que usan modelos como **Mixtral** y **DeepSeek**: escalar la cantidad total de parámetros sin escalar proporcionalmente el costo computacional por forward pass (*sparse computation*).

---

## Extensión: Tokenizador BPE

El tokenizador por carácter es ineficiente: requiere secuencias largas para representar poco contenido semántico y obliga al modelo a "aprender ortografía" desde cero. La extensión entrena un **BPE byte-level** custom con vocabulario de ~2000 tokens usando la librería `tokenizers` de HuggingFace sobre el mismo corpus de Shakespeare.

**Ventajas observadas:**

- Compresión del corpus en ~3-4x (menos tokens para el mismo texto).
- Con `block_size=64` BPE se captura contexto equivalente a ~200-250 caracteres vs 32 del modelo original.
- Calidad de generación notablemente superior: nombres de personajes bien formados, estructura de diálogo coherente, vocabulario shakespeariano reconocible.

La extensión mantiene el código original intacto y agrega la comparación como sección final independiente.

---

## Uso

### Instalación

```bash
# Dependencias base
pip install torch torchvision httpx matplotlib tqdm

# Para la extensión BPE
pip install tokenizers
```

### Entrenamiento

Abrir `TinyGPT.ipynb` y ejecutar las celdas en orden. El notebook:

1. Descarga el corpus de tinyshakespeare.
2. Construye el tokenizador char-level y los dataloaders.
3. Entrena el modelo vanilla (2 epochs).
4. Implementa `generateV2` y prueba las distintas estrategias de decodificación.
5. Extiende el modelo a MoE y re-entrena.
6. Visualiza las matrices de atención de cada capa y head.
7. Ejecuta la extensión BPE.

### Generación

```python
# Greedy
generateV2("To be", temperature=0.0)

# Sampling conservador
generateV2("To be", temperature=0.5, top_k=10, top_p=0.9)

# Sampling creativo
generateV2("First Citizen:", temperature=1.0, top_k=40, top_p=0.95)
```

---

## Observaciones y conclusiones

- **Greedy decoding degenera** en loops repetitivos incluso en prompts simples — ilustración práctica de por qué los LLMs de producción no lo usan por defecto.
- **Temperature + top-p** es la combinación más usada en práctica (GPT, Claude, etc.) y logra el mejor tradeoff diversidad-coherencia.
- **MoE** con 4 expertos y top-1 routing alcanza loss comparable al modelo vanilla con costo computacional por token similar, demostrando el principio de escalado sparse.
- **BPE** mejora notablemente la calidad de generación frente a char-level con el mismo presupuesto de entrenamiento, confirmando que la elección del tokenizador es tan importante como la arquitectura del modelo.
- La implementación usa un loop explícito sobre heads en `MultiHeadAttention` en vez de `F.scaled_dot_product_attention` — menos eficiente pero mucho más pedagógico para entender el mecanismo.

---

## Referencias

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) — arquitectura original del Transformer.
- Radford et al., [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2.
- Holtzman et al., [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751) — nucleus sampling (top-p).
- Shazeer et al., [*Outrageously Large Neural Networks*](https://arxiv.org/abs/1701.06538) — Sparsely-Gated Mixture-of-Experts.
- Karpathy, [char-rnn](https://github.com/karpathy/char-rnn) y [nanoGPT](https://github.com/karpathy/nanoGPT) — inspiración didáctica.
- HuggingFace, [`generate` docs](https://huggingface.co/docs/transformers/main_classes/text_generation) — referencia de estrategias de decodificación.

---

## Autor

**Ing. Luciano Ceballos** — CEIA, FIUBA.
