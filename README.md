# Triage de Soporte TI mediante Clasificación de Imágenes

> Visión IA + FastAPI + Despliegue en CPU
> Proyecto final — Desarrollo de Proyectos de Inteligencia Artificial
> Universidad Autónoma de Occidente

---

## Integrantes

| Nombre | Código |
|---|---|
| Jhojan Alexander Calambas | 22602907 |
| Angelo Parra Cortez | — |
| Oscar Eduardo Portela Ospina | — |
| Sebastian Torres Cabrera | — |

---

## Descripción general

En los help desks y áreas de soporte TI, el proceso de **triage inicial** es una de las actividades más repetitivas y propensas a errores. Los agentes deben leer manualmente descripciones de usuarios, interpretar capturas de pantalla y determinar el tipo de incidente para enrutarlo al equipo correcto. Este flujo es lento y susceptible a errores de enrutamiento que incrementan el **MTTR** (Mean Time To Resolution) y la carga operativa.

Este proyecto propone una solución basada en **Visión por Computadora con Transformers** para automatizar ese triage: el sistema recibe una captura de pantalla de un incidente TI, la clasifica en una categoría de soporte predefinida y sugiere el equipo responsable de atenderlo — sin intervención manual.

---

## Objetivo general

Desarrollar y desplegar una aplicación web y API HTTP, ejecutable en CPU, que clasifique imágenes de incidentes TI en categorías de soporte (triage visual) y entregue predicciones Top-K con probabilidades, alcanzando un **Macro-F1 >= 0.70** sobre el conjunto de evaluación, con manejo de errores, pruebas automatizadas y builds reproducibles mediante Docker.

---

## Objetivos específicos

- **OE-01** — Definir el catálogo de categorías de soporte (mínimo 6) con reglas de enrutamiento por equipo responsable.
- **OE-02** — Construir un dataset de demostración (20–50 imágenes por categoría) con capturas simuladas o públicas, asegurando ausencia de PII.
- **OE-03** — Implementar el módulo de inferencia sobre **DeiT-Tiny** con un clasificador final ajustado a las categorías de soporte.
- **OE-04** — Exponer la solución mediante **FastAPI**: endpoint `POST /api/predict` (Top-K con probabilidad) y `GET /health`.
- **OE-05** — Construir una interfaz web (HTML/Jinja2) para carga de imágenes y visualización del resultado de enrutamiento.
- **OE-06** — Incorporar pruebas automatizadas (mínimo 3) y pipeline CI en GitHub Actions para validación en cada push/PR.
- **OE-07** — Empaquetar con **Docker** para despliegue reproducible y prueba de despliegue en AWS EC2.

---

## Categorías de clasificación

El sistema clasifica imágenes en 8 categorías con enrutamiento sugerido al equipo responsable:

| Categoría | Ejemplos de imagen | Equipo sugerido |
|---|---|---|
| Red / Conectividad | Router, switch, Wi-Fi, errores de conexión | Equipo Redes |
| Acceso / Contraseñas | Pantallas de login, bloqueos, MFA | Soporte N1 / IAM |
| Correo / Office 365 | Outlook, OWA, errores de sincronización | Equipo O365 |
| Impresión / Periféricos | Impresoras, spooler, escáneres | Equipo Periféricos |
| Aplicación / Errores | Stack traces, errores de app, crash windows | Equipo Aplicaciones |
| Hardware / Equipo | PC sin encender, pantallas, componentes | Equipo Hardware |
| VPN / Remoto | Cliente VPN, errores de túnel, RDP | Equipo Seguridad/Redes |
| Otros / No clasifica | Casos raros o baja confianza | Revisión humana |

> **Regla de seguridad:** Si el score de confianza del modelo es menor a **0.40**, el sistema retorna automáticamente la categoría "Otros / Revisión humana", evitando enrutamientos automáticos con baja certeza.

---

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Backend API | FastAPI (async) |
| Frontend | HTML/CSS + Jinja2 templates |
| Modelo de IA | DeiT-Tiny (HuggingFace Transformers + PyTorch CPU) |
| Contenerización | Docker (compatible con CPU, imagen reproducible) |
| CI/CD | GitHub Actions (lint + tests en cada push/PR) |
| Despliegue demo | AWS EC2 (t2.micro / t3.micro) |

### Modelo de IA — DeiT-Tiny

Se utiliza **DeiT-Tiny** (Data-efficient Image Transformers) como modelo base para extracción de características visuales. Sobre él se ajusta un clasificador final entrenado con las categorías de soporte TI. La inferencia se ejecuta completamente en **CPU**, sin requerir GPU, lo que hace el sistema desplegable en instancias de bajo costo.

**Estrategia de respaldo:** Si el modelo primario no alcanza F1 >= 0.70, se contempla el uso de embeddings + clasificadores más simples (SVM o Regresión Logística) y técnicas de aumento de datos.

---

## Endpoints de la API

| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/api/predict` | Recibe una imagen y retorna Top-K categorías con probabilidades y equipo sugerido |
| `GET` | `/health` | Healthcheck del servicio — retorna `{"status": "ok"}` con HTTP 200 |

### Ejemplo de respuesta — `POST /api/predict`

```json
{
  "predictions": [
    {
      "category": "Red / Conectividad",
      "probability": 0.87,
      "team": "Equipo Redes"
    },
    {
      "category": "VPN / Remoto",
      "probability": 0.09,
      "team": "Equipo Seguridad/Redes"
    }
  ],
  "human_review_required": false
}
```

---

## Requisitos funcionales

| ID | Requisito | Criterio de aceptación |
|---|---|---|
| RF-01 | Subir imagen desde el navegador | Acepta imágenes válidas; retorna error controlado para entrada inválida |
| RF-02 | Clasificar en categoría de soporte | Retorna Top-K categorías con probabilidades (K configurable) |
| RF-03 | Sugerir enrutamiento | Retorna equipo recomendado por categoría predicha |
| RF-04 | API de inferencia JSON | `POST /api/predict` responde con predicciones, probabilidades y metadata |
| RF-05 | Healthcheck del servicio | `GET /health` retorna `{status: ok}` con HTTP 200 |

## Requisitos no funcionales

| ID | Categoría | Descripción |
|---|---|---|
| RNF-01 | CPU / Infraestructura | Inferencia sin GPU; compatible con instancias EC2 de bajo costo |
| RNF-02 | Privacidad | Dataset demo sin PII; capturas anonimizadas |
| RNF-03 | Confiabilidad | Respuestas 4xx controladas para entradas inválidas; sin excepciones no manejadas |
| RNF-04 | Calidad | Mínimo 3 pruebas automatizadas y CI activo en cada push/PR |
| RNF-05 | Portabilidad | Dockerfile para ejecución en AWS EC2 / Linux VM sin modificaciones |
| RNF-06 | Rendimiento | Macro-F1 >= 0.70; Top-1 routing accuracy reportado |

---

## Métricas de evaluación

- Accuracy
- Macro-F1 (umbral mínimo: **0.70**)
- Matriz de confusión
- Top-1 y Top-2 routing accuracy

---

## Fuera de alcance

- Resolución automatizada de incidentes (solo triage y enrutamiento).
- Integración con plataformas ITSM en producción (ServiceNow, Jira) — solo demo.
- Autenticación de usuarios o gestión empresarial de accesos.
- Datos reales que contengan información personal identificable (PII).
- Entrenamiento a gran escala o modelos de producción de alto rendimiento.

---

## Configuración del entorno

Este proyecto utiliza **uv** como gestor de dependencias y entornos virtuales.

```bash
# Instalar dependencias
make install

# Ejecutar la aplicación
make start

# Correr los tests
make test
```

### Con Docker

```bash
# Construir la imagen
docker build -t triage-ti .

# Ejecutar el contenedor
docker run -p 8000:8000 triage-ti
```

---

## Metodología

El proyecto sigue una **metodología Ágil** con sprints cortos de una semana, produciendo incrementos funcionales en cada sprint para retroalimentación temprana. Integra principios del Manifiesto Ágil, pipelines de CI/CD con GitHub Actions y prácticas DevOps entre desarrollo e infraestructura.

---

## Licencia

MIT License — ver [LICENSE](LICENSE) para más detalles.
