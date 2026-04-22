# 🤖 PACHABOT - Asistente Inteligente para Orientación Ciudadana

Sistema basado en inteligencia artificial diseñado para brindar orientación ciudadana en trámites municipales, enfocado en la Gerencia de Licencias y Desarrollo Económico (GLDE).

---

## 📌 Descripción

PACHABOT es un asistente virtual que permite a los ciudadanos consultar información sobre trámites municipales de manera rápida, clara y automatizada.

El sistema busca reducir la carga operativa en las áreas administrativas y mejorar la experiencia del ciudadano, proporcionando respuestas sobre:

- requisitos  
- normativas  
- procedimientos  
- validaciones básicas  

Actualmente, el sistema se encuentra en fase de prototipo, utilizando como caso inicial el módulo de **comercio ambulatorio**.

---

## 🎯 Objetivo del proyecto

Desarrollar un asistente inteligente que permita:

- orientar a ciudadanos antes de realizar un trámite  
- reducir consultas presenciales innecesarias  
- mejorar la eficiencia en la atención municipal  
- centralizar información normativa en un sistema accesible  

---

## ❗ Problema que resuelve

En las municipalidades:

- los ciudadanos realizan consultas repetitivas  
- existe desinformación sobre requisitos y normas  
- se generan colas para consultas simples  
- los procesos dependen de atención presencial  

PACHABOT busca automatizar esta orientación inicial.

---

## ⚙️ Funcionalidades actuales

- 💬 Respuesta automática a consultas ciudadanas  
- 📚 Consulta de normativa específica  
- 🔎 Búsqueda de información relevante  
- 🧠 Procesamiento básico de lenguaje natural  
- 📂 Sistema de documentos como fuente de conocimiento  

---

## 🧩 Alcance actual (fase prototipo)

El sistema actualmente trabaja con:

- 📄 Normativa de comercio ambulatorio  
- 🏛️ Ordenanzas municipales específicas  
- 📌 Casos de uso limitados para pruebas  

Este módulo es solo una primera implementación.

---

## 🔮 Escalabilidad del sistema

El proyecto está diseñado para expandirse a:

- licencias de funcionamiento  
- anuncios publicitarios  
- compatibilidad de uso  
- otros trámites municipales  
- múltiples áreas de la municipalidad  

---

## 🏗️ Arquitectura del sistema

El sistema sigue una arquitectura modular tipo asistente inteligente:

- 📡 Canal (Telegram / Web)
- 🧠 Motor de procesamiento (NLP)
- 📚 Base documental (ordenanzas, PDFs)
- 🔎 Motor de búsqueda (TF-IDF / similitud)
- ⚙️ Servicios de respuesta
- 🗂️ Módulos por tipo de trámite

---

## 🧠 Tecnologías utilizadas

### 🐍 Backend
<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
</p>

### 🤖 IA / NLP
<p>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/TF--IDF-000000?style=for-the-badge" />
  <img src="https://img.shields.io/badge/NLP-102230?style=for-the-badge" />
</p>

### 📡 Integración
<p>
  <img src="https://img.shields.io/badge/Telegram_Bot-2CA5E0?style=for-the-badge&logo=telegram" />
</p>

---

## 📂 Estructura del proyecto

```bash
PACHABOT/
├── channels/        # canales de comunicación (Telegram, etc.)
├── services/        # lógica de respuesta
├── tools/           # utilidades y procesamiento
├── memory/          # manejo de contexto
├── api/             # endpoints
├── data/            # documentos y normativa
└── main.py
