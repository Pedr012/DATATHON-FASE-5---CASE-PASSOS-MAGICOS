import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Sistema de Predição de Risco Educacional | PEDE",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def carregar_modelo():
    """Carrega o modelo treinado e seus artefatos."""
    caminho_modelo = Path(__file__).resolve().parent / "modelo_risco_defasagem.pkl"
    if not caminho_modelo.exists():
        st.error(
            "⚠️ Arquivo modelo_risco_defasagem.pkl não encontrado. "
            "Execute o notebook de análise para gerar o modelo."
        )
        st.stop()
    return joblib.load(caminho_modelo)


modelo_dados = carregar_modelo()


def validar_artefato(artefato):
    """Valida a estrutura do modelo carregado."""
    chaves_obrigatorias = [
        "pipeline_inferencia",
        "schema_entrada",
        "features",
        "feature_rules",
        "imputer",
        "label_encoders",
    ]
    faltantes = [chave for chave in chaves_obrigatorias if chave not in artefato]
    if faltantes:
        st.error(
            f"⚠️ Modelo incompatível. Chaves ausentes: {faltantes}. "
            "Regenere o modelo executando o notebook de análise."
        )
        st.stop()

    regras_obrigatorias = {"variancia_ddof", "desempenho_equilibrado_threshold", "notas_baixas_threshold"}
    if not regras_obrigatorias.issubset(set(artefato["feature_rules"].keys())):
        st.error(
            "⚠️ Modelo incompatível. Parâmetros de features ausentes."
        )
        st.stop()


validar_artefato(modelo_dados)

pipeline_inferencia = modelo_dados["pipeline_inferencia"]
features = modelo_dados["features"]
label_encoders = modelo_dados["label_encoders"]
feature_rules = modelo_dados["feature_rules"]
imputer = modelo_dados["imputer"]
calibrador_isotonic = modelo_dados.get("calibrador_isotonic")
feature_importance = modelo_dados.get("feature_importance")
metricas_teste = modelo_dados.get("metricas_teste", {})
threshold_risco = float(metricas_teste.get("optimal_threshold", 0.5))
if not np.isfinite(threshold_risco):
    threshold_risco = 0.5
threshold_risco = float(np.clip(threshold_risco, 0.0, 1.0))

# Header
st.title("Sistema de Predição de Risco Educacional - PEDE")
st.markdown(
    "Sistema preditivo baseado em **machine learning** para identificação precoce de alunos em risco de defasagem escolar. "
    "Desenvolvido com rigor científico usando dados longitudinais do programa PEDE (Passos Mágicos)."
)
roc_auc = metricas_teste.get("roc_auc", "N/A")
roc_auc_formatted = f"{roc_auc:.3f}" if isinstance(roc_auc, (int, float)) else str(roc_auc)
st.caption(f"ROC-AUC: {roc_auc_formatted} | Features: {len(features)}")
st.divider()


def mapear_genero_para_encoder(valor_extenso, encoder):
    classes = [str(c) for c in encoder.classes_]
    classes_lower = [c.lower() for c in classes]

    candidatos = {
        "Masculino": ["M", "Masculino", "masculino", "Male", "male"],
        "Feminino": ["F", "Feminino", "feminino", "Female", "female"],
    }

    for candidato in candidatos[valor_extenso]:
        if candidato in classes:
            return candidato
        if candidato.lower() in classes_lower:
            idx = classes_lower.index(candidato.lower())
            return classes[idx]

    return None


def _normalizar_acento(s):
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii").lower()


def mapear_valor_para_encoder(valor, encoder):
    if pd.isna(valor):
        return None
    classes = [str(c) for c in encoder.classes_]
    valor_limpo = str(valor).strip()
    if valor_limpo in classes:
        return valor_limpo
    classes_lower = [c.lower() for c in classes]
    if valor_limpo.lower() in classes_lower:
        return classes[classes_lower.index(valor_limpo.lower())]
    # Normalização de acentos (ex: 'Ágata' → 'agata' == 'Agata' no encoder)
    classes_norm = [_normalizar_acento(c) for c in classes]
    valor_norm = _normalizar_acento(valor_limpo)
    if valor_norm in classes_norm:
        return classes[classes_norm.index(valor_norm)]
    return None


def preparar_entrada_para_inferencia(df_base):
    colunas_basicas = ['NOTA_PORT', 'NOTA_MAT', 'NOTA_ING', 'GENERO', 'TEMPO_PROGRAMA', 'NUM_AVALIACOES']
    faltantes = [c for c in colunas_basicas if c not in df_base.columns]
    if faltantes:
        raise ValueError(f"Colunas de entrada ausentes: {faltantes}")

    df = df_base[colunas_basicas].copy()

    encoder_genero = label_encoders.get("GENERO")
    if encoder_genero is None:
        raise ValueError("Encoder de gênero não encontrado no artefato.")

    genero_entrada = df["GENERO"].astype(str).str.strip()

    valores_encoder = []
    for genero_ext in genero_entrada.tolist():
        valor = mapear_genero_para_encoder(genero_ext, encoder_genero)
        if valor is None:
            raise ValueError(
                "Não foi possível mapear gênero para o encoder do modelo. "
                f"Classes disponíveis: {list(encoder_genero.classes_)}"
            )
        valores_encoder.append(valor)

    # Substituir GENERO string pela versão codificada (numérica)
    df["GENERO"] = encoder_genero.transform(valores_encoder)

    variancia_ddof = int(feature_rules["variancia_ddof"])
    thr_equilibrio = float(feature_rules["desempenho_equilibrado_threshold"])
    thr_notas_baixas = float(feature_rules["notas_baixas_threshold"])

    df["MEDIA_NOTAS"] = df[["NOTA_PORT", "NOTA_MAT", "NOTA_ING"]].mean(axis=1)
    df["VARIANCIA_NOTAS"] = df[["NOTA_PORT", "NOTA_MAT", "NOTA_ING"]].var(axis=1, ddof=variancia_ddof)
    df["NOTA_MIN"] = df[["NOTA_PORT", "NOTA_MAT", "NOTA_ING"]].min(axis=1)
    df["NOTA_MAX"] = df[["NOTA_PORT", "NOTA_MAT", "NOTA_ING"]].max(axis=1)
    df["DESEMPENHO_EQUILIBRADO"] = (df["VARIANCIA_NOTAS"] < thr_equilibrio).astype(int)
    df["NOTAS_BAIXAS"] = (df["MEDIA_NOTAS"] < thr_notas_baixas).astype(int)
    
    # Processar campos avançados quando disponíveis
    if 'IPP_2023' in features:
        if 'IPP_2023' in df_base.columns:
            df['IPP_2023'] = pd.to_numeric(df_base['IPP_2023'], errors='coerce')
        else:
            df['IPP_2023'] = np.nan

    # Processar IAN_2022
    if 'IAN_2022' in features:
        if 'IAN_2022' in df_base.columns:
            df['IAN_2022'] = pd.to_numeric(df_base['IAN_2022'], errors='coerce')
        else:
            df['IAN_2022'] = np.nan
    
    # Processar IAN_2023
    if 'IAN_2023' in features:
        if 'IAN_2023' in df_base.columns:
            df['IAN_2023'] = pd.to_numeric(df_base['IAN_2023'], errors='coerce')
        else:
            df['IAN_2023'] = np.nan

    if 'INDE__2022' in features:
        origem_inde_2022 = 'INDE_2022' if 'INDE_2022' in df_base.columns else 'INDE__2022'
        if origem_inde_2022 in df_base.columns:
            df['INDE__2022'] = pd.to_numeric(df_base[origem_inde_2022], errors='coerce')
        else:
            df['INDE__2022'] = np.nan

    if 'INDE__2023' in features:
        origem_inde_2023 = 'INDE_2023' if 'INDE_2023' in df_base.columns else 'INDE__2023'
        if origem_inde_2023 in df_base.columns:
            df['INDE__2023'] = pd.to_numeric(df_base[origem_inde_2023], errors='coerce')
        else:
            df['INDE__2023'] = np.nan

    for coluna_pedra in ['PEDRA_2022', 'PEDRA_2023']:
        if coluna_pedra in features:
            encoder_pedra = label_encoders.get(coluna_pedra)
            if coluna_pedra in df_base.columns and encoder_pedra is not None:
                valores_mapeados = df_base[coluna_pedra].map(lambda v: mapear_valor_para_encoder(v, encoder_pedra))
                valores_codificados = []
                for valor in valores_mapeados.tolist():
                    if valor is None:
                        valores_codificados.append(np.nan)
                    else:
                        valores_codificados.append(float(encoder_pedra.transform([valor])[0]))
                df[coluna_pedra] = valores_codificados
            else:
                df[coluna_pedra] = np.nan

    # Completar as demais colunas do modelo
    colunas_historicas = [col for col in features if col not in df.columns]
    for col in colunas_historicas:
        df[col] = np.nan
    
    # Garantir ordem correta das features
    df_final = df[features].copy()

    # Imputar colunas numéricas com o imputer salvo no artefato
    cols_imputer = list(getattr(imputer, "feature_names_in_", []))
    if cols_imputer:
        faltantes_imputer = [c for c in cols_imputer if c not in df_final.columns]
        if faltantes_imputer:
            raise ValueError(f"Features esperadas pelo imputer ausentes: {faltantes_imputer}")

        df_imputadas = pd.DataFrame(
            imputer.transform(df_final[cols_imputer]),
            columns=cols_imputer,
            index=df_final.index,
        )
        df_final.loc[:, cols_imputer] = df_imputadas
    else:
        # Compatibilidade com artefatos antigos
        df_final = pd.DataFrame(imputer.transform(df_final), columns=features, index=df_final.index)

    # Imputação categórica (não há imputer_cat salvo no artefato)
    for coluna_cat in ["GENERO", "PEDRA_2022", "PEDRA_2023"]:
        if coluna_cat in df_final.columns and coluna_cat in label_encoders:
            encoder = label_encoders[coluna_cat]
            classe_padrao = float(encoder.transform([encoder.classes_[0]])[0])
            df_final[coluna_cat] = pd.to_numeric(df_final[coluna_cat], errors="coerce").fillna(classe_padrao)

    # Garantia adicional para evitar NaN residual em produção
    for col in df_final.columns:
        if df_final[col].isna().any():
            serie_numerica = pd.to_numeric(df_final[col], errors="coerce")
            if serie_numerica.notna().any():
                df_final[col] = serie_numerica.fillna(float(serie_numerica.median()))
            else:
                df_final[col] = 0.0

    return df_final


aba1, aba2 = st.tabs(["Predição Individual", "Informações do Modelo"])

with aba1:
    modo_entrada = st.radio(
        "Dados disponíveis",
        ["Básico", "Completo"],
        horizontal=True,
        help="Básico: notas + perfil do aluno. Completo: inclui indicadores PEDE de 2022/2023."
    )

    with st.form("form_predicao"):
        col1, col2 = st.columns(2)

        with col1:
            nota_port = st.number_input("Português", min_value=0.0, max_value=10.0, value=6.0, step=0.1,
                help="Nota de Português do aluno. Valores de 0 a 10.")
            nota_mat = st.number_input("Matemática", min_value=0.0, max_value=10.0, value=6.0, step=0.1,
                help="Nota de Matemática do aluno. Valores de 0 a 10.")
            nota_ing = st.number_input("Inglês", min_value=0.0, max_value=10.0, value=6.0, step=0.1,
                help="Nota de Inglês do aluno. Valores de 0 a 10.")

        with col2:
            genero_extenso = st.selectbox("Gênero", ["Masculino", "Feminino"],
                help="Gênero do aluno.")
            tempo_programa = st.number_input("Tempo no programa (anos)", min_value=0, max_value=20, value=3, step=1,
                help="Quantos anos o aluno participa do programa Passos Mágicos. Valores de 0 a 20.")
            num_avaliacoes = st.number_input("Número de avaliações", min_value=0, max_value=20, value=4, step=1,
                help="Total de avaliações realizadas pelo aluno ao longo do programa. Valores de 0 a 20.")

        if modo_entrada == "Completo":
            st.divider()
            col3, col4 = st.columns(2)
            with col3:
                ipp_2023 = st.number_input("IPP 2023", min_value=0.0, max_value=10.0, value=5.0, step=0.1,
                    help="Indicador Psicopedagógico de 2023. Mede o desenvolvimento socioemocional do aluno. Valores de 0 a 10.")
                ian_2023 = st.number_input("IAN 2023", min_value=0.0, max_value=10.0, value=5.0, step=0.1,
                    help="Indicador de Adequação ao Nível de 2023. Mede se o aluno está no nível de ensino esperado. Valores de 0 a 10.")
                inde_2023 = st.number_input("INDE 2023", min_value=0.0, max_value=10.0, value=6.0, step=0.1,
                    help="Índice de Desenvolvimento Educacional de 2023. Indicador sintético do desempenho geral. Valores de 0 a 10.")
                pedra_2023 = st.selectbox("Pedra 2023", ["Não informado", "Ágata", "Quartzo", "Topázio", "Ametista"],
                    help="Classificação do aluno no programa em 2023. Ágata (iniciante) → Ametista (excelente).")
            with col4:
                ian_2022 = st.number_input("IAN 2022", min_value=0.0, max_value=10.0, value=5.0, step=0.1,
                    help="Indicador de Adequação ao Nível de 2022. Valores de 0 a 10.")
                inde_2022 = st.number_input("INDE 2022", min_value=0.0, max_value=10.0, value=5.5, step=0.1,
                    help="Índice de Desenvolvimento Educacional de 2022. Valores de 0 a 10.")
                pedra_2022 = st.selectbox("Pedra 2022", ["Não informado", "Ágata", "Quartzo", "Topázio", "Ametista"],
                    help="Classificação do aluno no programa em 2022. Ágata (iniciante) → Ametista (excelente).")
        else:
            ipp_2023 = ian_2022 = ian_2023 = inde_2022 = inde_2023 = pedra_2022 = pedra_2023 = None

        enviar = st.form_submit_button("Analisar Risco", use_container_width=True, type="primary")

    if enviar:
        entrada_dados = {
            "NOTA_PORT": [nota_port],
            "NOTA_MAT": [nota_mat],
            "NOTA_ING": [nota_ing],
            "TEMPO_PROGRAMA": [tempo_programa],
            "NUM_AVALIACOES": [num_avaliacoes],
            "GENERO": [genero_extenso],
        }

        if modo_entrada == "Completo":
            entrada_dados["IPP_2023"] = [ipp_2023]
            entrada_dados["IAN_2022"] = [ian_2022]
            entrada_dados["IAN_2023"] = [ian_2023]
            entrada_dados["INDE_2022"] = [inde_2022]
            entrada_dados["INDE_2023"] = [inde_2023]
            if pedra_2022 != "Não informado":
                entrada_dados["PEDRA_2022"] = [pedra_2022]
            if pedra_2023 != "Não informado":
                entrada_dados["PEDRA_2023"] = [pedra_2023]

        entrada = pd.DataFrame(entrada_dados)

        try:
            X = preparar_entrada_para_inferencia(entrada)
            prob_bruta = float(pipeline_inferencia.predict_proba(X)[0][1])
            if calibrador_isotonic is not None:
                prob_risco = float(calibrador_isotonic.transform([prob_bruta])[0])
            else:
                prob_risco = prob_bruta
            classe = int(prob_risco >= threshold_risco)
        except Exception as erro:
            st.error(f"Erro durante análise: {erro}")
        else:
            st.divider()
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.metric("Probabilidade de Risco", f"{prob_risco * 100:.1f}%")
                st.progress(min(max(prob_risco, 0.0), 1.0))
            with col_r2:
                if classe == 1:
                    st.error("Aluno em Risco")
                else:
                    st.success("Sem Risco Identificado")
            st.caption(f"Limiar de decisão: {threshold_risco:.2f}")

with aba2:
    st.markdown("### Sobre o Modelo")

    # --- Métricas ---
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    metricas_exibir = [
        ("ROC-AUC", "roc_auc", col_m1),
        ("F1-Score", "f1", col_m2),
        ("Recall", "recall", col_m3),
        ("Precisão", "precision", col_m4),
    ]
    for label, key, col in metricas_exibir:
        with col:
            valor = metricas_teste.get(key)
            st.metric(label, f"{float(valor):.3f}" if isinstance(valor, (int, float)) else "—")

    metricas_cv = modelo_dados.get("metricas_cv", {})
    roc_cv = metricas_cv.get("roc_auc_cv_mean")
    roc_std = metricas_cv.get("roc_auc_cv_std")
    if isinstance(roc_cv, (int, float)) and isinstance(roc_std, (int, float)):
        st.caption(f"Validação cruzada 5-fold — ROC-AUC: {roc_cv:.3f} ± {roc_std:.3f}")

    st.divider()

    # --- Importância das variáveis ---
    if isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty:
        st.markdown("**Variáveis mais relevantes**")
        top_10 = feature_importance.head(10).copy()
        top_10["Importância (%)"] = (top_10["Importance"] * 100).round(1)
        st.dataframe(
            top_10[["Feature", "Importância (%)"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Feature": "Variável",
                "Importância (%)": st.column_config.ProgressColumn(
                    "Importância (%)",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
            },
        )

    st.divider()

    # --- Detalhes técnicos ---
    with st.expander("Detalhes técnicos"):
        st.markdown(
            f"**Algoritmo:** Gradient Boosting Classifier  \n"
            f"**Features:** {len(features)}  \n"
            f"**Limiar de decisão:** {threshold_risco:.2f}"
        )

st.divider()
st.caption("Sistema de Predição de Risco Educacional PEDE | Projeto Tech Challenge - Fase 5")
